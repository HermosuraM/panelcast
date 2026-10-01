"""Data-quality anomaly detectors for a multi-source card panel.

Key idea: compare every data source with the *cross-source consensus* on the same day. A COVID lockdown,
Black Friday, or a snowstorm moves all sources together, so the consensus absorbs it; an outage, a unit
error, or a broken descriptor feed moves ONE source away from the others. Each source's log-ratio to the
consensus is then scored against its own trailing baseline with a robust (median / MAD) z-score.

The per-series functions here are pure pandas: Spark computes the consensus ratios, then runs these
functions once per source (or per source x ticker) with `applyInPandas`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
MAD_SCALE = 1.4826  # makes the MAD a consistent estimator of the standard deviation under normality

SOURCE_FLAG_COLUMNS = ["date", "source", "type", "score", "action", "factor"]
BREAK_COLUMNS = ["source", "ticker", "start", "end", "drop"]


def trailing_baseline(x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Median and robust scale of the previous `window` values (the current value is excluded)."""
    prev = x.shift(1)
    min_periods = max(7, window // 2)
    med = prev.rolling(window, min_periods=min_periods).median()
    mad = prev.rolling(window, min_periods=min_periods).apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True)
    return med, MAD_SCALE * mad


def robust_z(x: pd.Series, window: int, floor: float = 0.02) -> pd.Series:
    med, scale = trailing_baseline(x, window)
    return (x - med) / scale.clip(lower=floor)


def consensus_log_ratio(df: pd.DataFrame, value: str, by: list[str]) -> pd.Series:
    """log(value / cross-source median of value) within each `by` group (e.g. date, or date+ticker)."""
    consensus = df.groupby(by)[value].transform("median")
    return np.log((df[value] + EPS) / (consensus + EPS))


def source_flags(
    g: pd.DataFrame, window: int, z_threshold: float, unit_band: tuple[float, float], min_drop: float = 0.4
) -> pd.DataFrame:
    """One source's daily series (date, source, txns, members, lr_rate, lr_ticket) -> flagged days.

    Volume drops must be both statistically extreme (robust z) and large (>= `min_drop`); sources differ
    demographically, so holidays move them by different amounts and small one-day wobbles are expected.
    Spikes are recorded but not excluded: duplicates are already removed in silver.
    """
    g = g.sort_values("date").set_index("date")
    rate_base, rate_scale = trailing_baseline(g["lr_rate"], window)
    z_rate = (g["lr_rate"] - rate_base) / rate_scale.clip(lower=0.02)
    change = np.exp(g["lr_rate"] - rate_base) - 1
    ticket_base, _ = trailing_baseline(g["lr_ticket"], window)
    jump = g["lr_ticket"] - ticket_base
    lo, hi = unit_band
    rows = []
    for date in g.index:
        if g.at[date, "members"] <= 0:
            continue
        if g.at[date, "txns"] == 0:
            rows.append((date, "outage", -99.0, "exclude", 1.0))
        elif lo <= jump[date] <= hi:
            rows.append((date, "unit_error", float(jump[date]), "rescale", 0.01))
        elif -hi <= jump[date] <= -lo:
            rows.append((date, "unit_error", float(jump[date]), "rescale", 100.0))
        elif z_rate[date] <= -z_threshold and change[date] <= -min_drop:
            rows.append((date, "volume_drop", float(z_rate[date]), "exclude", 1.0))
        elif z_rate[date] >= z_threshold and change[date] >= 0.5:
            rows.append((date, "volume_spike", float(z_rate[date]), "flag", 1.0))
    out = pd.DataFrame(rows, columns=["date", "type", "score", "action", "factor"])
    out.insert(1, "source", g["source"].iloc[0] if len(g) else None)
    return out[SOURCE_FLAG_COLUMNS]


def tagging_breaks(
    g: pd.DataFrame,
    baseline_days: int = 56,
    z_threshold: float = 5.0,
    min_drop: float = 0.35,
    persist: int = 7,
    window_days: int = 7,
) -> pd.DataFrame:
    """One (source, ticker) series -> break events, using a count test.

    Input columns: date, source, ticker, txns7 (tagged transactions in the trailing 7 days), members7
    (member-days), c7 (cross-source consensus rate for the ticker). Under "this source behaves like the
    others", txns7 ~ Poisson(members7 * c7 * m), where m is the source's own typical multiplier (trailing
    median). Counts are over-dispersed (a few loyal customers drive volume), so the Pearson residual is
    scaled by a trailing robust dispersion estimate (quasi-Poisson). A break needs a >= `min_drop` shortfall
    with z <= -`z_threshold` on `persist` consecutive days; the multiplier is frozen when it opens so a
    permanent break (the usual case for a descriptor format change) stays open.
    """
    g = g.sort_values("date")
    source, ticker = g["source"].iloc[0], g["ticker"].iloc[0]
    g = g.set_index("date")
    naive = g["members7"] * g["c7"]
    lr = np.log((g["txns7"] + 0.5) / (naive + 0.5))
    base, _ = trailing_baseline(lr, baseline_days)
    expected = naive * np.exp(base)
    pearson = (g["txns7"] - expected) / np.sqrt(expected.clip(lower=1.0))
    _, disp_scale = trailing_baseline(pearson, baseline_days)
    z = pearson / disp_scale.clip(lower=1.0)
    ratio = g["txns7"] / expected
    offset = int(np.ceil(min_drop * window_days)) - 1  # how far the 7-day sum lags an abrupt break
    dates = list(g.index)
    events, streak = [], []
    in_event, frozen, start = False, 0.0, None
    for i, d in enumerate(dates):
        if np.isnan(z[d]) or np.isnan(ratio[d]):
            continue
        if not in_event:
            streak = [*streak, i] if (ratio[d] <= 1 - min_drop and z[d] <= -z_threshold) else []
            if len(streak) >= persist:
                in_event, frozen = True, base[dates[max(0, streak[0] - window_days)]]
                start = dates[max(0, streak[0] - offset)]
        else:
            r = g.at[d, "txns7"] / (naive[d] * np.exp(frozen)) if naive[d] > 0 else np.nan
            if r >= 1 - min_drop / 2:  # recovered (hysteresis)
                events.append((source, ticker, start, dates[i - 1], float(1 - ratio[dates[i - 1]])))
                in_event, streak = False, []
    if in_event:
        last = dates[-1]
        r = g.at[last, "txns7"] / (naive[last] * np.exp(frozen)) if naive[last] > 0 else np.nan
        events.append((source, ticker, start, last, float(1 - r)))
    return pd.DataFrame(events, columns=BREAK_COLUMNS)


def days_to_events(flags: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    """Collapse day-level flags into events (consecutive days of the same source and type)."""
    cols = ["source", "type", "start", "end", "n_days", "action"]
    if flags.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (source, kind), g in flags.sort_values("date").groupby(["source", "type"]):
        dates = pd.to_datetime(g["date"]).tolist()
        start = prev = dates[0]
        for d in dates[1:]:
            if (d - prev).days > max_gap + 1:
                out.append((source, kind, start.date(), prev.date(), (prev - start).days + 1))
                start = d
            prev = d
        out.append((source, kind, start.date(), prev.date(), (prev - start).days + 1))
    ev = pd.DataFrame(out, columns=cols[:-1])
    ev.loc[(ev["type"] == "outage") & (ev["n_days"] > 14), "type"] = "source_dropout"
    ev["action"] = np.select([ev["type"] == "unit_error", ev["type"] == "volume_spike"], ["rescale", "flag"], "exclude")
    return ev
