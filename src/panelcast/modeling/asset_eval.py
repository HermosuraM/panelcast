"""Alternative-data asset evaluation: would this dataset improve revenue nowcasts, and for which tickers?

Every candidate asset is reduced to a fiscal-quarter YoY growth signal per ticker and scored on:
    coverage        tickers and quarters with usable history
    fit             correlation of signal growth with reported revenue growth (with and without COVID)
    incremental     walk-forward skill vs. a naive baseline: 1 - MAE(model) / MAE(naive)
    timeliness      days between the signal being available and the company filing its numbers
    quality         anomaly rate in the raw feed
Assets scored here: the simulated card panel (raked and raw) and REAL English-Wikipedia pageviews.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from panelcast.anomaly.detectors import robust_z
from panelcast.config import Settings
from panelcast.reference.wikipedia import rename_suspects
from panelcast.store import TableStore

log = logging.getLogger(__name__)


def wiki_quarterly(views: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    """Sum daily pageviews (all of a ticker's articles) over each fiscal quarter; YoY log growth."""
    daily = views.groupby(["ticker", "date"], as_index=False)["views"].sum()
    daily["date"] = pd.to_datetime(daily["date"])
    rows = []
    for r in revenue.itertuples():
        d = daily[
            (daily["ticker"] == r.ticker)
            & daily["date"].between(pd.Timestamp(r.period_start), pd.Timestamp(r.period_end))
        ]
        complete = len(d) >= 0.95 * r.period_days
        rows.append(
            {
                "ticker": r.ticker,
                "fiscal_year": r.fiscal_year,
                "fiscal_quarter": r.fiscal_quarter,
                "wiki_views": d["views"].sum() if complete else np.nan,
            }
        )
    q = pd.DataFrame(rows)
    prior = q.assign(fiscal_year=q["fiscal_year"] + 1).rename(columns={"wiki_views": "wiki_prior"})
    q = q.merge(prior, on=["ticker", "fiscal_year", "fiscal_quarter"], how="left")
    q["wiki_yoy"] = np.log(q["wiki_views"] / q["wiki_prior"])
    return q[["ticker", "fiscal_year", "fiscal_quarter", "wiki_yoy"]]


def walk_forward_skill(frame: pd.DataFrame, signal: str, start: str, min_train: int = 12) -> pd.DataFrame:
    """Point-in-time comparison of naive vs. ridge(naive lag + signal + ticker effects)."""
    tickers = sorted(frame["ticker"].unique())
    f = frame.dropna(subset=[signal, "rev_yoy", "rev_yoy_lag1"]).copy()
    f = f[np.isfinite(f[signal])]
    f["nowcast_date"] = pd.to_datetime(f["nowcast_date"])
    f["filed"] = pd.to_datetime(f["filed"])
    out = []
    for when, test in f[f["period_end"] >= pd.Timestamp(start).date()].groupby("nowcast_date"):
        train = f[f["filed"] < when]
        if len(train) < min_train:
            continue

        def design(df):
            onehot = (df["ticker"].to_numpy()[:, None] == np.asarray(tickers)[None, :]).astype(float)
            return np.hstack([df[[signal, "rev_yoy_lag1"]].to_numpy(float), onehot])

        model = Ridge(alpha=1.0).fit(design(train), train["rev_yoy"])
        out.append(test[["ticker", "period_end", "rev_yoy", "rev_yoy_lag1"]].assign(pred=model.predict(design(test))))
    res = pd.concat(out, ignore_index=True)
    res["err_model"] = (res["pred"] - res["rev_yoy"]).abs()
    res["err_naive"] = (res["rev_yoy_lag1"] - res["rev_yoy"]).abs()
    return res


def wiki_spikes(views: pd.DataFrame, z_threshold: float = 8.0, top: int = 10, min_views: int = 1000) -> pd.DataFrame:
    """Largest real attention spikes (robust z of log views vs. a trailing 28-day baseline), per ticker
    with all of its article titles summed so renames do not masquerade as spikes."""
    daily = views.groupby(["ticker", "date"], as_index=False)["views"].sum()
    out = []
    for ticker, g in daily.groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        z = robust_z(np.log1p(g["views"].astype(float)), 28, floor=0.05)
        hits = z[(z >= z_threshold) & (g["views"] >= min_views)]
        out.extend(
            {"ticker": ticker, "date": d, "views": int(g.at[d, "views"]), "z": float(v)} for d, v in hits.items()
        )
    cols = ["ticker", "date", "views", "z"]
    return pd.DataFrame(out, columns=cols).sort_values("z", ascending=False).head(top)


def score_asset(
    name: str,
    frame: pd.DataFrame,
    signal: str,
    availability_lag: int,
    start: str,
    covid: tuple[str, str],
    quality_note: str,
) -> tuple[dict, pd.DataFrame]:
    f = frame.dropna(subset=[signal, "rev_yoy"])
    f = f[np.isfinite(f[signal])]
    pe = pd.to_datetime(f["period_end"])
    ex_covid = f[~pe.between(pd.Timestamp(covid[0]), pd.Timestamp(covid[1]))]
    per_ticker = []
    for t, g in f.groupby("ticker"):
        gx = ex_covid[ex_covid["ticker"] == t]
        per_ticker.append(
            {
                "asset": name,
                "ticker": t,
                "quarters": len(g),
                "corr": g[signal].corr(g["rev_yoy"]) if len(g) >= 6 else np.nan,
                "corr_ex_covid": gx[signal].corr(gx["rev_yoy"]) if len(gx) >= 6 else np.nan,
            }
        )
    per_ticker = pd.DataFrame(per_ticker)
    skill = walk_forward_skill(frame, signal, start)
    by_ticker = skill.groupby("ticker").agg(mae_model=("err_model", "mean"), mae_naive=("err_naive", "mean"))
    by_ticker["skill"] = 1 - by_ticker["mae_model"] / by_ticker["mae_naive"]
    per_ticker = per_ticker.merge(by_ticker["skill"].reset_index(), on="ticker", how="left")
    overall_skill = 1 - skill["err_model"].mean() / skill["err_naive"].mean()
    lead = (
        pd.to_datetime(f["filed"]) - (pd.to_datetime(f["period_end"]) + pd.Timedelta(days=availability_lag))
    ).dt.days
    usable = per_ticker[per_ticker["quarters"] >= 8]
    row = {
        "asset": name,
        "tickers_covered": int(len(usable)),
        "quarters": int(len(f)),
        "first_period": str(pd.to_datetime(f["period_end"]).min().date()),
        "median_corr": float(usable["corr"].median()),
        "median_corr_ex_covid": float(usable["corr_ex_covid"].median()),
        "share_tickers_corr_gt_0_5": float((usable["corr_ex_covid"] > 0.5).mean()),
        "oos_skill_vs_naive": float(overall_skill),
        "share_tickers_improved": float((usable["skill"] > 0).mean()),
        "median_lead_days": float(lead.median()),
        "quality": quality_note,
    }
    row["verdict"] = verdict(row)
    return row, per_ticker


def verdict(row: dict) -> str:
    s, c = row["oos_skill_vs_naive"], row["median_corr_ex_covid"]
    if s >= 0.3 and c >= 0.5:
        return "Strong: production candidate for revenue nowcasts"
    if s >= 0.1:
        return "Moderate: useful as a secondary feature"
    if s > 0:
        return "Weak: marginal incremental value"
    return "Not useful for revenue nowcasting (as tested)"


def evaluate_assets(settings: Settings, store: TableStore) -> pd.DataFrame:
    cfg = settings["nowcast"]
    start, covid = str(cfg["backtest_start"]), tuple(str(x) for x in cfg["covid_window"])
    q = store.read_pandas("gold", "ticker_quarterly").sort_values(["ticker", "period_end"])
    q["rev_yoy_lag1"] = q.groupby("ticker")["rev_yoy"].shift(1)
    q = q[q["complete"] | q["rev_yoy"].notna()]
    views = store.read_pandas("ref", "wiki_pageviews")
    revenue = store.read_pandas("ref", "revenue")
    wiki = wiki_quarterly(views, revenue)
    base = store.read_pandas("ref", "revenue").sort_values(["ticker", "period_end"])
    base = base.merge(
        q[["ticker", "fiscal_year", "fiscal_quarter", "raked_yoy", "raw_yoy", "nowcast_date"]],
        on=["ticker", "fiscal_year", "fiscal_quarter"],
        how="left",
    ).merge(wiki, on=["ticker", "fiscal_year", "fiscal_quarter"], how="left")
    prior = base[["ticker", "fiscal_year", "fiscal_quarter", "revenue"]].assign(fiscal_year=lambda d: d.fiscal_year + 1)
    base = base.merge(prior, on=["ticker", "fiscal_year", "fiscal_quarter"], how="left", suffixes=("", "_prior"))
    base["rev_yoy"] = np.log(base["revenue"] / base["revenue_prior"])
    base["rev_yoy_lag1"] = base.groupby("ticker")["rev_yoy"].shift(1)
    base["nowcast_date"] = pd.to_datetime(base["period_end"]) + pd.Timedelta(days=int(cfg["delivery_lag_days"]))

    events = store.read_pandas("gold", "anomaly_events")
    card_quality = f"{len(events[events['action'] != 'flag'])} data-quality events detected and handled"
    spikes = wiki_spikes(views)
    store.write_pandas(spikes, "gold", "wiki_spikes")
    renamed = rename_suspects(views)
    wiki_quality = (
        f"{renamed['ticker'].nunique()} tickers had article renames (patched by summing old + new "
        f"titles); attention spikes are news-driven, not revenue-driven"
    )

    rows, details = [], []
    for name, signal, lag, note in (
        ("card_panel_raked", "raked_yoy", int(cfg["delivery_lag_days"]), card_quality),
        ("card_panel_raw", "raw_yoy", int(cfg["delivery_lag_days"]), card_quality + "; no normalization"),
        ("wikipedia_pageviews", "wiki_yoy", 1, wiki_quality),
    ):
        row, per_ticker = score_asset(name, base, signal, lag, start, covid, note)
        rows.append(row)
        details.append(per_ticker)
    card = pd.DataFrame(rows)
    store.write_pandas(card, "gold", "asset_scorecard")
    store.write_pandas(pd.concat(details, ignore_index=True), "gold", "asset_ticker_stats")
    log.info("alt-data asset scorecard:\n%s", card.drop(columns=["quality"]).round(3).to_string(index=False))
    return card
