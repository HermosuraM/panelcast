"""Turn real quarterly revenue into a smooth daily "card-visible spend" curve.

Interpolating *cumulative* revenue with a monotone cubic (PCHIP) gives smooth daily rates whose sums
reproduce every fiscal quarter exactly. Day-of-week and holiday factors then reshape spend inside each
quarter, and a renormalization step restores the quarter totals.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

RETAIL = {"retail", "weekend"}
RESTAURANT = {"restaurant", "dinner", "pizza", "coffee", "nightlife"}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def holiday_factor(dates: pd.DatetimeIndex, dow_profile: str) -> np.ndarray:
    """Multiplicative calendar effects by spending category (only shapes spend *within* a quarter)."""
    f = np.ones(len(dates))
    idx = {d.date(): i for i, d in enumerate(dates)}

    def bump(day: dt.date, mult: float) -> None:
        if day in idx:
            f[idx[day]] *= mult

    for year in sorted({d.year for d in dates}):
        thanksgiving = _nth_weekday(year, 11, 3, 4)
        black_friday = thanksgiving + dt.timedelta(days=1)
        super_bowl = _nth_weekday(year, 2, 6, 2 if year >= 2022 else 1)
        mothers_day = _nth_weekday(year, 5, 6, 2)
        christmas, christmas_eve = dt.date(year, 12, 25), dt.date(year, 12, 24)
        if dow_profile in RETAIL:
            bump(thanksgiving, 0.35)
            bump(black_friday, 1.9)
            bump(christmas, 0.1)
            bump(christmas_eve, 1.3)
            for day in range(15, 24):
                bump(dt.date(year, 12, day), 1.25)
        if dow_profile in RESTAURANT:
            bump(thanksgiving, 0.55)
            bump(christmas, 0.35)
            bump(dt.date(year, 7, 4), 0.9)
        if dow_profile == "pizza":
            bump(super_bowl, 1.6)
        if dow_profile == "dinner":
            bump(dt.date(year, 2, 14), 1.35)
            bump(mothers_day, 1.4)
        if dow_profile == "weekend":
            labor_day = _nth_weekday(year, 9, 0, 1)
            memorial_day = (
                _nth_weekday(year, 5, 0, 5) if _nth_weekday(year, 5, 0, 5).month == 5 else _nth_weekday(year, 5, 0, 4)
            )
            for anchor in (labor_day, memorial_day):
                for k in range(-2, 1):
                    bump(anchor + dt.timedelta(days=k), 1.3)
    return f


def daily_revenue(periods: pd.DataFrame, dow: list[float], dow_profile: str) -> pd.DataFrame:
    """periods: contiguous fiscal quarters with columns period_start, period_end, revenue.

    Returns one row per day with `revenue` such that each period's days sum to its reported revenue.
    """
    periods = periods.sort_values("period_start").reset_index(drop=True)
    starts = pd.to_datetime(periods["period_start"])
    ends = pd.to_datetime(periods["period_end"])
    knots = np.concatenate([[starts.iloc[0].value], (ends + pd.Timedelta(days=1)).astype("int64").to_numpy()])
    cum = np.concatenate([[0.0], periods["revenue"].cumsum().to_numpy()])
    day_ns = pd.Timedelta(days=1).value
    x = (knots - knots[0]) / day_ns
    interp = PchipInterpolator(x, cum)

    dates = pd.date_range(starts.iloc[0], ends.iloc[-1], freq="D")
    t = np.arange(len(dates) + 1, dtype=float)
    base = np.diff(interp(t))
    shape = base * np.asarray(dow)[dates.dayofweek] * holiday_factor(dates, dow_profile)

    period_id = np.searchsorted(x[1:], t[:-1], side="right")
    totals = np.bincount(period_id, weights=shape)
    scale = periods["revenue"].to_numpy() / totals
    return pd.DataFrame({"date": dates.date, "revenue": shape * scale[period_id], "period_id": period_id})


def visible_ratio(
    dates: pd.Series,
    periods: pd.DataFrame,
    ratio: float,
    drift: float,
    wobble: float,
    rng: np.random.Generator,
    phi: float = 0.7,
) -> np.ndarray:
    """ratio(t) = ratio0 * exp(drift * years + AR(1) quarterly wobble), linearly interpolated to days."""
    mids = (
        pd.to_datetime(periods["period_start"])
        + (pd.to_datetime(periods["period_end"]) - pd.to_datetime(periods["period_start"])) / 2
    )
    w = np.zeros(len(periods))
    for i in range(1, len(w)):
        w[i] = phi * w[i - 1] + wobble * rng.standard_normal()
    t_days = pd.to_datetime(pd.Series(dates)).astype("int64").to_numpy() / 8.64e13
    m_days = mids.astype("int64").to_numpy() / 8.64e13
    wob = np.interp(t_days, m_days, w)
    years = (t_days - t_days[0]) / 365.25
    return ratio * np.exp(drift * years + wob)
