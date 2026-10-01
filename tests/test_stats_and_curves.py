import datetime as dt

import numpy as np
import pandas as pd

from panelcast.simulate.revenue_curve import daily_revenue, holiday_factor
from panelcast.stats.raking import rake, raked_shares


def test_daily_revenue_reproduces_every_fiscal_quarter():
    periods = pd.DataFrame(
        {
            "period_start": [dt.date(2022, 1, 3), dt.date(2022, 3, 28), dt.date(2022, 6, 20), dt.date(2022, 9, 12)],
            "period_end": [dt.date(2022, 3, 27), dt.date(2022, 6, 19), dt.date(2022, 9, 11), dt.date(2023, 1, 1)],
            "revenue": [1.0e9, 1.1e9, 1.05e9, 1.6e9],  # 12/12/12/16-week quarters
        }
    )
    daily = daily_revenue(periods, [0.9, 0.9, 0.9, 1.0, 1.2, 1.2, 0.9], "pizza")
    sums = daily.groupby("period_id")["revenue"].sum().to_numpy()
    np.testing.assert_allclose(sums, periods["revenue"], rtol=1e-9)
    assert (daily["revenue"] > 0).all()
    assert len(daily) == (periods["period_end"].iloc[-1] - periods["period_start"].iloc[0]).days + 1


def test_black_friday_bump_for_retail_only():
    dates = pd.date_range("2024-11-25", "2024-12-01")
    retail, coffee = holiday_factor(dates, "retail"), holiday_factor(dates, "coffee")
    black_friday = list(dates.date).index(dt.date(2024, 11, 29))
    assert retail[black_friday] > 1.5 and coffee[black_friday] == 1.0


def _cells():
    # 2 x 2 grid: dimension A (levels 0,1) x dimension B (levels 0,1)
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    return a, b


def test_raking_hits_both_margins():
    a, b = _cells()
    counts = np.array([50, 10, 30, 10])  # panel skews to A=0
    targets = [np.array([0.4, 0.6]), np.array([0.7, 0.3])]
    w = rake(counts, [a, b], targets)
    mass = w * counts
    np.testing.assert_allclose(np.bincount(a, mass) / mass.sum(), targets[0], atol=1e-6)
    np.testing.assert_allclose(np.bincount(b, mass) / mass.sum(), targets[1], atol=1e-6)


def test_raking_survives_an_empty_category():
    a, b = _cells()
    counts = np.array([0, 0, 30, 10])  # nobody in A=0 today (e.g. a source outage)
    shares = raked_shares(counts, [a, b], [np.array([0.4, 0.6]), np.array([0.7, 0.3])])
    assert np.isfinite(shares).all() and abs(shares.sum() - 1) < 1e-9
    np.testing.assert_allclose(shares[2:] / shares[2:].sum(), [0.7, 0.3], atol=1e-6)
