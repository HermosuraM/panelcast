import numpy as np
import pandas as pd

from panelcast.anomaly.detectors import days_to_events, source_flags, tagging_breaks

RNG = np.random.default_rng(7)


def _source_series(n=200):
    dates = pd.date_range("2023-01-01", periods=n).date
    return pd.DataFrame(
        {
            "date": dates,
            "source": "SRC_X",
            "members": 1000,
            "txns": RNG.poisson(500, n).astype(float),
            "lr_rate": RNG.normal(0, 0.02, n),
            "lr_ticket": RNG.normal(0, 0.02, n),
        }
    )


def test_outage_and_unit_error_are_flagged_and_noise_is_not():
    g = _source_series()
    g.loc[120:121, "txns"] = 0  # outage
    g.loc[150:154, "lr_ticket"] += np.log(100)  # amounts delivered in cents
    flags = source_flags(g, window=28, z_threshold=5, unit_band=(3.9, 5.3))
    by_type = flags.groupby("type")["date"].apply(list)
    assert len(by_type["outage"]) == 2
    assert len(by_type["unit_error"]) == 5
    assert set(flags["type"]) <= {"outage", "unit_error"}
    assert (flags.loc[flags["type"] == "unit_error", "factor"] == 0.01).all()


def _ticker_series(n=300, break_at=None, expected=200.0):
    dates = pd.date_range("2022-01-01", periods=n).date
    lam = np.full(n, expected)
    if break_at is not None:
        lam[break_at:] *= 0.05  # descriptors stop resolving
    return pd.DataFrame(
        {
            "date": dates,
            "source": "SRC_X",
            "ticker": "TGT",
            "txns7": RNG.poisson(lam).astype(float),
            "members7": 7000.0,
            "c7": expected / 7000.0,
        }
    )


def test_tagging_break_detected_near_true_start():
    ev = tagging_breaks(_ticker_series(break_at=200), baseline_days=56, z_threshold=4, min_drop=0.35)
    assert len(ev) == 1
    start = pd.Timestamp(ev.iloc[0]["start"])
    assert abs((start - pd.Timestamp("2022-01-01") - pd.Timedelta(days=200)).days) <= 7
    assert ev.iloc[0]["drop"] > 0.8


def test_no_break_in_pure_noise():
    assert tagging_breaks(_ticker_series(), baseline_days=56, z_threshold=4, min_drop=0.35).empty


def test_days_collapse_into_events():
    flags = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-03-01"]).date,
            "source": "SRC_A",
            "type": "outage",
            "score": -99.0,
            "action": "exclude",
            "factor": 1.0,
        }
    )
    ev = days_to_events(flags)
    assert list(ev["n_days"]) == [3, 1]
