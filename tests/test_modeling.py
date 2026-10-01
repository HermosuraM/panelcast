import datetime as dt

import numpy as np
import pandas as pd

from panelcast.modeling import nowcast
from panelcast.pipeline.estimates import fiscal_quarters


def _quarters(n_tickers=4, n_q=28, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_tickers):
        x = rng.normal(0.05, 0.06, n_q)
        y = 0.01 + 0.9 * x + rng.normal(0, 0.01, n_q)
        for i in range(n_q):
            end = (pd.Timestamp("2018-03-31") + pd.offsets.QuarterEnd(i)).date()
            rows.append(
                {
                    "ticker": f"T{k}",
                    "fiscal_year": 2018 + i // 4,
                    "fiscal_quarter": i % 4 + 1,
                    "period_start": (pd.Timestamp(end) - pd.Timedelta(days=90)).date(),
                    "period_end": end,
                    "revenue": 1e9 * np.exp(y[i]),
                    "revenue_prior": 1e9,
                    "rev_yoy": y[i],
                    "raked_yoy": x[i],
                    "raw_yoy": x[i] + rng.normal(0, 0.2),
                    "complete": True,
                    "filed": end + dt.timedelta(days=35),
                    "nowcast_date": end + dt.timedelta(days=7),
                    "lead_days": 28,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_is_point_in_time(monkeypatch):
    seen = []
    real = nowcast.fit_predict

    def spy(train, test, tickers, seed):
        seen.append((pd.to_datetime(train["filed"]).max(), pd.to_datetime(test["nowcast_date"]).min()))
        return real(train, test, tickers, seed)

    monkeypatch.setattr(nowcast, "fit_predict", spy)
    preds = nowcast.walk_forward(nowcast.feature_frame(_quarters()), "2020-01-01", min_train=12, seed=1)
    assert seen and all(last_filed < when for last_filed, when in seen)
    assert set(preds["model"]) == set(nowcast.MODELS)


def test_panel_models_beat_naive_when_the_signal_is_informative():
    preds = nowcast.walk_forward(nowcast.feature_frame(_quarters()), "2020-01-01", min_train=12, seed=1)
    m = nowcast.metrics(preds, ("2020-01-01", "2020-12-31")).query("segment == 'all'").set_index("model")
    assert m.loc["ensemble", "mae_pp"] < 0.5 * m.loc["naive", "mae_pp"]
    assert np.isnan(m.loc["naive", "direction_acc"])


def test_fiscal_quarters_sum_exact_periods_and_compute_yoy():
    days = pd.date_range("2022-01-01", "2023-12-31")
    level = np.where(days.year == 2022, 1.0, 1.1)
    daily = pd.DataFrame(
        {"date": days.date, "ticker": "A", "raw": level, "per_member": level, "clean": level, "raked": level}
    )
    revenue = pd.DataFrame(
        {
            "ticker": "A",
            "fiscal_year": [2022, 2022, 2023, 2023],
            "fiscal_quarter": [1, 2, 1, 2],
            "period_start": [dt.date(2022, 1, 1), dt.date(2022, 4, 1), dt.date(2023, 1, 1), dt.date(2023, 4, 1)],
            "period_end": [dt.date(2022, 3, 31), dt.date(2022, 6, 30), dt.date(2023, 3, 31), dt.date(2023, 6, 30)],
            "period_days": [90, 91, 90, 91],
            "revenue": [100.0, 110.0, 108.0, 121.0],
            "filed": [dt.date(2022, 5, 1), dt.date(2022, 8, 1), dt.date(2023, 5, 1), dt.date(2023, 8, 1)],
        }
    )
    q = fiscal_quarters(daily, revenue, delivery_lag=7).set_index(["fiscal_year", "fiscal_quarter"])
    assert q.loc[(2022, 1), "panel_raked"] == 90.0
    np.testing.assert_allclose(q.loc[(2023, 1), "raked_yoy"], np.log(1.1))
    np.testing.assert_allclose(q.loc[(2023, 2), "rev_yoy"], np.log(121 / 110))
    assert q.loc[(2023, 2), "lead_days"] == (dt.date(2023, 8, 1) - dt.date(2023, 7, 7)).days
