"""Revenue nowcasting with a point-in-time walk-forward backtest.

Target: year-over-year log growth of reported revenue for a fiscal quarter, predicted `delivery_lag` days
after the quarter ends - typically weeks before the 10-Q/10-K is filed. A model may only train on quarters
whose revenue had been *filed* before the nowcast date, which removes look-ahead bias by construction.

Models (each adds one idea):
    naive         last quarter's reported YoY growth (no alt data)
    panel_raw     OLS on raw panel spend growth (the naive way to use alt data)
    panel_raked   ticker intercepts + pooled slope on raked, anomaly-cleaned panel growth
    ridge         multivariate: panel growth, last quarter's growth, last quarter's panel-vs-reported gap
    gbm           gradient-boosted trees on the same features (non-linear, pooled across tickers)
    ensemble      average of ridge and gbm
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from panelcast.config import Settings
from panelcast.store import TableStore

log = logging.getLogger(__name__)

MODELS = ["naive", "panel_raw", "panel_raked", "ridge", "gbm", "ensemble"]
INTERVAL_COVERAGE = 0.8


def feature_frame(q: pd.DataFrame) -> pd.DataFrame:
    """Quarter-level modeling frame with lagged features computed per ticker in fiscal order."""
    q = q.sort_values(["ticker", "period_end"]).copy()
    g = q.groupby("ticker", group_keys=False)
    q["rev_yoy_lag1"] = g["rev_yoy"].shift(1)
    q["raked_yoy_lag1"] = g["raked_yoy"].shift(1)
    q["gap_lag1"] = q["rev_yoy_lag1"] - q["raked_yoy_lag1"]
    q["raked_accel"] = q["raked_yoy"] - q["raked_yoy_lag1"]
    q["filed_lag1"] = g["filed"].shift(1)
    ok = (
        q["complete"]
        & q["rev_yoy"].notna()
        & np.isfinite(q["raked_yoy"])
        & np.isfinite(q["raw_yoy"])
        & q["rev_yoy_lag1"].notna()
        & q["gap_lag1"].notna()
    )
    return q[ok].reset_index(drop=True)


def _design(df: pd.DataFrame, tickers: list[str], cols: list[str], with_tickers: bool = True) -> np.ndarray:
    x = df[cols].to_numpy(float)
    if not with_tickers:
        return x
    onehot = (df["ticker"].to_numpy()[:, None] == np.asarray(tickers)[None, :]).astype(float)
    return np.hstack([x, onehot])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, tickers: list[str], seed: int) -> dict[str, np.ndarray]:
    preds = {"naive": test["rev_yoy_lag1"].to_numpy()}
    y = train["rev_yoy"].to_numpy()

    ols = LinearRegression().fit(train[["raw_yoy"]], y)
    preds["panel_raw"] = ols.predict(test[["raw_yoy"]])

    pr = Ridge(alpha=1.0).fit(_design(train, tickers, ["raked_yoy"]), y)
    preds["panel_raked"] = pr.predict(_design(test, tickers, ["raked_yoy"]))

    cols = ["raked_yoy", "rev_yoy_lag1", "gap_lag1"]
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 12)).fit(_design(train, tickers, cols), y)
    preds["ridge"] = ridge.predict(_design(test, tickers, cols))

    # Trees cannot extrapolate past their training range (the first COVID quarters were far outside it), so
    # anchor on the panel signal and let the boosted trees model only the residual.
    gcols = ["raked_yoy", "rev_yoy_lag1", "gap_lag1", "raked_accel"]
    gbm = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=10, l2_regularization=1.0, random_state=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gbm.fit(train[gcols], y - train["raked_yoy"].to_numpy())
    preds["gbm"] = test["raked_yoy"].to_numpy() + gbm.predict(test[gcols])
    preds["ensemble"] = (preds["ridge"] + preds["gbm"]) / 2
    return preds


def walk_forward(frame: pd.DataFrame, start: str, min_train: int, seed: int) -> pd.DataFrame:
    tickers = sorted(frame["ticker"].unique())
    frame = frame.copy()
    frame["nowcast_date"] = pd.to_datetime(frame["nowcast_date"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["filed_lag1"] = pd.to_datetime(frame["filed_lag1"])
    out = []
    targets = frame[frame["period_end"] >= pd.Timestamp(start).date()]
    for when, test in targets.groupby("nowcast_date"):
        train = frame[frame["filed"] < when]  # only revenue the market had already seen
        test = test[test["filed_lag1"] < when]  # the lag feature must be public too
        if len(train) < min_train or test.empty:
            continue
        preds = fit_predict(train, test, tickers, seed)
        for model, p in preds.items():
            out.append(
                test[
                    [
                        "ticker",
                        "fiscal_year",
                        "fiscal_quarter",
                        "period_start",
                        "period_end",
                        "nowcast_date",
                        "filed",
                        "lead_days",
                        "revenue",
                        "revenue_prior",
                        "rev_yoy",
                        "rev_yoy_lag1",
                        "raked_yoy",
                    ]
                ].assign(model=model, pred_yoy=p)
            )
    preds = pd.concat(out, ignore_index=True)
    preds["error_pp"] = 100 * (preds["pred_yoy"] - preds["rev_yoy"])
    preds["pred_revenue"] = preds["revenue_prior"] * np.exp(preds["pred_yoy"])
    preds["ape"] = (preds["pred_revenue"] / preds["revenue"] - 1).abs()
    return add_intervals(preds)


def add_intervals(preds: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Empirical 80% interval from each model's own *earlier* out-of-sample errors (no peeking)."""
    preds = preds.sort_values("nowcast_date").copy()
    lo, hi = np.full(len(preds), np.nan), np.full(len(preds), np.nan)
    q = (1 - INTERVAL_COVERAGE) / 2
    for _model, g in preds.groupby("model"):
        for i, row in zip(g.index, g.itertuples(), strict=True):
            past = g[(g["filed"] < row.nowcast_date)]["error_pp"].tail(lookback) / 100
            if len(past) >= 20:
                lo[preds.index.get_loc(i)] = row.pred_yoy - np.quantile(past, 1 - q)
                hi[preds.index.get_loc(i)] = row.pred_yoy - np.quantile(past, q)
    preds["pred_lo"], preds["pred_hi"] = lo, hi
    return preds


def metrics(preds: pd.DataFrame, covid: tuple[str, str]) -> pd.DataFrame:
    p = preds.copy()
    p["period_end"] = pd.to_datetime(p["period_end"])
    in_covid = p["period_end"].between(pd.Timestamp(covid[0]), pd.Timestamp(covid[1]))
    naive = p[p["model"] == "naive"].set_index(["ticker", "period_end"])["error_pp"].abs()
    rows = []
    for segment, mask in (("all", np.ones(len(p), bool)), ("covid_2020_21", in_covid), ("post_2021", ~in_covid)):
        for model, g in p[mask].groupby("model"):
            accel_true = np.sign(g["rev_yoy"] - g["rev_yoy_lag1"])
            accel_pred = np.sign(g["pred_yoy"] - g["rev_yoy_lag1"])
            err = g["error_pp"].abs()
            vs_naive = err.to_numpy() < naive.reindex(g.set_index(["ticker", "period_end"]).index).to_numpy()
            covered = g["pred_lo"].notna()
            inside = ((g["rev_yoy"] >= g["pred_lo"]) & (g["rev_yoy"] <= g["pred_hi"]))[covered]
            direction = np.nan if model == "naive" else float((accel_true == accel_pred).mean())  # naive = no change
            rows.append(
                {
                    "segment": segment,
                    "model": model,
                    "n": len(g),
                    "mae_pp": err.mean(),
                    "median_ae_pp": err.median(),
                    "rmse_pp": float(np.sqrt((g["error_pp"] ** 2).mean())),
                    "mape_revenue": g["ape"].mean(),
                    "direction_acc": direction,
                    "beats_naive": float(vs_naive.mean()),
                    "interval_coverage": float(inside.mean()) if len(inside) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_backtest(settings: Settings, store: TableStore) -> pd.DataFrame:
    cfg = settings["nowcast"]
    q = store.read_pandas("gold", "ticker_quarterly")
    frame = feature_frame(q)
    preds = walk_forward(frame, str(cfg["backtest_start"]), int(cfg["min_train_quarters"]), settings.seed)
    store.write_pandas(preds, "gold", "nowcast_predictions")
    m = metrics(preds, tuple(str(x) for x in cfg["covid_window"]))
    store.write_pandas(m, "gold", "nowcast_metrics")
    log.info(
        "nowcast backtest (%d quarter-tickers):\n%s",
        preds[preds.model == "naive"].shape[0],
        m[m["segment"] == "all"].round(3).to_string(index=False),
    )
    return m
