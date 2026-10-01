"""Panel estimates: turn tagged spend into population-level signals, then roll up to fiscal quarters.

Four estimators, each adding one correction, so their errors show what every step is worth:
    raw          total panel spend (what you get from a naive SUM)
    per_member   spend per enrolled member (fixes panel growth / churn), no anomaly handling
    clean        per-member after anomaly actions (excluded source-days, rescaled unit errors)
    raked        clean + daily raking to census margins (fixes demographic drift in the panel)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from panelcast.config import Settings
from panelcast.stats.raking import raked_shares
from panelcast.store import TableStore

log = logging.getLogger(__name__)

CELL = ["age_bucket", "income_bucket", "region"]
METHODS = ["raw", "per_member", "clean", "raked"]
SHARE_SCHEMA = "date date, age_bucket string, income_bucket string, region string, share double"


def _margins(store: TableStore) -> dict[str, tuple[list[str], np.ndarray]]:
    m = store.read_pandas("ref", "population_margins")
    out = {}
    for dim, col in (("age", "age_bucket"), ("income", "income_bucket"), ("region", "region")):
        part = m[m["dimension"] == dim]
        out[col] = (part["bucket"].tolist(), part["share"].to_numpy(float))
    return out


def make_raker(margins: dict, max_iter: int, tol: float):
    def rake_day(pdf: pd.DataFrame) -> pd.DataFrame:
        levels = [pdf[col].map({b: i for i, b in enumerate(margins[col][0])}).to_numpy() for col in CELL]
        targets = [margins[col][1] for col in CELL]
        shares = raked_shares(pdf["members"].to_numpy(float), levels, targets, max_iter=max_iter, tol=tol)
        return pdf[["date", *CELL]].assign(share=shares)

    return rake_day


def build_estimates(settings: Settings, store: TableStore) -> None:
    gcfg = settings["gold"]
    members = store.read("gold", "members_daily")
    spend = store.read("gold", "spend_daily")
    actions = store.read("gold", "anomaly_actions")
    src_excl = actions.where("ticker IS NULL AND action = 'exclude'").select("date", "source").distinct()
    tick_excl = actions.where("ticker IS NOT NULL AND action = 'exclude'").select("date", "source", "ticker").distinct()
    rescale = actions.where("action = 'rescale'").groupBy("date", "source").agg(F.first("factor").alias("factor"))

    # Daily raking over the sources that are usable that day.
    usable = members.join(src_excl, ["date", "source"], "left_anti")
    cells = usable.groupBy("date", *CELL).agg(F.sum("members").alias("members"))
    raker = make_raker(_margins(store), int(gcfg["raking_max_iter"]), float(gcfg["raking_tol"]))
    store.write(cells.groupBy("date").applyInPandas(raker, SHARE_SCHEMA), "gold", "cell_shares")
    shares = store.read("gold", "cell_shares")

    tickers = spend.select("ticker").distinct()
    clean = (
        usable.crossJoin(F.broadcast(tickers))
        .join(tick_excl, ["date", "source", "ticker"], "left_anti")
        .join(spend.select("date", "ticker", "source", *CELL, "spend"), ["date", "ticker", "source", *CELL], "left")
        .join(F.broadcast(rescale), ["date", "source"], "left")
        .withColumn("spend", F.coalesce("spend", F.lit(0.0)) * F.coalesce("factor", F.lit(1.0)))
        .groupBy("date", "ticker", *CELL)
        .agg(F.sum("spend").alias("spend"), F.sum("members").alias("members"))
        .where("members > 0")
        .join(shares, ["date", *CELL])
        .groupBy("date", "ticker")
        .agg(
            (F.sum("spend") / F.sum("members")).alias("clean"),
            (F.sum(F.col("share") * F.col("spend") / F.col("members")) / F.sum("share")).alias("raked"),
            F.sum("members").alias("members_used"),
        )
    )
    total_members = members.groupBy("date").agg(F.sum("members").alias("members_all"))
    naive = (
        spend.groupBy("date", "ticker")
        .agg(F.sum("spend").alias("raw"))
        .join(total_members, "date")
        .withColumn("per_member", F.col("raw") / F.col("members_all"))
    )
    daily = naive.join(clean, ["date", "ticker"], "full")
    store.write(daily, "gold", "ticker_daily")

    quarterly = fiscal_quarters(
        store.read_pandas("gold", "ticker_daily"),
        store.read_pandas("ref", "revenue"),
        int(settings["nowcast"]["delivery_lag_days"]),
    )
    store.write_pandas(quarterly, "gold", "ticker_quarterly")
    log.info(
        "ticker_quarterly: %d complete fiscal quarters across %d tickers",
        int(quarterly["complete"].sum()),
        quarterly["ticker"].nunique(),
    )


def fiscal_quarters(daily: pd.DataFrame, revenue: pd.DataFrame, delivery_lag: int) -> pd.DataFrame:
    """Sum daily signals over each company's exact fiscal periods; add YoY log growth for revenue and panel."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    rows = []
    for r in revenue.itertuples():
        d = daily[
            (daily["ticker"] == r.ticker)
            & (daily["date"] >= pd.Timestamp(r.period_start))
            & (daily["date"] <= pd.Timestamp(r.period_end))
        ]
        row = {
            "ticker": r.ticker,
            "fiscal_year": r.fiscal_year,
            "fiscal_quarter": r.fiscal_quarter,
            "period_start": r.period_start,
            "period_end": r.period_end,
            "period_days": r.period_days,
            "revenue": r.revenue,
            "filed": r.filed,
            "days_observed": int(d["raked"].notna().sum()),
        }
        for m in METHODS:
            row[f"panel_{m}"] = float(d[m].sum()) if len(d) else np.nan
        rows.append(row)
    q = pd.DataFrame(rows)
    q["complete"] = q["days_observed"] >= q["period_days"]
    for m in METHODS:
        q.loc[~q["complete"], f"panel_{m}"] = np.nan
    q["nowcast_date"] = pd.to_datetime(q["period_end"]) + pd.Timedelta(days=delivery_lag)
    q["nowcast_date"] = q["nowcast_date"].dt.date
    prior = q[["ticker", "fiscal_year", "fiscal_quarter", "revenue", *[f"panel_{m}" for m in METHODS]]].copy()
    prior["fiscal_year"] += 1
    q = q.merge(prior, on=["ticker", "fiscal_year", "fiscal_quarter"], how="left", suffixes=("", "_prior"))
    q["rev_yoy"] = np.log(q["revenue"] / q["revenue_prior"])
    for m in METHODS:
        q[f"{m}_yoy"] = np.log(q[f"panel_{m}"] / q[f"panel_{m}_prior"])
    q["lead_days"] = (pd.to_datetime(q["filed"]) - pd.to_datetime(q["nowcast_date"])).dt.days
    for col in ("period_start", "period_end", "filed"):
        q[col] = pd.to_datetime(q[col]).dt.date
    return q.sort_values(["ticker", "period_end"]).reset_index(drop=True)
