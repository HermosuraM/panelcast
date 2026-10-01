"""Anomaly stage: score sources and source x ticker series against the cross-source consensus.

Outputs
    gold.anomaly_source_days   day-level source problems (outage, unit error, volume drop/spike)
    gold.anomaly_breaks        source x ticker tagging breaks (descriptor format changes)
    gold.anomaly_actions       what estimates must do per (date, source[, ticker]): exclude or rescale
    gold.anomaly_events        everything collapsed into events, incl. duplicate deliveries from silver
"""

from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from panelcast.anomaly.detectors import days_to_events, source_flags, tagging_breaks
from panelcast.config import Settings
from panelcast.store import TableStore

log = logging.getLogger(__name__)

FLAG_SCHEMA = "date date, source string, type string, score double, action string, factor double"
BREAK_SCHEMA = "source string, ticker string, start date, end date, drop double"
DUPLICATE_MIN_ROWS = 25


def detect(settings: Settings, store: TableStore) -> None:
    spark = store.spark
    cfg = settings["anomaly"]
    window, z_thr = int(cfg["baseline_window"]), float(cfg["z_threshold"])
    unit_band = tuple(float(x) for x in cfg["unit_error_log_ratio"])

    # --- source level ---------------------------------------------------------------------------
    sd = store.read("gold", "source_daily").where("members > 0")
    live = sd.where("txns > 0").withColumn("rate", F.col("txns") / F.col("members"))
    consensus = live.groupBy("date").agg(F.median("rate").alias("c_rate"), F.median("median_ticket").alias("c_ticket"))
    scored = (
        sd.join(consensus, "date", "left")
        .withColumn("lr_rate", F.when(F.col("txns") > 0, F.log((F.col("txns") / F.col("members")) / F.col("c_rate"))))
        .withColumn("lr_ticket", F.when(F.col("txns") > 0, F.log(F.col("median_ticket") / F.col("c_ticket"))))
        .select("date", "source", "txns", "members", "lr_rate", "lr_ticket")
    )
    flags = scored.groupBy("source").applyInPandas(lambda g: source_flags(g, window, z_thr, unit_band), FLAG_SCHEMA)
    store.write(flags, "gold", "anomaly_source_days")
    flags_pdf = store.read_pandas("gold", "anomaly_source_days")

    # --- source x ticker tagging breaks ----------------------------------------------------------
    excluded = flags_pdf[flags_pdf["action"] == "exclude"][["date", "source"]].assign(_excl=True)
    st = store.read("gold", "spend_daily").groupBy("date", "source", "ticker").agg(F.sum("txns").alias("txns"))
    sm = store.read("gold", "members_daily").groupBy("date", "source").agg(F.sum("members").alias("members"))
    tickers = st.select("ticker").distinct()
    grid = sm.where("members > 0").crossJoin(tickers).join(st, ["date", "source", "ticker"], "left").fillna({"txns": 0})
    if len(excluded):
        grid = grid.join(F.broadcast(spark.createDataFrame(excluded)), ["date", "source"], "left")
        grid = (
            grid.withColumn("txns", F.when(F.col("_excl"), None).otherwise(F.col("txns")))
            .withColumn("members", F.when(F.col("_excl"), None).otherwise(F.col("members")))
            .drop("_excl")
        )
    w7 = Window.partitionBy("source", "ticker").orderBy("date").rowsBetween(-6, 0)
    grid = (
        grid.withColumn("txns7", F.sum("txns").over(w7))
        .withColumn("members7", F.sum("members").over(w7))
        .withColumn("n7", F.count("members").over(w7))
        .withColumn("rate7", F.when(F.col("n7") >= 5, F.col("txns7") / F.col("members7")))
    )
    ticker_consensus = grid.groupBy("date", "ticker").agg(F.median("rate7").alias("c7"))
    series = grid.join(ticker_consensus, ["date", "ticker"]).select(
        "date", "source", "ticker", F.col("txns7").cast("double"), F.col("members7").cast("double"), "c7"
    )
    min_drop, tz = float(cfg["ticker_min_drop"]), float(cfg["ticker_z_threshold"])
    breaks = series.groupBy("source", "ticker").applyInPandas(
        lambda g: tagging_breaks(g, baseline_days=2 * window, z_threshold=tz, min_drop=min_drop), BREAK_SCHEMA
    )
    store.write(breaks, "gold", "anomaly_breaks")
    breaks_pdf = store.read_pandas("gold", "anomaly_breaks")

    # --- actions for the estimates stage ---------------------------------------------------------
    actions = flags_pdf.assign(ticker=None)[["date", "source", "ticker", "type", "action", "factor"]]
    if len(breaks_pdf):
        spans = [
            pd.DataFrame(
                {
                    "date": pd.date_range(b.start, b.end, freq="D").date,
                    "source": b.source,
                    "ticker": b.ticker,
                    "type": "tagging_break",
                    "action": "exclude",
                    "factor": 1.0,
                }
            )
            for b in breaks_pdf.itertuples()
        ]
        actions = pd.concat([actions, *spans], ignore_index=True)
    store.write_pandas(
        actions,
        "gold",
        "anomaly_actions",
        schema="date date, source string, ticker string, type string, action string, factor double",
    )

    # --- events (for reporting / evaluation) -----------------------------------------------------
    events = days_to_events(flags_pdf).assign(ticker=None)
    if len(breaks_pdf):
        events = pd.concat(
            [
                events,
                breaks_pdf.assign(
                    type="tagging_break",
                    action="exclude",
                    n_days=(pd.to_datetime(breaks_pdf["end"]) - pd.to_datetime(breaks_pdf["start"])).dt.days + 1,
                ),
            ],
            ignore_index=True,
        )
    dups = (
        store.read("silver", "duplicate_log")
        .groupBy("source", "delivery")
        .agg(F.count("*").alias("rows"), F.min("txn_date").alias("start"), F.max("txn_date").alias("end"))
        .where(F.col("rows") >= DUPLICATE_MIN_ROWS)
        .toPandas()
    )
    if len(dups):
        events = pd.concat(
            [
                events,
                dups.assign(
                    type="duplicate_delivery",
                    action="deduplicated",
                    ticker=None,
                    n_days=(pd.to_datetime(dups["end"]) - pd.to_datetime(dups["start"])).dt.days + 1,
                ).drop(columns=["rows", "delivery"]),
            ],
            ignore_index=True,
        )
    events = events[["type", "source", "ticker", "start", "end", "n_days", "action"]].sort_values(["start", "source"])
    store.write_pandas(
        events,
        "gold",
        "anomaly_events",
        schema="type string, source string, ticker string, start date, end date, n_days long, action string",
    )
    log.info("anomaly events:\n%s", events.to_string(index=False))
