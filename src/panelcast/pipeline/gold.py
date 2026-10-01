"""Gold base tables: daily panel membership, ticker-tagged spend, and per-source health metrics.

gold.members_daily   (date, source, age_bucket, income_bucket, region) -> members enrolled
gold.spend_daily     (date, ticker, source, age_bucket, income_bucket, region) -> spend, txns
gold.source_daily    (date, source) -> txns, spend, median ticket, members  [anomaly detection input]
"""

from __future__ import annotations

import logging

from pyspark.sql import functions as F

from panelcast.config import Settings
from panelcast.store import TableStore

log = logging.getLogger(__name__)

CELL = ["age_bucket", "income_bucket", "region"]


def build_gold_base(settings: Settings, store: TableStore) -> None:
    txns = store.read("silver", "transactions")
    members = store.read("silver", "panel_members")
    last_day = txns.agg(F.max("txn_date")).first()[0]

    end = F.least(F.coalesce("leave_date", F.lit(last_day)), F.lit(last_day))
    members_daily = (
        # sequence() counts *down* when start > stop, so members who join after the data ends are dropped first
        members.where(F.col("join_date") <= end)
        .select("source", *CELL, F.explode(F.sequence("join_date", end)).alias("date"))
        .groupBy("date", "source", *CELL)
        .agg(F.count("*").alias("members"))
    )
    store.write(members_daily, "gold", "members_daily")

    dmap = store.read("silver", "descriptor_map").where("decision = 'MATCH'").select("descriptor_norm", "mcc", "brand")
    brands = store.read("ref", "brands").select("brand", "ticker", "valid_from", "valid_to")
    tagged = (
        txns.join(F.broadcast(dmap), ["descriptor_norm", "mcc"])
        .join(F.broadcast(brands), "brand")
        # brand -> ticker ownership is effective-dated (acquisitions count only after the close)
        .where(F.col("txn_date").between(F.col("valid_from"), F.col("valid_to")))
        .join(F.broadcast(members.select("user_id", *CELL)), "user_id")
    )
    spend_daily = tagged.groupBy(F.col("txn_date").alias("date"), "ticker", "source", *CELL).agg(
        F.sum("amount").cast("double").alias("spend"),
        F.count("*").alias("txns"),
    )
    store.write(spend_daily, "gold", "spend_daily")

    source_members = (
        store.read("gold", "members_daily").groupBy("date", "source").agg(F.sum("members").alias("members"))
    )
    source_daily = (
        txns.groupBy(F.col("txn_date").alias("date"), "source")
        .agg(
            F.count("*").alias("txns"),
            F.sum("amount").cast("double").alias("spend"),
            F.percentile_approx(F.when(F.col("amount") > 0, F.col("amount")), 0.5)
            .cast("double")
            .alias("median_ticket"),
        )
        .join(source_members, ["date", "source"], "full")
        .fillna({"txns": 0, "spend": 0.0, "members": 0})
    )
    store.write(source_daily, "gold", "source_daily")
    log.info("gold base tables written (data through %s)", last_day)
