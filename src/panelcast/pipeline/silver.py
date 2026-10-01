"""Silver: typed, validated, de-duplicated, normalized transactions.

Bronze is read as a stream; each micro-batch is
  1. parsed with ANSI-safe casts (try_cast / try_to_timestamp) so bad rows become nulls, not job failures,
  2. split: rows failing any expectation go to `silver.quarantine` with their reasons,
  3. de-duplicated on the vendor's txn_id (within the batch and against rows already loaded),
  4. MERGEd (insert-only) into `silver.transactions`, so re-runs are idempotent.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from panelcast.config import Settings
from panelcast.entity_resolution.normalize import normalize_col
from panelcast.store import TableStore

log = logging.getLogger(__name__)

MIN_DATE = "2010-01-01"


def parse_and_validate(df: DataFrame, max_abs_amount: float, currencies: list[str]) -> DataFrame:
    txn_date = F.to_date(F.try_to_timestamp(F.col("txn_date"), F.lit("yyyy-MM-dd")))
    post_date = F.to_date(F.try_to_timestamp(F.col("post_date"), F.lit("yyyy-MM-dd")))
    amount = F.expr("try_cast(amount AS DECIMAL(14,2))")
    mcc = F.expr("try_cast(mcc AS INT)")
    desc = F.trim(F.col("merchant_descriptor"))
    reasons = F.array_compact(
        F.array(
            F.when(F.col("txn_id").isNull(), F.lit("missing_txn_id")),
            F.when(F.col("user_id").isNull(), F.lit("missing_user_id")),
            F.when(txn_date.isNull(), F.lit("bad_txn_date")),
            F.when(
                txn_date.isNotNull() & ((txn_date < F.lit(MIN_DATE)) | (txn_date > F.current_date())),
                F.lit("txn_date_out_of_range"),
            ),
            F.when(amount.isNull(), F.lit("bad_amount")),
            F.when(F.abs(amount) > max_abs_amount, F.lit("amount_out_of_range")),
            F.when(~F.coalesce(F.col("currency"), F.lit("")).isin(currencies), F.lit("bad_currency")),
            F.when(desc.isNull() | (desc == ""), F.lit("missing_descriptor")),
            F.when(mcc.isNull(), F.lit("bad_mcc")),
        )
    )
    return df.select(
        "*",
        txn_date.alias("txn_date_parsed"),
        post_date.alias("post_date_parsed"),
        amount.alias("amount_parsed"),
        mcc.alias("mcc_parsed"),
        reasons.alias("dq_reasons"),
    )


def to_silver_columns(parsed: DataFrame) -> DataFrame:
    return parsed.select(
        "txn_id",
        "user_id",
        "source",
        "delivery",
        F.col("txn_date_parsed").alias("txn_date"),
        F.col("post_date_parsed").alias("post_date"),
        F.col("amount_parsed").alias("amount"),
        "currency",
        F.col("mcc_parsed").alias("mcc"),
        F.col("merchant_descriptor").alias("descriptor_raw"),
        normalize_col(F.col("merchant_descriptor")).alias("descriptor_norm"),
        "_source_file",
        "_ingested_at",
    )


def dedupe(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Keep one row per txn_id, preferring original deliveries over re-deliveries."""
    w = Window.partitionBy("txn_id").orderBy(
        F.col("delivery").startswith("redelivery").asc(), F.col("_ingested_at").asc(), F.col("_source_file").asc()
    )
    ranked = df.withColumn("_rn", F.row_number().over(w))
    return ranked.where("_rn = 1").drop("_rn"), ranked.where("_rn > 1").drop("_rn")


def _process_batch(batch: DataFrame, batch_id: int, settings: Settings, store: TableStore) -> None:
    spark = batch.sparkSession
    cfg = settings["silver"]
    parsed = parse_and_validate(batch, float(cfg["max_abs_amount"]), list(cfg["valid_currencies"]))

    bad = parsed.where(F.size("dq_reasons") > 0).select(
        *[c for c in batch.columns],
        F.concat_ws(",", "dq_reasons").alias("dq_reasons"),
        F.current_timestamp().alias("_quarantined_at"),
    )
    store.write(bad, "silver", "quarantine", mode="append")

    good = to_silver_columns(parsed.where(F.size("dq_reasons") == 0))
    first, dup_in_batch = dedupe(good)
    dup_cols = ["txn_id", "source", "delivery", "txn_date", "_source_file"]
    dup_rows = dup_in_batch.select(*dup_cols, F.lit("in_batch").alias("reason"))

    if store.exists("silver", "transactions"):
        loaded = store.read("silver", "transactions").select("txn_id")
        dup_rows = dup_rows.unionByName(
            first.join(loaded, "txn_id", "left_semi").select(*dup_cols, F.lit("already_loaded").alias("reason"))
        )
        first.createOrReplaceTempView("silver_batch")
        spark.sql(
            f"""
            MERGE INTO {store.ref("silver", "transactions")} AS t
            USING silver_batch AS s
            ON t.txn_id = s.txn_id
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    else:
        store.write(first, "silver", "transactions")
    store.write(dup_rows.withColumn("_logged_at", F.current_timestamp()), "silver", "duplicate_log", mode="append")
    log.info("silver batch %d merged", batch_id)


def build_silver(settings: Settings, store: TableStore) -> None:
    spark = store.spark
    reader = spark.readStream.format("delta").option("maxFilesPerTrigger", 1000)
    source = (
        reader.table(store.ref("bronze", "transactions"))
        if store.databricks
        else reader.load(store.location("bronze", "transactions"))
    )
    handle = (
        source.writeStream.foreachBatch(lambda df, bid: _process_batch(df, bid, settings, store))
        .option("checkpointLocation", store.checkpoint("silver_transactions"))
        .trigger(availableNow=True)
        .start()
    )
    handle.awaitTermination()
    build_members(store)
    counts = {
        name: store.read("silver", name).count()
        for name in ("transactions", "quarantine", "duplicate_log")
        if store.exists("silver", name)
    }
    log.info("silver row counts: %s", counts)


def build_members(store: TableStore) -> None:
    """Typed, validated panel membership (latest delivery wins per user)."""
    raw = store.read("bronze", "panel_members")
    w = Window.partitionBy("user_id").orderBy(F.col("_ingested_at").desc())
    members = (
        raw.withColumn("_rn", F.row_number().over(w))
        .where("_rn = 1")
        .select(
            "user_id",
            "source",
            "age_bucket",
            "income_bucket",
            "region",
            F.to_date(F.try_to_timestamp(F.col("join_date"), F.lit("yyyy-MM-dd"))).alias("join_date"),
            F.to_date(F.try_to_timestamp(F.col("leave_date"), F.lit("yyyy-MM-dd"))).alias("leave_date"),
        )
        .where(F.col("join_date").isNotNull() & F.col("user_id").isNotNull())
    )
    store.write(members, "silver", "panel_members")
