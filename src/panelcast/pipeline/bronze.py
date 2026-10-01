"""Bronze: land vendor files as-delivered (all strings), incrementally and exactly once.

Databricks uses Auto Loader (`cloudFiles`); locally the same reader options run on Spark's file stream
source. Either way a checkpoint records which files were ingested, so re-runs only pick up new
deliveries and a re-delivered file lands again (and is de-duplicated in silver).
"""

from __future__ import annotations

import logging

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from panelcast.config import Settings
from panelcast.store import TableStore

log = logging.getLogger(__name__)

TXN_COLUMNS = ["txn_id", "user_id", "txn_date", "post_date", "amount", "currency", "merchant_descriptor", "mcc"]
TXN_SCHEMA = StructType([StructField(c, StringType()) for c in TXN_COLUMNS])


def transactions_stream(store: TableStore, max_files: int = 250):
    spark = store.spark
    if store.databricks:
        reader = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("cloudFiles.maxFilesPerTrigger", max_files)
            .option("cloudFiles.schemaLocation", store.checkpoint("bronze_transactions_schema"))
        )
    else:
        reader = spark.readStream.format("csv").option("maxFilesPerTrigger", max_files)
    raw = reader.option("header", True).schema(TXN_SCHEMA).load(store.landing("transactions"))
    return raw.select(
        *TXN_COLUMNS,
        F.regexp_extract(F.col("_metadata.file_path"), r"source=([^/]+)", 1).alias("source"),
        F.regexp_extract(F.col("_metadata.file_path"), r"delivery=([^/]+)", 1).alias("delivery"),
        F.col("_metadata.file_path").alias("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
    )


def ingest(settings: Settings, store: TableStore) -> None:
    query = (
        transactions_stream(store)
        .writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", store.checkpoint("bronze_transactions"))
        .trigger(availableNow=True)
    )
    if store.databricks:
        handle = query.toTable(store.ref("bronze", "transactions"))
    else:
        handle = query.start(store.location("bronze", "transactions"))
    handle.awaitTermination()
    progress = handle.recentProgress
    rows = sum(p.get("numInputRows", 0) for p in progress) if progress else 0
    log.info("bronze.transactions: +%s rows in %d micro-batches", f"{rows:,}", len(progress or []))

    members = (
        store.spark.read.option("header", True)
        .csv(store.landing("panel"))
        .select("*", F.col("_metadata.file_path").alias("_source_file"), F.current_timestamp().alias("_ingested_at"))
    )
    store.write(members, "bronze", "panel_members")
