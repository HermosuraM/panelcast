"""One storage API for local Delta paths and Databricks Unity Catalog tables + volumes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from panelcast.config import Settings

log = logging.getLogger(__name__)

LAYERS = ("ref", "bronze", "silver", "gold", "sim_truth", "eval")


class TableStore:
    """Tables are addressed as (layer, name).

    local       -> Delta table at <root>/lakehouse/<layer>/<name>
    databricks  -> Unity Catalog table <catalog>.<schema>.<layer>_<name>
    Vendor files and streaming checkpoints live under `landing_root` (a UC Volume on Databricks).
    """

    def __init__(self, spark: SparkSession, settings: Settings):
        self.spark = spark
        self.settings = settings
        self.databricks = settings.env == "databricks"
        if self.databricks:
            db = settings["storage"]["databricks"]
            self.catalog, self.schema, self.volume = db["catalog"], db["schema"], db["volume"]
            self.landing_root = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"
            self.local_root = None
        else:
            self.local_root = settings.root / settings["storage"]["local_root"]
            self.landing_root = (self.local_root / "landing").as_posix()

    # ---- setup -------------------------------------------------------------------------------
    def setup(self) -> None:
        if self.databricks:
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}")
            self.spark.sql(f"CREATE VOLUME IF NOT EXISTS {self.catalog}.{self.schema}.{self.volume}")
        else:
            for layer in LAYERS:
                (self.local_root / layer).mkdir(parents=True, exist_ok=True)
            Path(self.landing_root).mkdir(parents=True, exist_ok=True)

    # ---- addressing --------------------------------------------------------------------------
    def _local_path(self, layer: str, name: str) -> Path:
        assert layer in LAYERS, layer
        return self.local_root / layer / name

    def location(self, layer: str, name: str) -> str:
        return self._local_path(layer, name).resolve().as_uri()

    def ref(self, layer: str, name: str) -> str:
        """SQL identifier usable in spark.sql()."""
        if self.databricks:
            return f"{self.catalog}.{self.schema}.{layer}_{name}"
        return f"delta.`{self.location(layer, name)}`"

    def landing(self, *parts: str) -> str:
        return "/".join([self.landing_root, *parts])

    def checkpoint(self, name: str) -> str:
        return self.landing("_checkpoints", name)

    # ---- io ----------------------------------------------------------------------------------
    def exists(self, layer: str, name: str) -> bool:
        if self.databricks:
            return self.spark.catalog.tableExists(self.ref(layer, name))
        return (self._local_path(layer, name) / "_delta_log").is_dir()

    def read(self, layer: str, name: str) -> DataFrame:
        if self.databricks:
            return self.spark.table(self.ref(layer, name))
        return self.spark.read.format("delta").load(self.location(layer, name))

    def write(
        self,
        df: DataFrame,
        layer: str,
        name: str,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
    ) -> None:
        writer = df.write.format("delta").mode(mode)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        if self.databricks:
            writer.saveAsTable(self.ref(layer, name))
        else:
            writer.save(self.location(layer, name))
        log.info("wrote %s.%s (%s)", layer, name, mode)

    def read_pandas(self, layer: str, name: str) -> pd.DataFrame:
        return self.read(layer, name).toPandas()

    def write_pandas(self, pdf: pd.DataFrame, layer: str, name: str, schema: str | None = None) -> None:
        sdf = self.spark.createDataFrame(pdf, schema=schema) if schema else self.spark.createDataFrame(pdf)
        self.write(sdf, layer, name)

    def drop(self, layer: str, name: str) -> None:
        if self.databricks:
            self.spark.sql(f"DROP TABLE IF EXISTS {self.ref(layer, name)}")
        else:
            shutil.rmtree(self._local_path(layer, name), ignore_errors=True)

    def reset_layers(self, layers: list[str]) -> None:
        """Drop every table in the given layers (a fresh simulation invalidates everything downstream)."""
        for layer in layers:
            if self.databricks:
                for row in self.spark.sql(f"SHOW TABLES IN {self.catalog}.{self.schema} LIKE '{layer}_*'").collect():
                    self.spark.sql(f"DROP TABLE IF EXISTS {self.catalog}.{self.schema}.{row['tableName']}")
            else:
                shutil.rmtree(self.local_root / layer, ignore_errors=True)
                (self.local_root / layer).mkdir(parents=True, exist_ok=True)

    def reset_landing(self) -> None:
        """Delete vendor files and streaming checkpoints (used before re-simulating)."""
        if self.databricks:
            self._rm_volume_dir(self.landing("transactions"))
            self._rm_volume_dir(self.landing("panel"))
            self._rm_volume_dir(self.landing("_checkpoints"))
        else:
            shutil.rmtree(self.landing_root, ignore_errors=True)
            Path(self.landing_root).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rm_volume_dir(path: str) -> None:
        # Volumes are mounted as a POSIX filesystem on Databricks compute.
        shutil.rmtree(path, ignore_errors=True)
