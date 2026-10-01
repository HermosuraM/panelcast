"""Run context shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from pyspark.sql import SparkSession

from panelcast.config import Settings
from panelcast.store import TableStore


@dataclass
class Context:
    settings: Settings
    _spark: SparkSession | None = field(default=None, repr=False)

    @cached_property
    def spark(self) -> SparkSession:
        if self._spark is not None:
            return self._spark
        from panelcast.spark import get_spark

        return get_spark()

    @cached_property
    def store(self) -> TableStore:
        store = TableStore(self.spark, self.settings)
        store.setup()
        return store
