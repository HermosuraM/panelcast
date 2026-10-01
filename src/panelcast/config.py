"""Configuration loading: runtime settings, the real-world universe, and the simulation spec."""

from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from panelcast.paths import require_root

FAR_FUTURE = dt.date(2099, 12, 31)
FAR_PAST = dt.date(1900, 1, 1)


def _as_date(value: Any, default: dt.date) -> dt.date:
    if value is None:
        return default
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


@dataclass(frozen=True)
class Brand:
    brand: str
    ticker: str
    aliases: tuple[str, ...]
    mcc: tuple[int, ...]
    valid_from: dt.date = FAR_PAST
    valid_to: dt.date = FAR_FUTURE


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str
    sector: str
    wiki: tuple[str, ...]
    brands: tuple[Brand, ...] = field(default_factory=tuple)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Settings:
    """Bundle of the three YAML files plus helpers. `overrides` deep-merge into panelcast.yaml/simulation.yaml."""

    def __init__(
        self,
        root: Path | None = None,
        overrides: dict | None = None,
        sim_overrides: dict | None = None,
        env: str | None = None,
    ):
        self.root = Path(root) if root else require_root()
        conf = self.root / "conf"
        self.raw = _deep_merge(_load(conf / "panelcast.yaml"), overrides or {})
        self.universe_raw = _load(conf / "universe.yaml")
        self.sim = _deep_merge(_load(conf / "simulation.yaml"), sim_overrides or {})
        from panelcast.spark import is_databricks

        self.env = env or ("databricks" if is_databricks() else "local")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    def path(self, relative: str) -> Path:
        return self.root / relative

    @property
    def reports_dir(self) -> Path:
        """Where evaluation JSON, research notes, RESULTS.md and figures go (absolute paths allowed).

        On Databricks the default is the project's Unity Catalog volume, not the deployed bundle folder."""
        configured = self.raw.get("reports", {}).get("dir")
        if configured:
            return self.root / configured
        if self.env == "databricks":
            db = self.raw["storage"]["databricks"]
            return Path(f"/Volumes/{db['catalog']}/{db['schema']}/{db['volume']}/reports")
        return self.root / "reports"

    @cached_property
    def companies(self) -> list[Company]:
        out = []
        for c in self.universe_raw["companies"]:
            brands = tuple(
                Brand(
                    brand=b["brand"],
                    ticker=c["ticker"],
                    aliases=tuple(b["aliases"]),
                    mcc=tuple(int(m) for m in b["mcc"]),
                    valid_from=_as_date(b.get("valid_from"), FAR_PAST),
                    valid_to=_as_date(b.get("valid_to"), FAR_FUTURE),
                )
                for b in c["brands"]
            )
            out.append(
                Company(
                    ticker=c["ticker"],
                    cik=int(c["cik"]),
                    name=c["name"],
                    sector=c["sector"],
                    wiki=tuple(c.get("wiki", [])),
                    brands=brands,
                )
            )
        return out

    @cached_property
    def tickers(self) -> list[str]:
        return [c.ticker for c in self.companies]

    @cached_property
    def brands(self) -> list[Brand]:
        return [b for c in self.companies for b in c.brands]

    @property
    def revenue_concepts(self) -> list[str]:
        return list(self.universe_raw["revenue_concepts"])

    def sim_date(self, key: str) -> dt.date:
        return _as_date(self.sim[key], FAR_PAST)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as fh:  # tolerate a BOM (Windows editors add one)
        return yaml.safe_load(fh) or {}
