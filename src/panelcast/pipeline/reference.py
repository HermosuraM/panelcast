"""Load reference data into the `ref` layer: real revenue, real pageviews, the universe, census margins."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from panelcast.config import Settings
from panelcast.entity_resolution.normalize import normalize
from panelcast.store import TableStore

log = logging.getLogger(__name__)


def _dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c]).dt.date
    return df


def brand_tables(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    brands = pd.DataFrame(
        [
            {
                "brand": b.brand,
                "ticker": b.ticker,
                "valid_from": b.valid_from,
                "valid_to": b.valid_to,
                "mcc_list": ",".join(str(m) for m in b.mcc),
            }
            for b in settings.brands
        ]
    )
    aliases = pd.DataFrame(
        [
            {"brand": b.brand, "ticker": b.ticker, "alias": a, "alias_norm": normalize(a)}
            for b in settings.brands
            for a in b.aliases
        ]
    ).drop_duplicates(["brand", "alias_norm"])
    return brands, aliases


def load_reference(settings: Settings, store: TableStore) -> None:
    ref = settings["reference"]
    revenue = _dates(pd.read_csv(settings.path(ref["edgar_revenue"])), ["period_start", "period_end", "filed"])
    store.write_pandas(revenue, "ref", "revenue")

    wiki = _dates(pd.read_csv(settings.path(ref["wiki_pageviews"])), ["date"])
    store.write_pandas(wiki, "ref", "wiki_pageviews")

    companies = pd.DataFrame(
        [{"ticker": c.ticker, "cik": c.cik, "name": c.name, "sector": c.sector} for c in settings.companies]
    )
    store.write_pandas(companies, "ref", "companies")
    brands, aliases = brand_tables(settings)
    store.write_pandas(brands, "ref", "brands")
    store.write_pandas(aliases, "ref", "brand_aliases")

    margins_path = Path(store.landing("reference", "population_margins.csv"))
    if margins_path.exists():
        store.write_pandas(pd.read_csv(margins_path), "ref", "population_margins")
    else:
        log.warning("population margins not found at %s (run the simulate stage first)", margins_path)
    log.info(
        "reference: %d revenue quarters, %d pageview rows, %d brands, %d aliases",
        len(revenue),
        len(wiki),
        len(brands),
        len(aliases),
    )
