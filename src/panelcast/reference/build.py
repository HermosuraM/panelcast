"""Download real reference data (SEC EDGAR revenue, Wikipedia pageviews) into data/reference/.

The outputs are committed to the repo so the pipeline runs offline (and on Databricks Free Edition,
whose serverless compute restricts outbound internet).
"""

from __future__ import annotations

import datetime as dt
import json
import logging

from panelcast.config import Settings
from panelcast.reference.edgar import build_revenue_table
from panelcast.reference.wikipedia import build_pageviews

log = logging.getLogger(__name__)


def fetch_reference(
    settings: Settings, refresh: bool = False, skip_edgar: bool = False, skip_wiki: bool = False
) -> dict:
    ref = settings["reference"]
    since = dt.date.fromisoformat(str(ref["history_start"]))
    report: dict = {}
    if not skip_edgar:
        revenue, report = build_revenue_table(
            settings.companies, settings.revenue_concepts, settings.path(ref["edgar_facts_cache"]), since, refresh
        )
        out = settings.path(ref["edgar_revenue"])
        out.parent.mkdir(parents=True, exist_ok=True)
        revenue.to_csv(out, index=False)
        out.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
        log.info("wrote %s (%d fiscal quarters)", out, len(revenue))

    if not skip_wiki:
        cache = settings.path(ref["edgar_facts_cache"]).parent / "wiki_cache"
        views = build_pageviews(settings.companies, since, settings.sim_date("end_date"), cache_dir=cache)
        views.to_csv(settings.path(ref["wiki_pageviews"]), index=False, compression="gzip")
        log.info("wrote %s (%d rows)", ref["wiki_pageviews"], len(views))
    return report
