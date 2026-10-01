"""Daily English-Wikipedia pageviews: a free, real alternative-data asset to evaluate."""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
    "{article}/daily/{start}/{end}"
)


def user_agent() -> str:
    """Wikimedia asks for '<tool>/<version> (<contact>)'; set WIKIMEDIA_USER_AGENT to include your contact."""
    return os.environ.get("WIKIMEDIA_USER_AGENT", "PanelCast/0.1 research-project")


def _get(url: str, params: dict | None = None, attempts: int = 7) -> requests.Response:
    delay = 5.0
    for attempt in range(attempts):
        resp = requests.get(url, params=params, headers={"User-Agent": user_agent()}, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = float(resp.headers.get("Retry-After", delay))
        log.info("rate limited by Wikimedia (attempt %d); sleeping %.0fs", attempt + 1, wait)
        time.sleep(wait)
        delay = min(delay * 2, 120)
    resp.raise_for_status()
    return resp


def resolve_titles(titles: list[str]) -> dict[str, str]:
    """Map titles to canonical article names (pageviews are counted per exact title, not per redirect)."""
    query = _get(API, {"action": "query", "titles": "|".join(titles), "redirects": 1, "format": "json"}).json()["query"]
    mapping = {t: t for t in titles}
    for step in ("normalized", "redirects"):
        for item in query.get(step, []):
            mapping = {k: (item["to"] if v == item["from"] else v) for k, v in mapping.items()}
    return mapping


def fetch_article(article: str, start: dt.date, end: dt.date, cache_dir: Path | None = None) -> pd.DataFrame:
    cache = cache_dir / f"{quote(article, safe='')}_{start:%Y%m%d}_{end:%Y%m%d}.csv" if cache_dir else None
    if cache and cache.exists():
        return pd.read_csv(cache, parse_dates=["date"]).assign(date=lambda d: d["date"].dt.date)
    url = PAGEVIEWS.format(
        article=quote(article.replace(" ", "_"), safe=""), start=f"{start:%Y%m%d}", end=f"{end:%Y%m%d}"
    )
    items = _get(url).json().get("items", [])
    df = pd.DataFrame(
        {
            "date": [dt.datetime.strptime(i["timestamp"][:8], "%Y%m%d").date() for i in items],
            "views": [int(i["views"]) for i in items],
        }
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    time.sleep(1.5)
    return df


def build_pageviews(companies, start: dt.date, end: dt.date, cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch every listed title AND its current canonical title.

    Articles get renamed ("Amazon.com" -> "Amazon (company)", "The Home Depot" -> "Home Depot"); views are
    counted under whichever title was requested, so summing old and new titles keeps the series continuous.
    """
    titles = sorted({t for c in companies for t in c.wiki})
    canonical = resolve_titles(titles)
    frames = []
    for comp in companies:
        for article in sorted({*comp.wiki, *(canonical[t] for t in comp.wiki)}):
            df = fetch_article(article, start, end, cache_dir)
            df.insert(0, "ticker", comp.ticker)
            df.insert(1, "article", article)
            frames.append(df)
            log.info("%s / %s: %d days", comp.ticker, article, len(df))
    return pd.concat(frames, ignore_index=True)


def rename_suspects(views: pd.DataFrame, jump: float = 10.0, by_ticker: bool = False) -> pd.DataFrame:
    """Series whose quarterly median views change >= `jump`x: usually an article rename or merge.

    Article level finds the renames; ticker level (old + new titles summed) confirms they are patched.
    """
    if by_ticker:
        views = views.groupby(["ticker", "date"], as_index=False)["views"].sum().assign(article="(all titles)")
    rows = []
    for (ticker, article), g in views.groupby(["ticker", "article"]):
        q = g.set_index(pd.to_datetime(g["date"]))["views"].resample("QE").median().clip(lower=1)
        ratio = q / q.shift(1)
        for when, r in ratio[(ratio >= jump) | (ratio <= 1 / jump)].items():
            rows.append({"ticker": ticker, "article": article, "quarter": when.date(), "ratio": float(r)})
    return pd.DataFrame(rows, columns=["ticker", "article", "quarter", "ratio"])
