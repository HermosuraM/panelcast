"""Quarterly revenue from SEC EDGAR XBRL "company facts", aligned to each company's fiscal calendar.

Lessons baked in (each one bites if ignored):
  * Identify periods by their start/end dates, never by the `fy`/`fp` tags - those describe the *filing*
    the fact appeared in, so a prior-year comparative in a 10-Q carries the current year's labels.
  * Keep the *first-filed* value for every period (what the market saw on earnings day) and record
    later restatements separately. Training a nowcast on restated numbers is look-ahead bias.
  * Revenue concepts change over time (ASC 606 moved most filers from SalesRevenueNet to
    RevenueFromContractWithCustomer... around 2018), so pick a primary concept per company and only
    back-fill from other concepts when they agree on the periods both report.
  * Q4 is rarely filed on its own: derive it as annual (10-K) minus Q1..Q3.
  * 52/53-week filers have 12-, 13-, 14-, 16- and 17-week quarters; nothing assumes calendar quarters.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DEFAULT_UA = "PanelCast research-project"
QUARTER_DAYS = (80, 125)
ANNUAL_DAYS = (350, 380)
CONSISTENCY_TOL = 0.005  # fallback concepts must agree within 0.5% on overlapping periods


def user_agent() -> str:
    """SEC asks automated clients to identify themselves ("Name email"); set SEC_USER_AGENT to do so."""
    return os.environ.get("SEC_USER_AGENT", DEFAULT_UA)


def fetch_companyfacts(cik: int, cache_dir: Path, refresh: bool = False) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"CIK{cik:010d}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_bytes())
    resp = requests.get(COMPANYFACTS_URL.format(cik=cik), headers={"User-Agent": user_agent()}, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    time.sleep(0.15)  # stay well under SEC's 10 requests/second fair-access limit
    return resp.json()


def facts_frame(doc: dict, concepts: list[str]) -> pd.DataFrame:
    """Flatten USD duration facts for the given us-gaap concepts."""
    gaap = doc.get("facts", {}).get("us-gaap", {})
    rows = []
    for concept in concepts:
        for fact in gaap.get(concept, {}).get("units", {}).get("USD", []):
            if "start" not in fact:
                continue
            rows.append(
                {
                    "concept": concept,
                    "start": dt.date.fromisoformat(fact["start"]),
                    "end": dt.date.fromisoformat(fact["end"]),
                    "val": float(fact["val"]),
                    "filed": dt.date.fromisoformat(fact["filed"]),
                    "form": fact.get("form", ""),
                    "accn": fact.get("accn", ""),
                    "fy": fact.get("fy"),
                    "fp": fact.get("fp"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["days"] = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days + 1
    df["kind"] = "other"
    df.loc[df["days"].between(*QUARTER_DAYS), "kind"] = "quarter"
    df.loc[df["days"].between(*ANNUAL_DAYS), "kind"] = "annual"
    return df


def first_reported(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (concept, start, end): the earliest-filed value, plus the latest value for restatement checks."""
    df = df.sort_values(["concept", "start", "end", "filed", "accn"])
    grp = df.groupby(["concept", "start", "end"], sort=False)
    first = grp.head(1).set_index(["concept", "start", "end"])
    latest = grp.tail(1).set_index(["concept", "start", "end"])["val"].rename("val_latest")
    out = first.join(latest).reset_index()
    out["restated"] = (out["val_latest"] - out["val"]).abs() > 0.001 * out["val"].abs()
    return out


def _choose_series(per_concept: pd.DataFrame, concepts: list[str], since: dt.date) -> tuple[pd.DataFrame, str]:
    """Primary concept = most periods since `since`; others back-fill only where they agree on overlaps."""
    recent = per_concept[per_concept["end"] >= since]
    coverage = recent.groupby("concept")["end"].nunique()
    ranked = sorted(concepts, key=lambda c: (-int(coverage.get(c, 0)), concepts.index(c)))
    primary = ranked[0]
    chosen = per_concept[per_concept["concept"] == primary].copy()
    for other in ranked[1:]:
        cand = per_concept[per_concept["concept"] == other]
        if cand.empty:
            continue
        overlap = chosen.merge(cand, on=["start", "end"], suffixes=("", "_o"))
        if len(overlap) and ((overlap["val"] - overlap["val_o"]).abs() / overlap["val"].abs()).max() > CONSISTENCY_TOL:
            log.info("skip back-fill from %s: disagrees with %s on overlapping periods", other, primary)
            continue
        have = set(zip(chosen["start"], chosen["end"], strict=True))
        extra = cand[[(s, e) not in have for s, e in zip(cand["start"], cand["end"], strict=True)]]
        chosen = pd.concat([chosen, extra], ignore_index=True)
    return chosen, primary


def quarterly_revenue(doc: dict, concepts: list[str], since: dt.date) -> tuple[pd.DataFrame, list[str]]:
    """Fiscal quarters with first-reported revenue. Returns (frame, validation_messages)."""
    facts = facts_frame(doc, concepts)
    issues: list[str] = []
    if facts.empty:
        return pd.DataFrame(), ["no revenue facts"]
    per = first_reported(facts[facts["kind"].isin(["quarter", "annual"])])
    # When revenue first became public, under ANY concept: companies sometimes re-tag history (SalesRevenueNet in
    # the original 10-Q, Revenues in a later 10-K), and the market learned the number at the earlier filing.
    first_public = per.groupby(["start", "end"])["filed"].min().to_dict()
    series, primary = _choose_series(per, concepts, since)
    series = series.assign(
        filed=[
            min(f, first_public[(s, e)])
            for s, e, f in zip(series["start"], series["end"], series["filed"], strict=True)
        ]
    )
    quarters = series[series["kind"] == "quarter"].copy()
    quarters["derived"] = False
    annuals = series[series["kind"] == "annual"].sort_values("end")

    derived_rows = []
    for _, fy in annuals.iterrows():
        inside = quarters[(quarters["start"] >= fy["start"]) & (quarters["end"] <= fy["end"])].sort_values("start")
        if len(inside) == 3 and inside["end"].max() < fy["end"]:
            q4_start = inside["end"].max() + dt.timedelta(days=1)
            derived_rows.append(
                {
                    "concept": fy["concept"],
                    "start": q4_start,
                    "end": fy["end"],
                    "val": fy["val"] - inside["val"].sum(),
                    "val_latest": fy["val_latest"] - inside["val_latest"].sum(),
                    "filed": fy["filed"],
                    "form": fy["form"],
                    "accn": fy["accn"],
                    "restated": bool(fy["restated"] or inside["restated"].any()),
                    "days": (fy["end"] - q4_start).days + 1,
                    "kind": "quarter",
                    "derived": True,
                }
            )
        elif len(inside) == 4 and fy["end"] >= since:
            gap = abs(inside["val"].sum() - fy["val"]) / fy["val"]
            if gap > 0.002:
                issues.append(f"FY ending {fy['end']}: quarters sum differs from annual by {gap:.2%}")
    if derived_rows:
        quarters = pd.concat([quarters, pd.DataFrame(derived_rows)], ignore_index=True)

    quarters = quarters[quarters["end"] >= since].sort_values(["start", "derived"])
    quarters = quarters.drop_duplicates(subset=["start", "end"], keep="first")
    quarters = _drop_overlaps(quarters, issues, quiet_before=since + dt.timedelta(days=366))
    quarters = _label_fiscal(quarters, annuals)

    for prev, cur in zip(quarters.itertuples(), quarters.iloc[1:].itertuples(), strict=False):
        # quirks before the window we actually model (e.g. YUM's 2016 move to calendar quarters) are expected
        if cur.start < since + dt.timedelta(days=366):
            continue
        if cur.start != prev.end + dt.timedelta(days=1):
            issues.append(f"gap/overlap between {prev.end} and {cur.start}")
    if (quarters["val"] <= 0).any():
        issues.append("non-positive quarterly revenue")
    quarters["primary_concept"] = primary
    return quarters.reset_index(drop=True), issues


def _drop_overlaps(q: pd.DataFrame, issues: list[str], quiet_before: dt.date) -> pd.DataFrame:
    """When a company changes its quarter definitions, keep a non-overlapping chain (latest-starting wins)."""
    keep = []
    for row in q.sort_values("start").itertuples():
        if keep and row.start <= keep[-1].end:
            if row.start >= quiet_before:
                issues.append(f"overlapping periods {keep[-1].start}..{keep[-1].end} and {row.start}..{row.end}")
            if row.derived and not keep[-1].derived:
                continue
            keep.pop()
        keep.append(row)
    return pd.DataFrame(keep).drop(columns="Index", errors="ignore")


def _label_fiscal(q: pd.DataFrame, annuals: pd.DataFrame) -> pd.DataFrame:
    """fiscal_year = calendar year in which the fiscal year ends; fiscal_quarter = position within it."""
    q = q.copy()
    fy_ends = sorted(annuals["end"].unique())
    labels = []
    for row in q.itertuples():
        fy_end = next((e for e in fy_ends if e >= row.end), None)
        if fy_end is None:  # fiscal year still in progress: project one year past the last FY end
            last = fy_ends[-1] if fy_ends else row.end
            years_ahead = max(1, (row.end - last).days // 365 + 1)
            fy_end = last + dt.timedelta(days=364 * years_ahead)
        # 52/53-week years can end in the first days of January; shift a week so they keep last year's label.
        labels.append((fy_end - dt.timedelta(days=7)).year)
    q["fiscal_year"] = labels
    q = q.sort_values("start")
    q["fiscal_quarter"] = q.groupby("fiscal_year").cumcount() + 1
    # A partially observed first fiscal year would be mislabeled Q1; re-anchor so its LAST quarter is Q4.
    counts = q.groupby("fiscal_year")["fiscal_quarter"].transform("max")
    first_year = q["fiscal_year"].min()
    mask = (q["fiscal_year"] == first_year) & (counts < 4)
    q.loc[mask, "fiscal_quarter"] = q.loc[mask, "fiscal_quarter"] + (4 - counts[mask])
    return q


def build_revenue_table(companies, concepts: list[str], cache_dir: Path, since: dt.date, refresh: bool = False):
    frames, report = [], {}
    for comp in companies:
        doc = fetch_companyfacts(comp.cik, cache_dir, refresh=refresh)
        q, issues = quarterly_revenue(doc, concepts, since)
        report[comp.ticker] = issues
        if q.empty:
            continue
        q.insert(0, "ticker", comp.ticker)
        q.insert(1, "cik", comp.cik)
        frames.append(q)
        log.info(
            "%s: %d quarters (%s .. %s), primary=%s, issues=%d",
            comp.ticker,
            len(q),
            q["start"].min(),
            q["end"].max(),
            q["primary_concept"].iat[0],
            len(issues),
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(
        columns={
            "start": "period_start",
            "end": "period_end",
            "val": "revenue",
            "val_latest": "revenue_latest",
            "days": "period_days",
        }
    )
    cols = [
        "ticker",
        "cik",
        "fiscal_year",
        "fiscal_quarter",
        "period_start",
        "period_end",
        "period_days",
        "revenue",
        "revenue_latest",
        "restated",
        "filed",
        "form",
        "accn",
        "concept",
        "primary_concept",
        "derived",
    ]
    return out[cols].sort_values(["ticker", "period_start"]).reset_index(drop=True), report
