"""Read-only, point-in-time research tools over the gold tables.

`ResearchData.as_of(date)` hides everything that was not public on the nowcast date: revenue filed later,
later predictions, later anomaly events. The agent writing a pre-earnings note can therefore never see the
number it is trying to predict.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from panelcast.store import TableStore

SQL_ROW_LIMIT = 200
SQL_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def _r(x, nd: int = 2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _pct(x) -> float | None:
    """log growth -> simple growth in percent (plain Python float: tool outputs must be JSON/msgpack friendly)"""
    return None if x is None or pd.isna(x) else round(float(100 * (np.exp(float(x)) - 1)), 2)


@dataclass
class ResearchData:
    companies: pd.DataFrame
    brands: pd.DataFrame
    quarters: pd.DataFrame
    predictions: pd.DataFrame
    events: pd.DataFrame
    scorecard: pd.DataFrame
    er_coverage: pd.DataFrame
    as_of_date: dt.date | None = None

    @classmethod
    def load(cls, store: TableStore) -> ResearchData:
        dmap = store.read_pandas("silver", "descriptor_map")
        er = dmap[dmap["ticker"].notna()].groupby(["ticker", "decision", "method"], as_index=False)["abs_spend"].sum()
        frames = {
            "companies": store.read_pandas("ref", "companies"),
            "brands": store.read_pandas("ref", "brands"),
            "quarters": store.read_pandas("gold", "ticker_quarterly"),
            "predictions": store.read_pandas("gold", "nowcast_predictions"),
            "events": store.read_pandas("gold", "anomaly_events"),
            "scorecard": store.read_pandas("gold", "asset_scorecard"),
            "er_coverage": er,
        }
        for name in ("quarters", "predictions"):
            for col in ("period_start", "period_end", "filed", "nowcast_date"):
                if col in frames[name]:
                    frames[name][col] = pd.to_datetime(frames[name][col]).dt.date
            # Delta does not preserve row order; every "latest N" below relies on this sort
            frames[name] = frames[name].sort_values(["ticker", "period_end"]).reset_index(drop=True)
        for col in ("start", "end"):
            frames["events"][col] = pd.to_datetime(frames["events"][col]).dt.date
        return cls(**frames)

    def latest_target(self, ticker: str) -> pd.Series:
        p = self.predictions[(self.predictions["ticker"] == ticker) & (self.predictions["model"] == "ensemble")]
        return p.sort_values("period_end").iloc[-1]

    def as_of(self, when: dt.date) -> ResearchData:
        """Everything as it was known on `when`: unfiled revenue (and anything derived from it) is masked."""
        q = self.quarters[self.quarters["period_start"] < when].copy()
        q.loc[q["filed"] >= when, ["revenue", "rev_yoy", "filed", "lead_days"]] = np.nan
        p = self.predictions[self.predictions["nowcast_date"] <= when].copy()
        p.loc[p["filed"] >= when, ["revenue", "rev_yoy", "filed", "lead_days", "error_pp", "ape"]] = np.nan
        return replace(
            self, quarters=q, predictions=p, events=self.events[self.events["start"] < when], as_of_date=when
        )


def make_tools(data: ResearchData) -> list:
    when = data.as_of_date

    def _target(ticker: str) -> pd.Series:
        p = data.predictions[(data.predictions["ticker"] == ticker) & (data.predictions["model"] == "ensemble")]
        if p.empty:
            raise ValueError(f"no nowcast for {ticker} as of {when}")
        return p.sort_values("period_end").iloc[-1]

    @tool
    def get_company_overview(ticker: str) -> dict:
        """Company name, sector, brands tracked in the card panel (with acquisition dates), fiscal calendar."""
        c = data.companies[data.companies["ticker"] == ticker].iloc[0]
        b = data.brands[data.brands["ticker"] == ticker]
        q = data.quarters[data.quarters["ticker"] == ticker].tail(4)
        return {
            "ticker": ticker,
            "name": c["name"],
            "sector": c["sector"],
            "brands": [
                {"brand": r.brand, "counts_from": str(r.valid_from) if str(r.valid_from) > "1900-01-01" else None}
                for r in b.itertuples()
            ],
            "recent_fiscal_quarters": [
                {
                    "label": f"FY{r.fiscal_year} Q{r.fiscal_quarter}",
                    "start": str(r.period_start),
                    "end": str(r.period_end),
                    "days": int(r.period_days),
                }
                for r in q.itertuples()
            ],
        }

    @tool
    def get_nowcast(ticker: str) -> dict:
        """The ensemble nowcast for the most recent fiscal quarter, its 80% band, and its components."""
        t = _target(ticker)
        comp = data.predictions[
            (data.predictions["ticker"] == ticker) & (data.predictions["period_end"] == t["period_end"])
        ]
        rev_lo = t["revenue_prior"] * np.exp(t["pred_lo"]) if pd.notna(t["pred_lo"]) else None
        rev_hi = t["revenue_prior"] * np.exp(t["pred_hi"]) if pd.notna(t["pred_hi"]) else None
        return {
            "ticker": ticker,
            "quarter": f"FY{int(t['fiscal_year'])} Q{int(t['fiscal_quarter'])}",
            "period_start": str(t["period_start"]),
            "period_end": str(t["period_end"]),
            "nowcast_date": str(t["nowcast_date"]),
            "nowcast_yoy_pct": _pct(t["pred_yoy"]),
            "band_coverage_pct": 80.0,
            "band_80_low_pct": _pct(t["pred_lo"]),
            "band_80_high_pct": _pct(t["pred_hi"]),
            "implied_revenue_usd_bn": _r(t["pred_revenue"] / 1e9, 2),
            "implied_revenue_band_usd_bn": [_r(rev_lo / 1e9 if rev_lo else None), _r(rev_hi / 1e9 if rev_hi else None)],
            "prior_year_revenue_usd_bn": _r(t["revenue_prior"] / 1e9, 2),
            "last_reported_yoy_pct": _pct(t["rev_yoy_lag1"]),
            "panel_raked_spend_yoy_pct": _pct(t["raked_yoy"]),
            "model_components_yoy_pct": {m: _pct(v) for m, v in zip(comp["model"], comp["pred_yoy"], strict=True)},
        }

    @tool
    def get_quarter_history(ticker: str, quarters: int = 8) -> list[dict]:
        """Reported revenue growth vs. panel growth and past nowcast errors for recent quarters (point-in-time)."""
        q = data.quarters[(data.quarters["ticker"] == ticker) & data.quarters["rev_yoy"].notna()].tail(quarters)
        p = data.predictions[(data.predictions["ticker"] == ticker) & (data.predictions["model"] == "ensemble")]
        p = p.set_index("period_end")
        rows = []
        for r in q.itertuples():
            pred = p["pred_yoy"].get(r.period_end)
            rows.append(
                {
                    "quarter": f"FY{r.fiscal_year} Q{r.fiscal_quarter}",
                    "period_end": str(r.period_end),
                    "reported_yoy_pct": _pct(r.rev_yoy),
                    "panel_raked_yoy_pct": _pct(r.raked_yoy),
                    "ensemble_nowcast_yoy_pct": _pct(pred),
                    "nowcast_error_pp": _r(_pct(pred) - _pct(r.rev_yoy))
                    if pred is not None and not pd.isna(pred)
                    else None,
                    "revenue_usd_bn": _r(r.revenue / 1e9, 2),
                    "filed": str(r.filed),
                }
            )
        errs = [abs(x["nowcast_error_pp"]) for x in rows if x["nowcast_error_pp"] is not None]
        if rows:
            rows.append(
                {
                    "summary": {
                        "quarters_with_nowcast": len(errs),
                        "mean_abs_error_pp": _r(float(np.mean(errs))) if errs else None,
                    }
                }
            )
        return rows

    @tool
    def get_data_quality(ticker: str) -> dict:
        """Data problems that touched this ticker's latest quarter, and entity-resolution coverage."""
        t = _target(ticker)
        ev = data.events
        overlap = ev[
            (ev["end"] >= t["period_start"])
            & (ev["start"] <= t["period_end"])
            & (ev["ticker"].isna() | (ev["ticker"] == ticker))
        ]
        cov = data.er_coverage[data.er_coverage["ticker"] == ticker]
        matched = cov[cov["decision"] == "MATCH"].groupby("method")["abs_spend"].sum()
        review = cov[cov["decision"] == "REVIEW"]["abs_spend"].sum()
        total = matched.sum()
        return {
            "quarter": f"FY{int(t['fiscal_year'])} Q{int(t['fiscal_quarter'])}",
            "events_in_quarter": [
                {"type": r.type, "source": r.source, "start": str(r.start), "end": str(r.end), "handling": r.action}
                for r in overlap.itertuples()
            ],
            "entity_resolution": {
                "matched_spend_by_method_pct": {m: _r(100 * v / total) for m, v in matched.items()} if total else {},
                "pending_review_share_pct": _r(100 * review / (total + review)) if total + review else None,
            },
        }

    @tool
    def get_asset_scorecard() -> list[dict]:
        """How each alternative-data asset scored for revenue nowcasting (coverage, fit, out-of-sample skill)."""
        keep = ["asset", "tickers_covered", "median_corr_ex_covid", "oos_skill_vs_naive", "median_lead_days", "verdict"]
        return json.loads(data.scorecard[keep].round(3).to_json(orient="records"))

    @tool
    def run_sql(query: str) -> list[dict]:
        """Read-only SQL (SQLite dialect) over tables `quarters`, `predictions`, `events` (point-in-time)."""
        if SQL_FORBIDDEN.search(query) or ";" in query.strip().rstrip(";"):
            return [{"error": "only a single read-only SELECT statement is allowed"}]
        con = sqlite3.connect(":memory:")
        try:
            for name, df in (("quarters", data.quarters), ("predictions", data.predictions), ("events", data.events)):
                df.astype({c: str for c in df.columns if df[c].dtype == object}).to_sql(name, con, index=False)
            rows = pd.read_sql_query(f"SELECT * FROM ({query.rstrip(';')}) LIMIT {SQL_ROW_LIMIT}", con)
            return json.loads(rows.round(4).to_json(orient="records"))
        except Exception as exc:  # noqa: BLE001 - report SQL errors back to the model
            return [{"error": str(exc)}]
        finally:
            con.close()

    return [get_company_overview, get_nowcast, get_quarter_history, get_data_quality, get_asset_scorecard, run_sql]
