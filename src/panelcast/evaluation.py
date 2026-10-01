"""Score every pipeline stage against the simulator's ground truth (the only module that reads `sim_truth`).

entity resolution   precision / recall / F1, by transactions and by dollars, brand- and ticker-level
data quality        malformed rows caught by quarantine, duplicate deliveries removed
anomaly detection   event-level precision / recall and detection delay
panel estimators    YoY growth error of raw / per-member / clean / raked panel vs. TRUE card-visible spend
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from panelcast.config import Settings
from panelcast.pipeline.estimates import METHODS
from panelcast.store import TableStore

log = logging.getLogger(__name__)

TYPE_MAP = {
    "outage": "outage",
    "silent_departure": "source_dropout",
    "duplicate_delivery": "duplicate_delivery",
    "unit_error": "unit_error",
    "descriptor_change": "tagging_break",
}
MATCH_SLACK_DAYS = 3


def er_confusion(store: TableStore) -> pd.DataFrame:
    """(true_brand, predicted brand) -> transactions and dollars."""
    silver = (
        store.read("silver", "transactions")
        .groupBy("descriptor_raw", "mcc", "descriptor_norm")
        .agg(F.count("*").alias("txns"), F.sum(F.abs("amount")).cast("double").alias("spend"))
    )
    truth = store.read("sim_truth", "descriptor_labels").select(
        F.col("merchant_descriptor").alias("descriptor_raw"), F.col("mcc").cast("int").alias("mcc"), "true_brand"
    )
    pred = (
        store.read("silver", "descriptor_map")
        .where("decision = 'MATCH'")
        .select("descriptor_norm", "mcc", F.col("brand").alias("pred_brand"))
    )
    return (
        silver.join(truth, ["descriptor_raw", "mcc"], "left")
        .join(pred, ["descriptor_norm", "mcc"], "left")
        .fillna({"true_brand": "NONE", "pred_brand": "NONE"})
        .groupBy("true_brand", "pred_brand")
        .agg(F.sum("txns").alias("txns"), F.sum("spend").alias("spend"))
        .toPandas()
    )


def prf(confusion: pd.DataFrame, weight: str, key: dict | None = None) -> dict:
    """Precision/recall/F1 where a prediction counts if it equals the truth (after mapping through `key`)."""
    c = confusion.copy()
    f = (lambda b: key.get(b, "NONE")) if key else (lambda b: b)
    c["t"], c["p"] = c["true_brand"].map(f), c["pred_brand"].map(f)
    tp = c.loc[(c["p"] != "NONE") & (c["p"] == c["t"]), weight].sum()
    pred_pos = c.loc[c["p"] != "NONE", weight].sum()
    true_pos = c.loc[c["t"] != "NONE", weight].sum()
    precision = tp / pred_pos if pred_pos else np.nan
    recall = tp / true_pos if true_pos else np.nan
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)}


def evaluate_er(store: TableStore) -> dict:
    conf = er_confusion(store)
    brands = store.read_pandas("ref", "brands")
    ticker_of = dict(zip(brands["brand"], brands["ticker"], strict=True))
    store.write_pandas(conf, "eval", "er_confusion")
    return {
        "brand_by_txns": prf(conf, "txns"),
        "brand_by_spend": prf(conf, "spend"),
        "ticker_by_txns": prf(conf, "txns", ticker_of),
        "ticker_by_spend": prf(conf, "spend", ticker_of),
    }


def evaluate_anomalies(store: TableStore) -> tuple[dict, pd.DataFrame]:
    truth = store.read_pandas("sim_truth", "anomaly_log")
    detected = store.read_pandas("gold", "anomaly_events")
    brands = store.read_pandas("ref", "brands")
    ticker_of = dict(zip(brands["brand"], brands["ticker"], strict=True))
    truth["det_type"] = truth["type"].map(TYPE_MAP)
    truth["ticker"] = truth["brand"].map(ticker_of)
    slack = pd.Timedelta(days=MATCH_SLACK_DAYS)
    rows, used = [], set()
    for t in truth.itertuples():
        cand = detected[(detected["source"] == t.source) & (detected["type"] == t.det_type)]
        if isinstance(t.ticker, str):
            cand = cand[cand["ticker"] == t.ticker]
        cand = cand[
            (pd.to_datetime(cand["start"]) <= pd.Timestamp(t.end) + slack)
            & (pd.to_datetime(cand["end"]) >= pd.Timestamp(t.start) - slack)
        ]
        hit = cand.index[0] if len(cand) else None
        if hit is not None:
            used.add(hit)
        delay = (pd.Timestamp(detected.at[hit, "start"]) - pd.Timestamp(t.start)).days if hit is not None else None
        rows.append(
            {
                "type": t.type,
                "source": t.source,
                "brand": t.brand,
                "start": t.start,
                "end": t.end,
                "detected": hit is not None,
                "delay_days": delay,
            }
        )
    table = pd.DataFrame(rows)
    flagged = detected[detected["action"] != "flag"]
    false_pos = [i for i in flagged.index if i not in used]
    summary = {
        "injected": len(truth),
        "detected": int(table["detected"].sum()),
        "recall": float(table["detected"].mean()),
        "precision": (len(used) / len(flagged)) if len(flagged) else np.nan,
        "false_positives": len(false_pos),
        "median_abs_delay_days": float(table["delay_days"].abs().median()),
    }
    return summary, table


def evaluate_dq(store: TableStore) -> dict:
    truth = store.read("sim_truth", "transactions")
    malformed = truth.where(F.col("anomaly").startswith("malformed")).select("txn_id").distinct()
    quarantined = store.read("silver", "quarantine").select("txn_id", "dq_reasons")
    caught = malformed.join(quarantined.select("txn_id").distinct(), "txn_id", "left_semi").count()
    n_malformed = malformed.count()
    injected_dupes = truth.where("anomaly = 'duplicate'").count()
    removed = store.read("silver", "duplicate_log").count()
    reasons = (
        quarantined.select(F.explode(F.split("dq_reasons", ",")).alias("reason"))
        .groupBy("reason")
        .count()
        .toPandas()
        .set_index("reason")["count"]
        .to_dict()
    )
    return {
        "malformed_injected": n_malformed,
        "malformed_quarantined": caught,
        "quarantine_recall": caught / n_malformed if n_malformed else np.nan,
        "duplicates_injected": injected_dupes,
        "duplicates_removed": removed,
        "quarantine_reasons": reasons,
    }


def evaluate_panel(store: TableStore) -> tuple[dict, pd.DataFrame]:
    """Compare each estimator's fiscal-quarter YoY growth with the TRUE card-visible spend growth."""
    q = store.read_pandas("gold", "ticker_quarterly")
    truth = store.read_pandas("sim_truth", "daily_truth")
    truth["date"] = pd.to_datetime(truth["date"])
    true_sum = []
    for r in q.itertuples():
        mask = (truth["ticker"] == r.ticker) & truth["date"].between(
            pd.Timestamp(r.period_start), pd.Timestamp(r.period_end)
        )
        true_sum.append(truth.loc[mask, "per_capita"].sum() if mask.sum() >= r.period_days else np.nan)
    q["true_visible"] = true_sum
    prior = q[["ticker", "fiscal_year", "fiscal_quarter", "true_visible"]].assign(
        fiscal_year=lambda d: d.fiscal_year + 1
    )
    q = q.merge(prior, on=["ticker", "fiscal_year", "fiscal_quarter"], how="left", suffixes=("", "_prior"))
    q["true_yoy"] = np.log(q["true_visible"] / q["true_visible_prior"])
    ok = q["true_yoy"].notna() & np.isfinite(q["raked_yoy"])
    errs = {m: 100 * (q.loc[ok, f"{m}_yoy"] - q.loc[ok, "true_yoy"]) for m in METHODS}
    summary = {
        m: {
            "mae_pp": float(e.abs().mean()),
            "median_ae_pp": float(e.abs().median()),
            "p90_ae_pp": float(e.abs().quantile(0.9)),
        }
        for m, e in errs.items()
    }
    detail = q.loc[
        ok,
        [
            "ticker",
            "fiscal_year",
            "fiscal_quarter",
            "period_end",
            "true_yoy",
            "rev_yoy",
            *[f"{m}_yoy" for m in METHODS],
        ],
    ]
    return summary, detail


def evaluate_against_truth(settings: Settings, store: TableStore) -> dict:
    er = evaluate_er(store)
    anomalies, anomaly_table = evaluate_anomalies(store)
    dq = evaluate_dq(store)
    panel, panel_detail = evaluate_panel(store)
    store.write_pandas(anomaly_table, "eval", "anomaly_detection")
    store.write_pandas(panel_detail, "eval", "panel_accuracy")
    result = {"entity_resolution": er, "anomaly_detection": anomalies, "data_quality": dq, "panel_estimators": panel}
    out = settings.reports_dir
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2, default=_jsonable))
    log.info("evaluation vs ground truth:\n%s", json.dumps(result, indent=2, default=_jsonable))
    return result


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)
