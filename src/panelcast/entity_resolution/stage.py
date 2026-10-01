"""Entity-resolution stage: silver descriptors -> `silver.descriptor_map` (descriptor_norm, mcc -> brand)."""

from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import functions as F

from panelcast.config import Settings
from panelcast.entity_resolution.llm_adjudicator import adjudicate, make_decider
from panelcast.entity_resolution.matcher import DescriptorMatcher
from panelcast.llm import llm_enabled
from panelcast.store import TableStore

log = logging.getLogger(__name__)

SCORE_SCHEMA = (
    "descriptor_norm string, mcc int, brand string, score double, method string, decision string, "
    "mcc_ok boolean, runner_up string, runner_up_score double, ticker string"
)


def descriptor_catalog(store: TableStore):
    return (
        store.read("silver", "transactions")
        .groupBy("descriptor_norm", "mcc")
        .agg(
            F.count("*").alias("n_txns"),
            F.sum(F.abs("amount")).cast("double").alias("abs_spend"),
            F.countDistinct("source").alias("n_sources"),
            F.min("txn_date").alias("first_seen"),
            F.max("txn_date").alias("last_seen"),
            F.min("descriptor_raw").alias("example_raw"),
        )
    )


def score_descriptors(store: TableStore, cfg: dict):
    aliases = store.read_pandas("ref", "brand_aliases")
    brands = store.read_pandas("ref", "brands")
    params = {
        "auto_accept": float(cfg["auto_accept"]),
        "review_floor": float(cfg["review_floor"]),
        "top_k": int(cfg["top_k_candidates"]),
    }

    def score(batches):
        matcher = DescriptorMatcher(aliases, brands, **params)  # built once per task, from pickled tables
        for pdf in batches:
            yield matcher.match_frame(pdf)

    catalog = store.read("silver", "descriptor_catalog").select("descriptor_norm", "mcc")
    return catalog.repartition(8).mapInPandas(score, SCORE_SCHEMA)


def review_queue(result: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gray = result["decision"] == "REVIEW"
    triage = (
        (result["decision"] == "NONE")
        & (result["method"] == "fuzzy")
        & (result["score"] >= float(cfg["triage_min_score"]))
    )
    triaged = result[triage].nlargest(int(cfg["triage_top_n"]), "abs_spend")
    queue = pd.concat([result[gray], triaged]).drop_duplicates(["descriptor_norm", "mcc"])
    return queue.sort_values("abs_spend", ascending=False)


def resolve(settings: Settings, store: TableStore) -> pd.DataFrame:
    cfg = settings["entity_resolution"]
    store.write(descriptor_catalog(store), "silver", "descriptor_catalog")
    store.write(score_descriptors(store, cfg), "silver", "descriptor_scores")
    catalog = store.read("silver", "descriptor_catalog").select("descriptor_norm", "mcc", "n_txns", "abs_spend")
    result = store.read("silver", "descriptor_scores").join(catalog, ["descriptor_norm", "mcc"]).toPandas()
    result["reviewed_by"] = None

    queue = review_queue(result, cfg)
    llm_cfg = cfg["llm"]
    if len(queue) and llm_enabled(llm_cfg["enabled"]):
        result = _apply_llm(settings, store, result, queue, llm_cfg)
    elif len(queue):
        log.info("%d descriptors in the review queue; LLM adjudication disabled (set ANTHROPIC_API_KEY)", len(queue))

    brands = store.read_pandas("ref", "brands")
    result["ticker"] = result["brand"].map(dict(zip(brands["brand"], brands["ticker"], strict=True)))
    store.write_pandas(result, "silver", "descriptor_map")
    _log_summary(result)
    return result


def _apply_llm(settings, store, result, queue, llm_cfg) -> pd.DataFrame:
    brands = store.read_pandas("ref", "brands")
    cached = (
        store.read_pandas("silver", "er_llm_decisions")
        if store.exists("silver", "er_llm_decisions")
        else pd.DataFrame(columns=["descriptor_norm", "mcc", "llm_brand", "llm_confidence", "llm_reason", "model"])
    )
    cached = cached[cached["model"] == llm_cfg["model"]]
    todo = queue.merge(cached[["descriptor_norm", "mcc"]], on=["descriptor_norm", "mcc"], how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge").head(int(llm_cfg["max_descriptors"]))
    if len(todo):
        log.info("asking %s to adjudicate %d descriptors", llm_cfg["model"], len(todo))
        decide = make_decider(llm_cfg["model"], llm_cfg["effort"], brands)
        fresh = adjudicate(todo, brands, decide, int(llm_cfg["batch_size"]), float(llm_cfg["min_confidence"]))
        fresh["model"] = llm_cfg["model"]
        cached = pd.concat([cached, fresh], ignore_index=True)
        store.write_pandas(cached.astype({"mcc": "int32"}), "silver", "er_llm_decisions")
    decided = queue[["descriptor_norm", "mcc"]].merge(cached, on=["descriptor_norm", "mcc"])
    merged = result.merge(
        decided[["descriptor_norm", "mcc", "llm_brand"]], on=["descriptor_norm", "mcc"], how="left", indicator=True
    )
    reviewed = merged["_merge"] == "both"
    merged.loc[reviewed, "brand"] = merged.loc[reviewed, "llm_brand"]
    merged.loc[reviewed, "decision"] = merged.loc[reviewed, "llm_brand"].notna().map({True: "MATCH", False: "NONE"})
    merged.loc[reviewed, "method"] = "llm"
    merged.loc[reviewed, "reviewed_by"] = llm_cfg["model"]
    return merged.drop(columns=["llm_brand", "_merge"])


def _log_summary(result: pd.DataFrame) -> None:
    total = result["abs_spend"].sum()
    summary = result.groupby(["decision", "method"]).agg(
        descriptors=("descriptor_norm", "size"), spend=("abs_spend", "sum")
    )
    summary["spend_share"] = (summary["spend"] / total).round(4)
    log.info("entity resolution summary:\n%s", summary.drop(columns="spend").to_string())
