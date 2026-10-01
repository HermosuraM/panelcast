# Databricks notebook source
# MAGIC %md
# MAGIC # PanelCast walkthrough (Databricks Free Edition)
# MAGIC
# MAGIC Runs the whole alt-data pipeline on serverless compute from a **Git folder** clone of this repo, then
# MAGIC shows the main tables. Tables land in `workspace.panelcast.*`; vendor files and checkpoints land in the
# MAGIC Unity Catalog volume `/Volumes/workspace/panelcast/landing`.
# MAGIC
# MAGIC 1. Workspace -> Create -> Git folder -> paste the repo URL.
# MAGIC 2. Open this notebook from the Git folder, attach **Serverless**, and Run all (15-25 minutes at scale 0.5).

# COMMAND ----------

# MAGIC %pip install -q "pandas>=2.2,<3" rapidfuzz scikit-learn scipy pyyaml requests matplotlib

# COMMAND ----------

# MAGIC %pip install -q langgraph langchain-core langchain-anthropic

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

ROOT = os.path.abspath("..")  # the Git folder root (this notebook lives in notebooks/)
sys.path.insert(0, os.path.join(ROOT, "src"))

from panelcast.cli import main  # noqa: E402


def run(*stages: str, scale: str | None = None) -> None:
    args = ["--root", ROOT, "--env", "databricks", "run", *stages]
    main(args + (["--scale", scale] if scale else []))


# COMMAND ----------

# MAGIC %md ## 1. Simulate the vendor deliveries (synthetic card panel anchored to real SEC revenue)

# COMMAND ----------

run("simulate", scale="0.5")
display(dbutils.fs.ls("/Volumes/workspace/panelcast/landing/transactions"))

# COMMAND ----------

# MAGIC %md ## 2. Medallion pipeline: Auto Loader -> bronze -> silver -> entity resolution -> gold

# COMMAND ----------

run("reference", "bronze", "silver", "resolve", "gold")
display(spark.sql("""
    SELECT decision, method, COUNT(*) AS descriptors, ROUND(SUM(abs_spend) / 1e6, 2) AS spend_musd
    FROM workspace.panelcast.silver_descriptor_map GROUP BY ALL ORDER BY spend_musd DESC"""))

# COMMAND ----------

# MAGIC %md ## 3. Anomaly detection, population-weighted estimates, nowcasts, asset scorecard

# COMMAND ----------

run("anomalies", "estimates", "nowcast", "assets", "evaluate")
display(spark.table("workspace.panelcast.gold_anomaly_events"))

# COMMAND ----------

display(spark.sql("""
    SELECT model, ROUND(mae_pp, 2) AS mae_pp, ROUND(mape_revenue, 4) AS mape, ROUND(beats_naive, 3) AS beats_naive
    FROM workspace.panelcast.gold_nowcast_metrics WHERE segment = 'all' ORDER BY mae_pp"""))

# COMMAND ----------

display(spark.table("workspace.panelcast.gold_asset_scorecard"))

# COMMAND ----------

# MAGIC %md ## 4. Research notes (LangGraph agent; uses Claude if secret `panelcast/anthropic_api_key` exists)

# COMMAND ----------

run("agent", "report")
display(spark.table("workspace.panelcast.gold_research_notes").select("ticker", "mode", "note"))
