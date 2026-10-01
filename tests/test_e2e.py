"""Whole pipeline on a toy panel (a few minutes on a laptop): simulate -> ... -> report, then sanity checks."""

import pytest

from panelcast.config import Settings
from panelcast.context import Context
from panelcast.stages import run_stages

pytestmark = [pytest.mark.spark, pytest.mark.e2e]

SIM = {
    "start_date": "2018-01-01",
    "end_date": "2021-12-31",
    "backfill_until": "2021-11-30",
    "panel": {"n_panelists": 500},
    "anomalies": {
        "outages": [{"source": "SRC_A", "start": "2019-05-06", "days": 3}],
        "silent_departures": [],
        "duplicate_deliveries": [{"source": "SRC_B", "start": "2019-08-01", "days": 2}],
        "unit_errors": [{"source": "SRC_A", "start": "2020-02-03", "days": 5}],
        "descriptor_changes": [
            {"source": "SRC_A", "brand": "TARGET", "start": "2021-03-01", "template": "TRGT STR {store}"}
        ],
        "malformed_rate": 0.001,
    },
}


@pytest.fixture(scope="module")
def run(spark, root, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lakehouse")
    overrides = {
        "storage": {"local_root": str(tmp / "lakehouse")},
        "reports": {"dir": str(tmp / "reports")},
        "entity_resolution": {"llm": {"enabled": False}},
        "agent": {"llm": False},
    }
    ctx = Context(Settings(root=root, overrides=overrides, sim_overrides=SIM), _spark=spark)
    run_stages(ctx, ["all"])
    return ctx


def test_ingestion_reconciles_and_is_idempotent(run):
    store = run.store
    simulated = store.read("sim_truth", "transactions").count()
    bronze = store.read("bronze", "transactions").count()
    assert bronze == simulated
    silver = store.read("silver", "transactions").count()
    quarantined = store.read("silver", "quarantine").count()
    dups = store.read("silver", "duplicate_log").count()
    assert silver + quarantined + dups == bronze and dups > 0 and quarantined > 0
    run_stages(run, ["bronze"])  # no new files -> nothing re-ingested
    assert store.read("bronze", "transactions").count() == bronze


def test_quality_against_ground_truth(run):
    import json

    ev = json.loads((run.settings.reports_dir / "evaluation.json").read_text())
    er = ev["entity_resolution"]["ticker_by_spend"]
    assert er["precision"] >= 0.99 and er["recall"] >= 0.9
    assert ev["data_quality"]["quarantine_recall"] == 1.0
    detected = run.store.read_pandas("eval", "anomaly_detection").set_index("type")["detected"]
    assert detected["outage"] and detected["unit_error"] and detected["duplicate_delivery"]
    panel = ev["panel_estimators"]
    assert panel["raked"]["mae_pp"] < panel["raw"]["mae_pp"]


def test_downstream_outputs_exist(run):
    store = run.store
    assert store.read("gold", "nowcast_predictions").count() > 0
    assert store.read("gold", "asset_scorecard").count() == 3
    notes = store.read_pandas("gold", "research_notes")
    assert len(notes) == 20 and (notes["issues"] == 0).all()
    assert (run.settings.reports_dir / "RESULTS.md").exists()
