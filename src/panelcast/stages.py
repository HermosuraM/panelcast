"""Pipeline stage registry. Each stage reads/writes tables through ctx.store, so stages can run
independently (locally, or as separate tasks in a Databricks job)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from panelcast.context import Context

log = logging.getLogger(__name__)


def _simulate(ctx: Context) -> None:
    from panelcast.simulate.generate import run_simulation

    run_simulation(ctx.settings, ctx.store)


def _reference(ctx: Context) -> None:
    from panelcast.pipeline.reference import load_reference

    load_reference(ctx.settings, ctx.store)


def _bronze(ctx: Context) -> None:
    from panelcast.pipeline.bronze import ingest

    ingest(ctx.settings, ctx.store)


def _silver(ctx: Context) -> None:
    from panelcast.pipeline.silver import build_silver

    build_silver(ctx.settings, ctx.store)


def _resolve(ctx: Context) -> None:
    from panelcast.entity_resolution.stage import resolve

    resolve(ctx.settings, ctx.store)


def _gold(ctx: Context) -> None:
    from panelcast.pipeline.gold import build_gold_base

    build_gold_base(ctx.settings, ctx.store)


def _anomalies(ctx: Context) -> None:
    from panelcast.anomaly.stage import detect

    detect(ctx.settings, ctx.store)


def _estimates(ctx: Context) -> None:
    from panelcast.pipeline.estimates import build_estimates

    build_estimates(ctx.settings, ctx.store)


def _nowcast(ctx: Context) -> None:
    from panelcast.modeling.nowcast import run_backtest

    run_backtest(ctx.settings, ctx.store)


def _assets(ctx: Context) -> None:
    from panelcast.modeling.asset_eval import evaluate_assets

    evaluate_assets(ctx.settings, ctx.store)


def _evaluate(ctx: Context) -> None:
    from panelcast.evaluation import evaluate_against_truth

    evaluate_against_truth(ctx.settings, ctx.store)


def _agent(ctx: Context) -> None:
    from panelcast.agent.run import write_research_notes

    write_research_notes(ctx.settings, ctx.store)


def _report(ctx: Context) -> None:
    from panelcast.report import build_report

    build_report(ctx.settings, ctx.store)


STAGES: dict[str, Callable[[Context], None]] = {
    "simulate": _simulate,
    "reference": _reference,
    "bronze": _bronze,
    "silver": _silver,
    "resolve": _resolve,
    "gold": _gold,
    "anomalies": _anomalies,
    "estimates": _estimates,
    "nowcast": _nowcast,
    "assets": _assets,
    "evaluate": _evaluate,
    "agent": _agent,
    "report": _report,
}


def run_stages(ctx: Context, names: list[str]) -> None:
    order = list(STAGES) if "all" in names else [n for n in STAGES if n in names]
    for name in order:
        t0 = time.time()
        log.info("=== stage: %s ===", name)
        STAGES[name](ctx)
        log.info("=== %s finished in %.1fs ===", name, time.time() - t0)
