"""Write one pre-earnings research note per ticker for its latest backtested quarter.

The note is generated as of the nowcast date (before the 10-Q/10-K existed); a deterministic post-mortem
line comparing the nowcast with the revenue the company later reported is appended afterwards.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from panelcast.agent.graph import build_graph
from panelcast.agent.tools import ResearchData
from panelcast.config import Settings
from panelcast.llm import chat_model, llm_enabled
from panelcast.store import TableStore

log = logging.getLogger(__name__)


def post_mortem(target: pd.Series) -> str:
    actual = 100 * (np.exp(target["rev_yoy"]) - 1)
    pred = 100 * (np.exp(target["pred_yoy"]) - 1)
    return (
        f"\n\n> **Post-mortem (added after the filing on {target['filed']}):** reported growth was {actual:+.1f}% "
        f"vs. the {pred:+.1f}% nowcast ({pred - actual:+.1f} pp)."
    )


def write_research_notes(
    settings: Settings, store: TableStore, tickers: list[str] | None = None, use_llm: bool | None = None
) -> pd.DataFrame:
    cfg = settings["agent"]
    data = ResearchData.load(store)
    use_llm = llm_enabled(cfg.get("llm", "auto")) if use_llm is None else use_llm
    llm = chat_model(cfg["model"], effort=cfg["effort"]) if use_llm else None
    out_dir = settings.reports_dir / "notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker in tickers or settings.tickers:
        target = data.latest_target(ticker)
        graph = build_graph(
            data.as_of(target["nowcast_date"]), llm, int(cfg["max_revisions"]), int(cfg["max_tool_rounds"])
        )
        state = graph.invoke(
            {"ticker": ticker}, config={"configurable": {"thread_id": f"{ticker}-{target['period_end']}"}}
        )
        note = state["note"] + post_mortem(target)
        (out_dir / f"{ticker}.md").write_text(note, encoding="utf-8")
        rows.append(
            {
                "ticker": ticker,
                "period_end": target["period_end"],
                "nowcast_date": target["nowcast_date"],
                "mode": state.get("mode"),
                "revisions": state.get("revisions", 0),
                "tool_rounds": state.get("tool_rounds", 0),
                "issues": len(state.get("issues", [])),
                "note": note,
            }
        )
        log.info("%s note written (%s, %d revisions)", ticker, state.get("mode"), state.get("revisions", 0))
    notes = pd.DataFrame(rows)
    store.write_pandas(notes, "gold", "research_notes")
    return notes
