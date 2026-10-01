"""LangGraph analyst agent: gather -> analyst <-> tools -> verify -> (revise -> analyst) -> finalize.

gather    deterministic: pull the core facts (company, nowcast, history, data quality)
analyst   Claude with read-only tools drafts the note (or a deterministic template when offline)
tools     executes tool calls and adds every result to the fact base used for verification
verify    deterministic grounding check (every figure must match a tool output) + section/compliance rules
revise    sends the specific problems back to the analyst (bounded number of rounds)
finalize  publishes the note; if issues remain, falls back to the always-grounded template
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from panelcast.agent.grounding import verify_note
from panelcast.agent.tools import ResearchData, make_tools
from panelcast.llm import was_refused

log = logging.getLogger(__name__)

SYSTEM = """You are an equity research analyst at an alternative-data firm. You write short pre-earnings notes
that tell portfolio managers what card-panel data implies for a company's reported revenue.

Rules:
- Use only numbers returned by your tools. Never compute new figures, never round differently than one
  decimal place, and never invent numbers. If something is not in the tool output, leave it out.
- Report growth as year-over-year percentages for the company's own fiscal quarter.
- Be explicit about uncertainty (the 80% band, the track record, data-quality problems in the quarter).
- No investment advice: no ratings, price targets, or buy/sell language.
- Markdown, under 250 words, with these bold section labels in order: **Nowcast.** **Drivers.**
  **Track record.** **Data quality.** **Confidence.** Start with a one-line H2 headline."""

TASK = """Write the pre-earnings note for {ticker} as of {as_of}. Core data from the pipeline is below; call
tools if you need more (for example run_sql over `quarters` / `predictions`).

{facts}"""


class NoteState(TypedDict, total=False):
    ticker: str
    messages: Annotated[list[AnyMessage], add_messages]
    facts: Annotated[dict[str, Any], operator.or_]
    draft: str
    issues: list[str]
    revisions: int
    tool_rounds: int
    mode: str
    note: str


def template_note(facts: dict) -> str:
    """Deterministic note built only from tool outputs (used offline and as the safety fallback)."""
    o, n, h, d = facts["overview"], facts["nowcast"], facts["history"], facts["data_quality"]
    summary = next((x["summary"] for x in h if "summary" in x), {})
    events = d["events_in_quarter"]
    er = d["entity_resolution"]
    band_low, band_high = n["band_80_low_pct"], n["band_80_high_pct"]
    accel = "an acceleration" if n["nowcast_yoy_pct"] > (n["last_reported_yoy_pct"] or 0) else "a deceleration"
    lines = [
        f"## {o['name']} ({o['ticker']}): {n['quarter']} card-panel read",
        f"*As of {n['nowcast_date']}, for the fiscal quarter {n['period_start']} to {n['period_end']}.*",
        "",
        f"**Nowcast.** The ensemble model points to revenue growth of {n['nowcast_yoy_pct']:+.1f}% YoY"
        + (
            f" ({n['band_coverage_pct']:.0f}% band {band_low:+.1f}% to {band_high:+.1f}%)"
            if band_low is not None
            else ""
        )
        + f", implying revenue of about ${n['implied_revenue_usd_bn']:.2f}B.",
        f"**Drivers.** Raked panel spend grew {n['panel_raked_spend_yoy_pct']:+.1f}% YoY over the same fiscal days,"
        f" versus {n['last_reported_yoy_pct']:+.1f}% reported growth last quarter, i.e. {accel}.",
    ]
    if summary.get("mean_abs_error_pp") is not None:
        lines.append(
            f"**Track record.** Across the last {summary['quarters_with_nowcast']} nowcasted quarters the mean"
            f" absolute error was {summary['mean_abs_error_pp']:.1f} pp."
        )
    else:
        lines.append("**Track record.** Not enough history yet to score this ticker.")
    if events:
        desc = "; ".join(f"{e['type'].replace('_', ' ')} on {e['source']} ({e['handling']})" for e in events)
        lines.append(f"**Data quality.** Issues touched this quarter: {desc}.")
    else:
        lines.append("**Data quality.** No outages, unit errors or tagging breaks touched this quarter.")
    by_method = er.get("matched_spend_by_method_pct") or {}
    if by_method:
        method, share = max(by_method.items(), key=lambda kv: kv[1])
        lines[-1] += (
            f" {share:.1f}% of matched spend came from {method.replace('rule:', '').replace('_', '-')}"
            f" matches; {er['pending_review_share_pct']:.1f}% of potential spend awaits review."
        )
    width = (band_high - band_low) if band_low is not None else None
    mae = summary.get("mean_abs_error_pp")
    level = (
        "High"
        if width is not None and width < 6 and mae is not None and mae < 3 and not events
        else ("Low" if width is None or width > 12 or (mae or 0) > 6 else "Medium")
    )
    lines.append(f"**Confidence.** {level}, based on the band width, the track record and the data-quality checks.")
    return "\n".join(lines)


def build_graph(data: ResearchData, llm=None, max_revisions: int = 2, max_tool_rounds: int = 6, checkpointer=None):
    tools = make_tools(data)
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools) if llm is not None else None

    def gather(state: NoteState) -> dict:
        t = state["ticker"]
        facts = {
            "overview": by_name["get_company_overview"].invoke({"ticker": t}),
            "nowcast": by_name["get_nowcast"].invoke({"ticker": t}),
            "history": by_name["get_quarter_history"].invoke({"ticker": t}),
            "data_quality": by_name["get_data_quality"].invoke({"ticker": t}),
        }
        task = TASK.format(ticker=t, as_of=facts["nowcast"]["nowcast_date"], facts=json.dumps(facts, indent=1))
        return {
            "facts": facts,
            "messages": [SystemMessage(SYSTEM), HumanMessage(task)],
            "revisions": 0,
            "tool_rounds": 0,
        }

    def analyst(state: NoteState) -> dict:
        if model is None:
            return {"draft": template_note(state["facts"]), "mode": "template"}
        response: AIMessage = model.invoke(state["messages"])
        if was_refused(response):
            log.warning("model declined for %s; using the template", state["ticker"])
            return {"messages": [response], "draft": template_note(state["facts"]), "mode": "template_after_refusal"}
        if response.tool_calls:
            if state.get("tool_rounds", 0) < max_tool_rounds:
                return {"messages": [response]}
            # out of tool budget: do not leave an unanswered tool call in the transcript
            return {"draft": template_note(state["facts"]), "mode": "template_tool_limit"}
        return {"messages": [response], "draft": response.text, "mode": "llm"}

    def run_tools(state: NoteState) -> dict:
        last = state["messages"][-1]
        outputs, facts = [], {}
        for call in last.tool_calls:
            tool_fn = by_name.get(call["name"])
            result = tool_fn.invoke(call["args"]) if tool_fn else {"error": f"unknown tool {call['name']}"}
            facts[f"{call['name']}:{json.dumps(call['args'], sort_keys=True)}"] = result
            outputs.append(ToolMessage(json.dumps(result, default=str), tool_call_id=call["id"]))
        return {"messages": outputs, "facts": facts, "tool_rounds": state.get("tool_rounds", 0) + 1}

    def verify(state: NoteState) -> dict:
        return {"issues": verify_note(state["draft"], state["facts"])}

    def revise(state: NoteState) -> dict:
        problems = "\n".join(f"- {i}" for i in state["issues"])
        msg = f"The draft failed verification:\n{problems}\nRewrite the complete note fixing every problem."
        return {"messages": [HumanMessage(msg)], "revisions": state.get("revisions", 0) + 1, "draft": ""}

    def finalize(state: NoteState) -> dict:
        if state["issues"]:
            log.warning(
                "%s: %d unresolved issues; publishing the template note instead", state["ticker"], len(state["issues"])
            )
            return {"note": template_note(state["facts"]), "mode": f"{state.get('mode')}_fallback"}
        return {"note": state["draft"]}

    def after_analyst(state: NoteState) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if not state.get("draft") and isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "verify"

    def after_verify(state: NoteState) -> str:
        if state["issues"] and state.get("mode") == "llm" and state.get("revisions", 0) < max_revisions:
            return "revise"
        return "finalize"

    g = StateGraph(NoteState)
    for name, fn in (
        ("gather", gather),
        ("analyst", analyst),
        ("tools", run_tools),
        ("verify", verify),
        ("revise", revise),
        ("finalize", finalize),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "gather")
    g.add_edge("gather", "analyst")
    g.add_conditional_edges("analyst", after_analyst, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "analyst")
    g.add_conditional_edges("verify", after_verify, {"revise": "revise", "finalize": "finalize"})
    g.add_edge("revise", "analyst")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
