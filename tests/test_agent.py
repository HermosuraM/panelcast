import datetime as dt

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from panelcast.agent.graph import build_graph, template_note
from panelcast.agent.grounding import verify_note
from panelcast.agent.tools import ResearchData, make_tools
from panelcast.entity_resolution.llm_adjudicator import Decision, Decisions, adjudicate


def _data() -> ResearchData:
    starts = pd.date_range("2024-01-01", periods=8, freq="QS").date
    ends = [(pd.Timestamp(s) + pd.offsets.QuarterEnd(0)).date() for s in starts]
    quarters = pd.DataFrame(
        {
            "ticker": "TEST",
            "fiscal_year": [2024] * 4 + [2025] * 4,
            "fiscal_quarter": [1, 2, 3, 4] * 2,
            "period_start": starts,
            "period_end": ends,
            "period_days": [91] * 8,
            "revenue": np.linspace(10e9, 11.5e9, 8),
            "revenue_prior": np.linspace(9.5e9, 10.8e9, 8),
            "rev_yoy": np.linspace(0.03, 0.06, 8),
            "raked_yoy": np.linspace(0.035, 0.065, 8),
            "filed": [e + dt.timedelta(days=35) for e in ends],
            "lead_days": 28,
            "complete": True,
        }
    )
    rows = []
    for _, q in quarters.iloc[4:].iterrows():
        for model, bump in (("naive", -0.01), ("ridge", 0.002), ("gbm", 0.004), ("ensemble", 0.003)):
            pred = q["rev_yoy"] + bump
            rows.append(
                {
                    **q.to_dict(),
                    "nowcast_date": q["period_end"] + dt.timedelta(days=7),
                    "model": model,
                    "rev_yoy_lag1": q["rev_yoy"] - 0.004,
                    "pred_yoy": pred,
                    "pred_lo": pred - 0.02,
                    "pred_hi": pred + 0.02,
                    "pred_revenue": q["revenue_prior"] * np.exp(pred),
                    "error_pp": 100 * bump,
                    "ape": abs(bump),
                }
            )
    return ResearchData(
        companies=pd.DataFrame({"ticker": ["TEST"], "name": ["Test Co"], "sector": ["Retail"]}),
        brands=pd.DataFrame({"brand": ["TESTMART"], "ticker": ["TEST"], "valid_from": [dt.date(1900, 1, 1)]}),
        quarters=quarters,
        predictions=pd.DataFrame(rows),
        events=pd.DataFrame(
            {
                "type": ["outage"],
                "source": ["SRC_A"],
                "ticker": [None],
                "start": [dt.date(2025, 11, 3)],
                "end": [dt.date(2025, 11, 4)],
                "n_days": [2],
                "action": ["exclude"],
            }
        ),
        scorecard=pd.DataFrame(
            {
                "asset": ["card_panel_raked"],
                "tickers_covered": [1],
                "median_corr_ex_covid": [0.8],
                "oos_skill_vs_naive": [0.4],
                "median_lead_days": [26.0],
                "verdict": ["Strong"],
            }
        ),
        er_coverage=pd.DataFrame(
            {
                "ticker": ["TEST", "TEST"],
                "decision": ["MATCH", "REVIEW"],
                "method": ["exact", "fuzzy"],
                "abs_spend": [990.0, 10.0],
            }
        ),
    )


def _view():
    data = _data()
    return data.as_of(data.latest_target("TEST")["nowcast_date"])


def test_point_in_time_view_hides_the_actual():
    view = _view()
    tools = {t.name: t for t in make_tools(view)}
    rows = tools["run_sql"].invoke({"query": "SELECT period_end, rev_yoy FROM predictions WHERE model = 'ensemble'"})
    assert rows[-1]["rev_yoy"] is None  # filed after the nowcast date -> masked
    assert all(r["rev_yoy"] is not None for r in rows[:-1])
    assert "error" in tools["run_sql"].invoke({"query": "DROP TABLE quarters"})[0]


def test_template_mode_produces_a_verified_note():
    state = build_graph(_view()).invoke({"ticker": "TEST"}, config={"configurable": {"thread_id": "t1"}})
    assert state["mode"] == "template" and state["issues"] == []
    assert "Test Co (TEST)" in state["note"] and "outage on SRC_A" in state["note"]


class ScriptedChat:
    """Minimal stand-in for a tool-calling chat model: replays scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_llm_mode_calls_tools_and_revises_an_ungrounded_draft():
    view = _view()
    facts_nowcast = {t.name: t for t in make_tools(view)}["get_nowcast"].invoke({"ticker": "TEST"})
    good = template_note(
        {
            "overview": {"name": "Test Co", "ticker": "TEST"},
            "nowcast": facts_nowcast,
            "history": [{"summary": {}}],
            "data_quality": {"events_in_quarter": [], "entity_resolution": {}},
        }
    )
    bad = good.replace(f"{facts_nowcast['nowcast_yoy_pct']:+.1f}%", "+42.0%")
    fake = ScriptedChat(
        [
            AIMessage(content="", tool_calls=[{"name": "get_asset_scorecard", "args": {}, "id": "call_1"}]),
            AIMessage(content=bad),
            AIMessage(content=good),
        ]
    )
    state = build_graph(view, llm=fake).invoke({"ticker": "TEST"}, config={"configurable": {"thread_id": "t2"}})
    assert state["note"] == good and state["mode"] == "llm"
    assert state["revisions"] == 1 and state["tool_rounds"] == 1
    assert any(k.startswith("get_asset_scorecard") for k in state["facts"])
    assert "+42.0%" in fake.calls[2][-1].content  # the verifier's feedback names the bad figure
    assert verify_note(good, state["facts"]) == []


def test_adjudicator_validates_brands_and_confidence():
    queue = pd.DataFrame(
        {
            "descriptor_norm": ["TRGT STR", "STARBRIGHT COFFEE", "WMT PLUS"],
            "mcc": [5310, 5814, 5310],
            "brand": ["TARGET", "STARBUCKS", "WALMART"],
            "score": [0.57, 0.79, 0.43],
            "runner_up": [None, None, None],
            "runner_up_score": [0.0, 0.0, 0.0],
        }
    )
    brands = pd.DataFrame(
        {
            "brand": ["TARGET", "STARBUCKS", "WALMART"],
            "ticker": ["TGT", "SBUX", "WMT"],
            "mcc_list": ["5310", "5814", "5310"],
        }
    )

    def decide(prompt: str) -> Decisions:
        assert "TRGT STR" in prompt
        return Decisions(
            decisions=[
                Decision(id=0, brand="TARGET", confidence=0.95, reason="abbreviation"),
                Decision(id=1, brand="STARBUCKS", confidence=0.4, reason="unsure"),
                Decision(id=2, brand="WALMART PLUS INC", confidence=0.9, reason="not a brand"),
            ]
        )

    out = adjudicate(queue, brands, decide, batch_size=20, min_confidence=0.8)
    assert list(out["llm_brand"]) == ["TARGET", None, None]


@pytest.mark.parametrize("missing", [True, False])
def test_adjudicator_handles_missing_answers(missing):
    queue = pd.DataFrame(
        {
            "descriptor_norm": ["A"],
            "mcc": [1],
            "brand": ["X"],
            "score": [0.8],
            "runner_up": [None],
            "runner_up_score": [0.0],
        }
    )
    brands = pd.DataFrame({"brand": ["X"], "ticker": ["XX"], "mcc_list": ["1"]})
    out = adjudicate(queue, brands, lambda p: None if missing else Decisions(decisions=[]), 20, 0.8)
    assert out.loc[0, "llm_brand"] is None
