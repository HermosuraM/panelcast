"""LLM adjudication of gray-zone descriptors with Claude structured output (LangChain).

Only the review queue reaches the model (typically a few hundred descriptors, highest spend first), the
answer is constrained to a JSON schema, every brand is validated against the universe, and decisions are
cached in `silver.er_llm_decisions` so a descriptor is never paid for twice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

SYSTEM = """You resolve card-statement merchant descriptors to brands for an alternative-data research team.

Each descriptor comes with its normalized text, merchant category code (MCC), and the matcher's best
candidate brands. Choose a brand ONLY if the purchase was made at that brand (the brand is the merchant of
record). Answer null when the descriptor is a different business with a similar name, a marketplace or
delivery service buying on someone else's behalf, or when you are not confident.

Precision matters more than recall: a wrong match silently corrupts that company's spend signal, while a
miss only shrinks the sample. Brand names in your answer must be copied exactly from the brand list.

Brand list (brand | ticker | MCCs the brand normally transacts under):
{brand_list}"""


class Decision(BaseModel):
    id: int = Field(description="The descriptor id from the request")
    brand: str | None = Field(description="Exact brand name from the brand list, or null")
    confidence: float = Field(description="Probability the brand is correct, between 0 and 1")
    reason: str = Field(description="At most 20 words")


class Decisions(BaseModel):
    decisions: list[Decision]


def brand_list(brands: pd.DataFrame) -> str:
    return "\n".join(f"{r.brand} | {r.ticker} | {r.mcc_list}" for r in brands.itertuples())


def batch_prompt(batch: pd.DataFrame) -> str:
    lines = ["Decide each descriptor:", "id | descriptor | MCC | matcher candidates (score)"]
    for r in batch.itertuples():
        cands = f"{r.brand} ({r.score:.2f})"
        if isinstance(r.runner_up, str) and r.runner_up:
            cands += f", {r.runner_up} ({r.runner_up_score:.2f})"
        lines.append(f"{r.Index} | {r.descriptor_norm} | {r.mcc} | {cands}")
    return "\n".join(lines)


def make_decider(model: str, effort: str, brands: pd.DataFrame) -> Callable[[str], Decisions | None]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from panelcast.llm import chat_model

    structured = chat_model(model, effort=effort, max_tokens=8000).with_structured_output(
        Decisions, method="json_schema", include_raw=True
    )
    system = SystemMessage(SYSTEM.format(brand_list=brand_list(brands)))

    def decide(prompt: str) -> Decisions | None:
        out = structured.invoke([system, HumanMessage(prompt)])
        if out.get("parsing_error") is not None:
            log.warning("structured output failed: %s", out["parsing_error"])
            return None
        return out["parsed"]

    return decide


def adjudicate(
    queue: pd.DataFrame,
    brands: pd.DataFrame,
    decide: Callable[[str], Decisions | None],
    batch_size: int,
    min_confidence: float,
) -> pd.DataFrame:
    """Return one row per queued descriptor: llm_brand (validated or None), llm_confidence, llm_reason."""
    valid = set(brands["brand"])
    queue = queue.reset_index(drop=True)
    rows = []
    for start in range(0, len(queue), batch_size):
        batch = queue.iloc[start : start + batch_size]
        result = decide(batch_prompt(batch))
        by_id = {d.id: d for d in result.decisions} if result else {}
        for idx, r in batch.iterrows():
            d = by_id.get(int(idx))
            brand = d.brand if d and d.brand in valid and d.confidence >= min_confidence else None
            rows.append(
                {
                    "descriptor_norm": r.descriptor_norm,
                    "mcc": int(r.mcc),
                    "llm_brand": brand,
                    "llm_confidence": float(d.confidence) if d else None,
                    "llm_reason": d.reason if d else "no decision returned",
                }
            )
    return pd.DataFrame(rows)
