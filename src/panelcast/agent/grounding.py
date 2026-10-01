"""Deterministic checks on an LLM-written research note.

Every figure in the note must trace back to a number a tool returned (allowing for the rounding the note
itself shows), the note must contain the required sections, and it must not read as investment advice.
Failures go back to the model as specific feedback; the LLM never grades its own work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

REQUIRED_SECTIONS = ["Nowcast", "Drivers", "Track record", "Data quality", "Confidence"]
ADVICE = re.compile(
    r"\b(buy|sell|hold)\s+(rating|recommendation|the stock|shares)\b|\bprice target\b|\b(overweight|underweight)\b"
    r"|\brecommend (buying|selling|investors)\b|\bshould (buy|sell)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"(?<![\w.$])(?P<sign>[-+−])?(?P<dollar>\$)?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"\s?(?P<unit>%|pp\b|bps\b|bn\b|B\b|billion\b|M\b|mn\b|million\b|k\b)?",
    re.IGNORECASE,
)
SCALE = {"bn": 1e9, "b": 1e9, "billion": 1e9, "m": 1e6, "mn": 1e6, "million": 1e6, "k": 1e3}


@dataclass(frozen=True)
class Claim:
    raw: str
    value: float  # absolute value in natural units (percent points for pct, dollars for usd)
    unit: str  # pct | usd | count
    tolerance: float  # half a unit of the last digit the note shows


def _skip(text: str, m: re.Match) -> bool:
    start, end = m.span()
    before, after = text[max(0, start - 4) : start], text[end : end + 1]
    plain = not (m.group("unit") or m.group("dollar") or m.group("dec"))
    value = float(m.group("num").replace(",", ""))
    if re.search(r"(FY|Q|SRC_|H|#)$", before, re.IGNORECASE):  # FY2026, Q3, SRC_A, H1, #1 labels
        return True
    if after in ("-", "/") or before.endswith(("-", "/")):  # dates like 2026-03-31
        return True
    return plain and (1990 <= value <= 2040 or value < 10)  # bare years, small ordinals ("5 sources")


def extract_claims(text: str) -> list[Claim]:
    claims = []
    for m in NUMBER.finditer(text):
        if _skip(text, m):
            continue
        dec = m.group("dec") or ""
        value = abs(float((m.group("num") + dec).replace(",", "")))
        half_ulp = 0.5 * 10 ** -(len(dec) - 1) if dec else 0.5
        unit = (m.group("unit") or "").lower()
        if unit in ("%", "pp"):
            claims.append(Claim(m.group(0).strip(), value, "pct", half_ulp))
        elif unit == "bps":
            claims.append(Claim(m.group(0).strip(), value / 100, "pct", half_ulp / 100))
        elif m.group("dollar") or unit in SCALE:
            scale = SCALE.get(unit, 1.0)
            claims.append(Claim(m.group(0).strip(), value * scale, "usd", half_ulp * scale))
        else:
            claims.append(Claim(m.group(0).strip(), value, "count", half_ulp))
    return claims


def fact_values(obj: Any) -> list[float]:
    """Every number inside the tool outputs (nested dicts / lists / numeric strings)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in fact_values(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in fact_values(v)]
    if isinstance(obj, str):
        return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", obj)]
    return []


def is_grounded(claim: Claim, facts: list[float]) -> bool:
    tol = claim.tolerance * 1.0001 + 1e-9
    for f in facts:
        a = abs(f)
        candidates = (a, a * 1e9, a * 1e6) if claim.unit == "usd" else (a,)
        if any(abs(claim.value - c) <= tol for c in candidates):
            return True
    return False


def verify_note(note: str, facts: dict) -> list[str]:
    issues = []
    numbers = fact_values(facts)
    for claim in extract_claims(note):
        if not is_grounded(claim, numbers):
            issues.append(f"Figure '{claim.raw}' does not match any number returned by the tools.")
    for section in REQUIRED_SECTIONS:
        if section.lower() not in note.lower():
            issues.append(f"Missing required section: {section}.")
    if ADVICE.search(note):
        issues.append("Remove investment-advice language (ratings, price targets, buy/sell calls).")
    return issues
