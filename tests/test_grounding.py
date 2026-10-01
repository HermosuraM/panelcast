from panelcast.agent.grounding import extract_claims, verify_note

FACTS = {
    "nowcast": {
        "nowcast_yoy_pct": 4.31,
        "band_coverage_pct": 80.0,
        "band_80_low_pct": -0.51,
        "implied_revenue_usd_bn": 40.86,
        "quarter": "FY2026 Q2",
    },
    "history": [{"summary": {"mean_abs_error_pp": 2.94, "quarters_with_nowcast": 8}}],
}
GOOD = """## Test Co: FY2026 Q2 read
**Nowcast.** Growth of +4.3% YoY (80% band -0.5% to ...), about $40.86B in revenue.
**Drivers.** Panel spend. **Track record.** MAE 2.9 pp over 8 quarters.
**Data quality.** Clean. **Confidence.** Medium."""


def test_labels_years_and_dates_are_not_claims():
    raws = [c.raw for c in extract_claims("FY2026 Q2 ended 2026-06-30 on SRC_A, 5 sources, up 4.3% in 2025")]
    assert raws == ["4.3%"]


def test_grounded_note_passes():
    assert verify_note(GOOD, FACTS) == []


def test_invented_number_is_caught():
    issues = verify_note(GOOD.replace("+4.3%", "+7.9%"), FACTS)
    assert any("7.9%" in i for i in issues)


def test_rounding_must_match_the_digits_shown():
    assert verify_note(GOOD.replace("$40.86B", "$40.9B"), FACTS) == []  # legitimate rounding
    assert verify_note(GOOD.replace("$40.86B", "$41.2B"), FACTS) != []


def test_missing_section_and_advice_are_caught():
    issues = verify_note(GOOD.replace("**Confidence.** Medium.", "We recommend buying the stock."), FACTS)
    assert any("Confidence" in i for i in issues)
    assert verify_note(GOOD + " Price target $50.", FACTS) != []
