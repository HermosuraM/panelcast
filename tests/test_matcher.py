import pandas as pd
import pytest

from panelcast.entity_resolution.matcher import DescriptorMatcher
from panelcast.entity_resolution.normalize import normalize

BRANDS = pd.DataFrame(
    [
        {"brand": "TARGET", "ticker": "TGT", "mcc_list": "5310,5311,5411"},
        {"brand": "HOME DEPOT", "ticker": "HD", "mcc_list": "5200,5211"},
        {"brand": "ULTA BEAUTY", "ticker": "ULTA", "mcc_list": "5977,5999"},
        {"brand": "PIZZA HUT", "ticker": "YUM", "mcc_list": "5814,5812"},
        {"brand": "DOMINOS", "ticker": "DPZ", "mcc_list": "5814,5812"},
        {"brand": "DOORDASH", "ticker": "DASH", "mcc_list": "5812,5814"},
        {"brand": "UBER", "ticker": "UBER", "mcc_list": "4121,5812"},
        {"brand": "YARD HOUSE", "ticker": "DRI", "mcc_list": "5812,5813"},
    ]
)
ALIASES = pd.DataFrame(
    [
        {"brand": b, "alias_norm": normalize(a)}
        for b, aliases in {
            "TARGET": ["TARGET", "TARGET.COM"],
            "HOME DEPOT": ["THE HOME DEPOT", "HOMEDEPOT.COM"],
            "ULTA BEAUTY": ["ULTA BEAUTY", "ULTA"],
            "PIZZA HUT": ["PIZZA HUT"],
            "DOMINOS": ["DOMINOS PIZZA", "DOMINOS"],
            "DOORDASH": ["DOORDASH"],
            "UBER": ["UBER"],
            "YARD HOUSE": ["YARD HOUSE"],
        }.items()
        for a in aliases
    ]
)


@pytest.fixture(scope="module")
def matcher():
    return DescriptorMatcher(ALIASES, BRANDS, auto_accept=0.90, review_floor=0.72)


def one(matcher, raw, mcc):
    return matcher.match_many([normalize(raw)], [mcc])[0]


def test_exact_and_prefix(matcher):
    assert one(matcher, "TARGET 01234", 5310).brand == "TARGET"
    m = one(matcher, "THE HOME DEPOT #123 DALLAS TX", 5200)
    assert (m.brand, m.decision, m.method) == ("HOME DEPOT", "MATCH", "exact")


def test_spacing_variants_match(matcher):
    assert one(matcher, "YARDHOUSE #12", 5813).decision == "MATCH"
    assert one(matcher, "PIZZAHUT.COM 5X9Q2", 5814).brand == "PIZZA HUT"


def test_mcc_guard_blocks_lookalikes(matcher):
    m = one(matcher, "TARGET OPTICAL", 8043)
    assert m.decision != "MATCH" and not m.mcc_ok


def test_short_alias_needs_whole_token(matcher):
    assert one(matcher, "ULTRA CLEAN CAR WASH", 5977).decision == "NONE"


def test_merchant_of_record_rules(matcher):
    assert one(matcher, "IC* COSTCO BY INSTACART", 5411).decision == "NONE"
    dash = one(matcher, "DD *DOORDASH CHIPOTLE", 5812)
    assert (dash.brand, dash.method) == ("DOORDASH", "rule:merchant_of_record")
    assert one(matcher, "UBER *TRIP HELP.UBER.COM", 4121).brand == "UBER"
    assert one(matcher, "UBERCUTS SALON", 7230).decision == "NONE"


def test_unrelated_business_is_not_matched(matcher):
    assert one(matcher, "OFFICE DEPOT #12", 5943).decision == "NONE"


def test_ambiguous_candidates_go_to_review_not_match(matcher):
    m = one(matcher, "DPZ PIZZA 1234", 5814)
    assert m.decision in {"REVIEW", "NONE"}


def test_match_frame_shape(matcher):
    pdf = pd.DataFrame({"descriptor_norm": ["TARGET", "", "OFFICE DEPOT"], "mcc": [5310, 5999, 5943]})
    out = matcher.match_frame(pdf)
    assert list(out["decision"]) == ["MATCH", "NONE", "NONE"]
    assert out.loc[0, "ticker"] == "TGT"
