import datetime as dt

from panelcast.reference.edgar import quarterly_revenue

D = dt.date.fromisoformat
CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]


def fact(start, end, val, filed, form="10-Q", fy=2024, fp="Q1", accn="0001"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fy": fy, "fp": fp, "accn": accn}


def doc(by_concept: dict) -> dict:
    return {"facts": {"us-gaap": {c: {"units": {"USD": rows}} for c, rows in by_concept.items()}}}


RETAILER = [  # 53-week fiscal year ending 2024-02-03 (like Target); Q4 only in the 10-K annual total
    fact("2023-01-29", "2023-04-29", 100.0, "2023-05-26"),
    fact("2023-04-30", "2023-07-29", 110.0, "2023-08-25", fp="Q2"),
    fact("2023-07-30", "2023-10-28", 105.0, "2023-11-22", fp="Q3"),
    fact("2023-01-29", "2024-02-03", 455.0, "2024-03-13", form="10-K", fp="FY"),
    # next year's 10-Q repeats Q1 2023 as a comparative, restated, and labeled with the NEW fiscal year
    fact("2023-01-29", "2023-04-29", 101.0, "2024-05-31", fy=2025, fp="Q1", accn="0002"),
    fact("2024-02-04", "2024-05-04", 103.0, "2024-05-31", fy=2025, fp="Q1", accn="0002"),
]


def test_q4_is_derived_from_the_annual_total():
    q, issues = quarterly_revenue(doc({"Revenues": RETAILER}), CONCEPTS, D("2016-01-01"))
    q4 = q[q["derived"]].iloc[0]
    assert (q4["start"], q4["end"]) == (D("2023-10-29"), D("2024-02-03"))
    assert q4["days"] == 98  # 14-week quarter in a 53-week year
    assert q4["val"] == 455.0 - (100 + 110 + 105)
    assert q4["fiscal_year"] == 2024 and q4["fiscal_quarter"] == 4
    assert not issues


def test_first_filed_value_wins_and_restatements_are_flagged():
    q, _ = quarterly_revenue(doc({"Revenues": RETAILER}), CONCEPTS, D("2016-01-01"))
    q1 = q[q["start"] == D("2023-01-29")].iloc[0]
    assert q1["val"] == 100.0 and q1["val_latest"] == 101.0 and q1["restated"]


def test_periods_are_labeled_by_dates_not_by_fy_fp_tags():
    q, _ = quarterly_revenue(doc({"Revenues": RETAILER}), CONCEPTS, D("2016-01-01"))
    nxt = q[q["start"] == D("2024-02-04")].iloc[0]
    assert (nxt["fiscal_year"], nxt["fiscal_quarter"]) == (2025, 1)
    assert list(q["start"]) == sorted(q["start"])


def test_fallback_concept_only_when_consistent():
    new = [fact("2018-01-01", "2018-03-31", 50.0, "2018-05-01"), fact("2018-04-01", "2018-06-30", 52.0, "2018-08-01")]
    old_ok = [
        fact("2017-10-01", "2017-12-31", 48.0, "2018-02-01"),
        fact("2018-01-01", "2018-03-31", 50.0, "2018-05-01"),
    ]
    old_bad = [
        fact("2017-10-01", "2017-12-31", 48.0, "2018-02-01"),
        fact("2018-01-01", "2018-03-31", 57.0, "2018-05-01"),
    ]
    filled, _ = quarterly_revenue(doc({"Revenues": new, "SalesRevenueNet": old_ok}), CONCEPTS, D("2016-01-01"))
    assert D("2017-10-01") in set(filled["start"])
    skipped, _ = quarterly_revenue(doc({"Revenues": new, "SalesRevenueNet": old_bad}), CONCEPTS, D("2016-01-01"))
    assert D("2017-10-01") not in set(skipped["start"])


def test_filed_date_is_first_public_across_concepts():
    # the original 10-Q tagged SalesRevenueNet; a later 10-K re-tagged the same quarter as Revenues
    old_tag = [fact("2017-10-01", "2017-12-31", 48.0, "2018-02-01")]
    new_tag = [
        fact("2017-10-01", "2017-12-31", 48.0, "2018-11-16", form="10-K"),
        fact("2018-01-01", "2018-03-31", 50.0, "2018-05-01"),
        fact("2018-04-01", "2018-06-30", 52.0, "2018-08-01"),
    ]
    q, _ = quarterly_revenue(doc({"Revenues": new_tag, "SalesRevenueNet": old_tag}), CONCEPTS, D("2016-01-01"))
    assert q.loc[q["start"] == D("2017-10-01"), "filed"].iloc[0] == D("2018-02-01")


def test_year_ending_in_early_january_keeps_prior_year_label():
    rows = [  # Domino's-style: 12-week quarters, 16-week Q4, fiscal 2022 ends on 2023-01-01
        fact("2022-01-03", "2022-03-27", 1.0, "2022-04-28"),
        fact("2022-03-28", "2022-06-19", 1.0, "2022-07-21"),
        fact("2022-06-20", "2022-09-11", 1.0, "2022-10-13"),
        fact("2022-01-03", "2023-01-01", 4.5, "2023-02-23", form="10-K"),
    ]
    q, _ = quarterly_revenue(doc({"Revenues": rows}), CONCEPTS, D("2016-01-01"))
    assert set(q["fiscal_year"]) == {2022}
    assert sorted(q["days"]) == [84, 84, 84, 112]
