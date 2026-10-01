import pytest
from pyspark.sql import functions as F

from panelcast.pipeline.silver import dedupe, parse_and_validate, to_silver_columns

pytestmark = pytest.mark.spark

COLS = (
    "txn_id string, user_id string, txn_date string, post_date string, amount string, currency string, "
    "merchant_descriptor string, mcc string, source string, delivery string, _source_file string, "
    "_ingested_at timestamp"
)


def _rows(spark):
    import datetime as dt

    now = dt.datetime(2026, 1, 1)
    rows = [
        (
            "t1",
            "u1",
            "2024-03-01",
            "2024-03-02",
            "12.50",
            "USD",
            "TARGET 01234",
            "5310",
            "SRC_A",
            "daily-2024-03-02",
            "f1",
            now,
        ),
        (
            "t2",
            "u1",
            "2024-13-45",
            "2024-03-02",
            "9.99",
            "USD",
            "STARBUCKS",
            "5814",
            "SRC_A",
            "daily-2024-03-02",
            "f1",
            now,
        ),
        (
            "t3",
            "u2",
            "2024-03-01",
            "2024-03-02",
            "12.3O",
            "USD",
            "CHIPOTLE 12",
            "5814",
            "SRC_A",
            "daily-2024-03-02",
            "f1",
            now,
        ),
        (
            "t4",
            "u2",
            "2024-03-01",
            "2024-03-02",
            "99999",
            "USD",
            "BEST BUY",
            "5732",
            "SRC_A",
            "daily-2024-03-02",
            "f1",
            now,
        ),
        ("t5", "u3", "2024-03-01", "2024-03-02", "5.00", "USD", None, "5814", "SRC_A", "daily-2024-03-02", "f1", now),
        # the same transaction delivered twice: the re-delivery must lose even though it was ingested first
        (
            "t6",
            "u3",
            "2024-03-01",
            "2024-03-02",
            "7.00",
            "USD",
            "KFC #1",
            "5814",
            "SRC_B",
            "redelivery-2024-03-05",
            "f3",
            dt.datetime(2025, 12, 31),
        ),
        (
            "t6",
            "u3",
            "2024-03-01",
            "2024-03-02",
            "7.00",
            "USD",
            "KFC #1",
            "5814",
            "SRC_B",
            "daily-2024-03-02",
            "f2",
            now,
        ),
    ]
    return spark.createDataFrame(rows, COLS)


def test_expectations_route_bad_rows_to_quarantine(spark):
    parsed = parse_and_validate(_rows(spark), max_abs_amount=25000, currencies=["USD"])
    reasons = {r["txn_id"]: r["dq_reasons"] for r in parsed.select("txn_id", "dq_reasons").collect()}
    assert reasons["t1"] == []
    assert reasons["t2"] == ["bad_txn_date"]
    assert reasons["t3"] == ["bad_amount"]
    assert reasons["t4"] == ["amount_out_of_range"]
    assert reasons["t5"] == ["missing_descriptor"]


def test_dedupe_prefers_the_original_delivery(spark):
    parsed = parse_and_validate(_rows(spark), 25000, ["USD"]).where(F.size("dq_reasons") == 0)
    first, dups = dedupe(to_silver_columns(parsed))
    kept = first.where("txn_id = 't6'").first()
    assert kept["delivery"] == "daily-2024-03-02" and kept["descriptor_norm"] == "KFC"
    assert dups.count() == 1 and first.count() == 2
