import pytest

from panelcast.entity_resolution.normalize import normalize

CASES = [
    ("POS PURCHASE STARBUCKS STORE 01234 SEATTLE WA", "STARBUCKS"),
    ("DEBIT CARD PURCHASE XXXXX1234 WAL-MART #1234", "WAL MART"),
    ("AMZN MKTP US*2K4L91XZ3", "AMZN MKTP US"),
    ("AMAZON.COM*2K4L91XZ3 AMZN.COM/BILL WA", "AMAZON AMZN"),
    ("MCDONALD'S F1234", "MCDONALDS"),
    ("Mcdonald'S M1234 Houston", "MCDONALDS"),
    ("7-Eleven 1669", "SEVEN ELEVEN"),
    ("T.J. MAXX 123 DALLAS TX", "T J MAXX"),
    ("UBER *TRIP HELP.UBER.COM", "UBER UBER"),
    ("NETFLIX.COM LOS GATOS CA", "NETFLIX"),
    ("PAYPAL *NETFLIX", "NETFLIX"),
    ("DD *DOORDASH PANERA", "DOORDASH DOORDASH PANERA"),
    ("IC* COSTCO BY INSTACART", "INSTACART COSTCO BY INSTACART"),
    ("SQ *SUNRISE TACOS AUSTIN TX", "SUNRISE TACOS"),
    ("THE HOME DEPOT #6543", "HOME DEPOT"),
    ("", ""),
]


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_normalize_python(raw, expected):
    assert normalize(raw) == expected


def test_normalize_handles_none():
    assert normalize(None) == ""


@pytest.mark.spark
def test_spark_twin_matches_python(spark):
    from pyspark.sql import functions as F

    from panelcast.entity_resolution.normalize import normalize_col

    df = spark.createDataFrame([(raw,) for raw, _ in CASES], "raw string")
    got = {r["raw"]: r["norm"] for r in df.select("raw", normalize_col(F.col("raw")).alias("norm")).collect()}
    for raw, _ in CASES:
        assert got[raw] == normalize(raw), raw
