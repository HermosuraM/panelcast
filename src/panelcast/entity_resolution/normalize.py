"""Card-descriptor normalization, defined once and executed by both Python (`re`) and Spark (Java regex).

The same rules normalize brand aliases, so "WAL-MART #1234 DALLAS TX" and the alias "WAL-MART" meet
as "WAL MART". A unit test asserts the Python and Spark implementations agree.
"""

from __future__ import annotations

import re

from pyspark.sql import Column
from pyspark.sql import functions as F

# fmt: off
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]
# fmt: on

# Top US cities by population (a real pipeline would load a Census gazetteer). Multi-word names first.
GAZETTEER = sorted(
    {
        "NEW YORK",
        "LOS ANGELES",
        "CHICAGO",
        "HOUSTON",
        "PHOENIX",
        "PHILADELPHIA",
        "SAN ANTONIO",
        "SAN DIEGO",
        "DALLAS",
        "SAN JOSE",
        "AUSTIN",
        "JACKSONVILLE",
        "FORT WORTH",
        "COLUMBUS",
        "CHARLOTTE",
        "SAN FRANCISCO",
        "INDIANAPOLIS",
        "SEATTLE",
        "DENVER",
        "WASHINGTON",
        "BOSTON",
        "EL PASO",
        "NASHVILLE",
        "DETROIT",
        "OKLAHOMA CITY",
        "PORTLAND",
        "LAS VEGAS",
        "MEMPHIS",
        "LOUISVILLE",
        "BALTIMORE",
        "MILWAUKEE",
        "ALBUQUERQUE",
        "TUCSON",
        "FRESNO",
        "SACRAMENTO",
        "KANSAS CITY",
        "MESA",
        "ATLANTA",
        "OMAHA",
        "COLORADO SPRINGS",
        "RALEIGH",
        "MIAMI",
        "OAKLAND",
        "MINNEAPOLIS",
        "TULSA",
        "CLEVELAND",
        "WICHITA",
        "NEW ORLEANS",
        "TAMPA",
        "ORLANDO",
        "PITTSBURGH",
        "CINCINNATI",
        "ST LOUIS",
        "SAINT LOUIS",
        "NEWARK",
        "BUFFALO",
        "RICHMOND",
        "BIRMINGHAM",
        "SALT LAKE CITY",
        "BOISE",
        "SPOKANE",
        "DES MOINES",
        "MADISON",
        "HARTFORD",
        "PROVIDENCE",
        "STAMFORD",
        "BURLINGTON",
        "ALBANY",
        "RICHARDSON",
        "PLANO",
        "IRVING",
        "ARLINGTON",
        "LOS GATOS",
    },
    key=lambda c: (-len(c.split()), c),
)

PAYMENT_WRAPPERS = r"^(SQ ?\*|TST\* ?|SP ?\* ?|PAYPAL ?\*|PP ?\*|CLV ?\*)\s*"
BANK_BOILERPLATE = r"^(POS PURCHASE|POS DEBIT|DEBIT CARD PURCHASE|CHECKCARD|RECURRING PAYMENT)\s+"
STOPWORDS = ["THE", "STORE", "STORES", "INC", "LLC", "CORP", "CO", "AND"]

# (pattern, replacement) applied in order. Patterns use the subset of syntax shared by Python and Java.
RULES: list[tuple[str, str]] = [
    (r"\s+", " "),
    (r"\b7-ELEVEN\b", "SEVEN ELEVEN"),  # brands whose names contain digits
    (BANK_BOILERPLATE, ""),
    (r"X{3,}\d{2,4}", " "),  # masked card numbers
    (r"^DD ?\*\s*", "DOORDASH "),  # marketplace prefixes name the merchant of record
    (r"^IC ?\*\s*", "INSTACART "),
    (PAYMENT_WRAPPERS, ""),  # pass-through wallets keep the real merchant
    (r"\bWWW\.", ""),
    (r"\b[A-Z0-9.]*\.(COM|NET)(/\S*)?", "__DOMAIN__"),  # AMAZON.COM -> AMAZON, HELP.UBER.COM -> UBER
    (r"\*\S*", " "),  # order / reference codes glued with '*'
    (r"\b\S*\d\S*\b", " "),  # store numbers, refs, phone numbers
    (r"[']", ""),  # MCDONALD'S -> MCDONALDS
    (r"[^A-Z& ]", " "),  # other punctuation -> space
    (r"\b(" + "|".join(STOPWORDS) + r")\b", " "),
    (r"\s+", " "),
]


def _domain_stem(text: str) -> str:
    """Python twin of the Spark domain handling: AMAZON.COM*X -> AMAZON, HELP.UBER.COM -> UBER."""

    def repl(m: re.Match) -> str:
        host = m.group(0).split("/")[0]
        parts = [p for p in host.split(".") if p and p not in ("COM", "NET", "HELP", "WWW")]
        return f" {parts[-1]} " if parts else " "

    return re.sub(r"\b[A-Z0-9.]*\.(COM|NET)(/\S*)?", repl, text)


def _strip_location(text: str) -> str:
    tokens = text.split()
    if tokens and tokens[-1] in US_STATES and len(tokens) > 1:
        tokens = tokens[:-1]
    joined = " ".join(tokens)
    for city in GAZETTEER:
        if joined.endswith(" " + city):
            joined = joined[: -len(city) - 1]
            break
    return joined.strip()


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    s = text.upper().strip()
    for pattern, repl in RULES:
        if repl == "__DOMAIN__":
            s = _domain_stem(s)
            continue
        s = re.sub(pattern, repl, s).strip()
    return _strip_location(s)


def normalize_col(col: Column) -> Column:
    """Spark twin of `normalize` (pure JVM expressions - no Python UDF on the hot path)."""
    s = F.trim(F.upper(col))
    for pattern, repl in RULES:
        if repl == "__DOMAIN__":
            # keep the last host label before .COM/.NET, dropping HELP./WWW. style prefixes
            s = F.regexp_replace(s, r"\b(?:[A-Z0-9]+\.)*?(?:HELP\.|WWW\.)?([A-Z0-9]+)\.(?:COM|NET)(?:/\S*)?", " $1 ")
        else:
            s = F.trim(F.regexp_replace(s, pattern, repl))
    # trailing US state code, then a trailing gazetteer city
    states = "|".join(US_STATES)
    s = F.when(F.size(F.split(s, " ")) > 1, F.regexp_replace(s, rf" ({states})$", "")).otherwise(s)
    cities = "|".join(re.escape(c) for c in GAZETTEER)
    s = F.regexp_replace(s, rf" ({cities})$", "")
    return F.trim(s)
