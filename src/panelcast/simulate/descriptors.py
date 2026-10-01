"""Messy card-statement merchant descriptors, the way banks and processors actually print them."""

from __future__ import annotations

import numpy as np

# (city, state, region) - the synthetic panel's geography.
CITIES = [
    ("NEW YORK", "NY", "NE"),
    ("BOSTON", "MA", "NE"),
    ("PHILADELPHIA", "PA", "NE"),
    ("PITTSBURGH", "PA", "NE"),
    ("NEWARK", "NJ", "NE"),
    ("HARTFORD", "CT", "NE"),
    ("PROVIDENCE", "RI", "NE"),
    ("BUFFALO", "NY", "NE"),
    ("STAMFORD", "CT", "NE"),
    ("PORTLAND", "ME", "NE"),
    ("BURLINGTON", "VT", "NE"),
    ("ALBANY", "NY", "NE"),
    ("CHICAGO", "IL", "MW"),
    ("DETROIT", "MI", "MW"),
    ("COLUMBUS", "OH", "MW"),
    ("INDIANAPOLIS", "IN", "MW"),
    ("MILWAUKEE", "WI", "MW"),
    ("MINNEAPOLIS", "MN", "MW"),
    ("KANSAS CITY", "MO", "MW"),
    ("ST LOUIS", "MO", "MW"),
    ("OMAHA", "NE", "MW"),
    ("CLEVELAND", "OH", "MW"),
    ("DES MOINES", "IA", "MW"),
    ("MADISON", "WI", "MW"),
    ("HOUSTON", "TX", "S"),
    ("DALLAS", "TX", "S"),
    ("AUSTIN", "TX", "S"),
    ("SAN ANTONIO", "TX", "S"),
    ("ATLANTA", "GA", "S"),
    ("MIAMI", "FL", "S"),
    ("ORLANDO", "FL", "S"),
    ("TAMPA", "FL", "S"),
    ("CHARLOTTE", "NC", "S"),
    ("RALEIGH", "NC", "S"),
    ("NASHVILLE", "TN", "S"),
    ("MEMPHIS", "TN", "S"),
    ("NEW ORLEANS", "LA", "S"),
    ("RICHMOND", "VA", "S"),
    ("BIRMINGHAM", "AL", "S"),
    ("LOUISVILLE", "KY", "S"),
    ("OKLAHOMA CITY", "OK", "S"),
    ("RICHARDSON", "TX", "S"),
    ("JACKSONVILLE", "FL", "S"),
    ("LOS ANGELES", "CA", "W"),
    ("SAN DIEGO", "CA", "W"),
    ("SAN FRANCISCO", "CA", "W"),
    ("SAN JOSE", "CA", "W"),
    ("SACRAMENTO", "CA", "W"),
    ("SEATTLE", "WA", "W"),
    ("PORTLAND", "OR", "W"),
    ("DENVER", "CO", "W"),
    ("PHOENIX", "AZ", "W"),
    ("LAS VEGAS", "NV", "W"),
    ("SALT LAKE CITY", "UT", "W"),
    ("ALBUQUERQUE", "NM", "W"),
    ("BOISE", "ID", "W"),
    ("SPOKANE", "WA", "W"),
]

LOCAL_WORDS_A = [
    "SUNRISE",
    "GOLDEN",
    "MAIN ST",
    "BLUE MOON",
    "CORNER",
    "LUCKY",
    "GREEN LEAF",
    "OLD TOWN",
    "RIVERSIDE",
    "MAPLE",
    "HARBOR",
    "CEDAR",
    "SILVER",
    "URBAN",
    "PRAIRIE",
    "SUMMIT",
    "BAYSIDE",
    "HILLTOP",
    "COPPER",
    "LONE STAR",
]
LOCAL_WORDS_B = [
    "TACOS",
    "DELI",
    "CAFE",
    "BAKERY",
    "NAILS",
    "CLEANERS",
    "PIZZERIA",
    "BBQ",
    "AUTO CARE",
    "BARBER",
    "THAI KITCHEN",
    "PHO",
    "COFFEE CO",
    "BREWING",
    "MARKET",
    "FLOWERS",
    "BAGELS",
    "SUSHI",
    "TAQUERIA",
    "WINE & SPIRITS",
]
REF_ALPHABET = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"))
DIGITS = np.array(list("0123456789"))


def make_ref(rng: np.random.Generator, n: int, length: int = 9) -> list[str]:
    """Order/reference codes like 2K4L91XZ3 (always at least two digits, as real processor refs have)."""
    chars = rng.choice(REF_ALPHABET, size=(n, length))
    pos = rng.integers(0, length, size=(n, 2))
    chars[np.arange(n), pos[:, 0]] = rng.choice(DIGITS, n)
    chars[np.arange(n), pos[:, 1]] = rng.choice(DIGITS, n)
    return ["".join(row) for row in chars]


def fill(template: str, store: int, city: str, st: str, ref: str, rest: str) -> str:
    return template.format(store=store, store5=f"{store:05d}", city=city, st=st, ref=ref, rest=rest, abbr=city[:3])


def apply_source_format(desc: str, fmt: str, rng_u: float, last4: int) -> str:
    """Each data contributor formats statements differently - the noise ER has to see through."""
    desc = " ".join(desc.split())
    if fmt == "classic":
        return desc.upper()[:25].rstrip()
    if fmt == "pos_prefix":
        prefix = "POS PURCHASE " if rng_u < 0.7 else "POS DEBIT "
        return (prefix + desc.upper())[:40].rstrip()
    if fmt == "titlecase_short":
        return desc.title()[:22].rstrip()
    if fmt == "debit_purchase":
        return f"DEBIT CARD PURCHASE XXXXX{last4:04d} {desc.upper()}"[:48].rstrip()
    if fmt == "online_ref":
        out = desc.upper()
        return out if rng_u < 0.5 else f"{out} {last4:04d}{int(rng_u * 1e4):04d}"
    raise ValueError(f"unknown source format {fmt}")
