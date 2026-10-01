"""Distributed generation of the synthetic card panel (one Spark task per slice of panel members).

Writes:
  sim_truth.*                     ground truth (labels, true spend, injected anomalies) - evaluation only
  landing/transactions/...        what a data vendor would deliver: gzip CSVs per source and delivery
  landing/panel/panel_members.csv vendor membership file (stale for silently departed sources!)
  landing/reference/population_margins.csv   "census" margins used for raking
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from panelcast.config import Settings
from panelcast.simulate.descriptors import CITIES, apply_source_format, fill, make_ref
from panelcast.simulate.world import World, build_members, build_world
from panelcast.store import TableStore

log = logging.getLogger(__name__)

TXN_SCHEMA = """
    txn_id string, user_id string, txn_date string, post_date string, amount string, currency string,
    merchant_descriptor string, mcc string, source string, delivery string,
    merchant_idx int, true_brand string, true_kind string, true_ticker string, true_amount double,
    true_txn_date date, anomaly string
"""


def _windows(items: list[dict], key: str = "source") -> dict[str, list[tuple[int, int, dict]]]:
    """source -> [(first_ordinal, last_ordinal_inclusive, spec)]"""
    out: dict[str, list] = {}
    for a in items:
        start = dt.date.fromisoformat(str(a["start"])).toordinal()
        days = int(a.get("days", 1))
        out.setdefault(a[key], []).append((start, start + days - 1, a))
    return out


def _in(windows: list, ordinals: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(ordinals), bool)
    for lo, hi, _ in windows:
        mask |= (ordinals >= lo) & (ordinals <= hi)
    return mask


def make_generator(world: World, sim: dict, seed: int):
    """Build the mapInPandas function. Everything it closes over is plain numpy/python (pickled to workers)."""
    spend = world.spend.astype(np.float64)
    mult = world.mult.astype(np.float64)
    start_ord = world.start.toordinal()
    merchants = world.merchants
    tickets = np.array([m.ticket for m in merchants])
    sigmas = np.array([m.sigma for m in merchants])
    fixed_cols = np.array([m.idx for m in merchants if m.kind != "local"])
    locals_by_region = {
        r: np.array([m.idx for m in merchants if m.kind == "local" and m.region == r]) for r in world.regions
    }
    panel = sim["panel"]
    activity, shape = float(panel["activity_multiplier"]), float(panel["heterogeneity_shape"])
    lag_p = np.asarray(sim["posting_lag_probs"], float)
    refund_rate = float(sim["refund_rate"])
    backfill_until = dt.date.fromisoformat(str(sim["backfill_until"])).toordinal()
    an = sim["anomalies"]
    outages = _windows(an["outages"])
    dupes = _windows(an["duplicate_deliveries"])
    units = _windows(an["unit_errors"])
    changes = {
        (c["source"], c["brand"]): (dt.date.fromisoformat(str(c["start"])).toordinal(), c["template"])
        for c in an["descriptor_changes"]
    }
    malformed_rate = float(an["malformed_rate"])
    restaurants = list(sim["delivery_restaurants"])
    city_by_region = {r: [i for i, c in enumerate(CITIES) if c[2] == r] for r in world.regions}

    def member_rows(m) -> dict[str, list] | None:
        rng = np.random.default_rng([seed, int(m.member_idx)])
        d0 = m.join_date.toordinal() - start_ord
        d1 = m.data_end.toordinal() - start_ord + 1
        if d1 <= d0:
            return None
        region = world.regions[m.region_idx]
        favs = rng.choice(locals_by_region[region], size=min(8, len(locals_by_region[region])), replace=False)
        cols = np.concatenate([fixed_cols, favs])
        loyalty = rng.gamma(shape, 1.0 / shape, len(cols))
        lam = spend[d0:d1, cols] * (mult[m.cell, cols] * loyalty * activity / tickets[cols])
        counts = rng.poisson(lam)
        di, mj = np.nonzero(counts)
        reps = counts[di, mj]
        day = np.repeat(di, reps) + d0
        merch = cols[np.repeat(mj, reps)]
        n = len(day)
        if n == 0:
            return None
        txn_ord = day + start_ord
        sig = sigmas[merch]
        amount = tickets[merch] * np.exp(sig * rng.standard_normal(n) - 0.5 * sig**2)
        amount = np.where(rng.random(n) < refund_rate, -amount, amount).round(2)
        post_ord = txn_ord + rng.choice(len(lag_p), n, p=lag_p)

        src = m.source
        keep = ~_in(outages.get(src, []), txn_ord)
        unit_err = _in(units.get(src, []), txn_ord)
        dup = _in(dupes.get(src, []), txn_ord)
        malformed = rng.random(n) < malformed_rate
        bad_field = rng.integers(0, 3, n)
        bad_pick = rng.integers(0, 3, n)
        refs = make_ref(rng, n)
        u = rng.random(n)  # format / layout choices
        u_city = rng.random(n)
        alt_city = rng.choice(city_by_region[region], n)
        u_store = rng.random(n)
        store_slot = rng.choice(3, n, p=[0.6, 0.3, 0.1])
        u_online = rng.random(n)
        u_tmpl = rng.random(n)
        rest_idx = rng.integers(0, len(restaurants), n)
        any_store = rng.integers(1, 1_000_000, n)
        last4 = int(rng.integers(0, 10000))
        home_stores: dict[int, np.ndarray] = {}

        rows: dict[str, list] = {
            k: []
            for k in (
                "txn_id",
                "txn_date",
                "post_date",
                "amount",
                "descriptor",
                "mcc",
                "delivery",
                "merchant_idx",
                "true_brand",
                "true_kind",
                "true_ticker",
                "true_amount",
                "true_txn_date",
                "anomaly",
            )
        }
        for k in np.nonzero(keep)[0]:
            mer = merchants[merch[k]]
            t_ord = int(txn_ord[k])
            city, st, _ = CITIES[m.home_city if u_city[k] < 0.9 else int(alt_city[k])]
            anomaly = None
            mcc = mer.mcc
            if mer.kind == "brand":
                if mer.idx not in home_stores:
                    home_stores[mer.idx] = rng.integers(1, mer.stores + 1, 3)
                store = (
                    int(home_stores[mer.idx][store_slot[k]]) if u_store[k] < 0.9 else 1 + int(any_store[k]) % mer.stores
                )
                online = u_online[k] < mer.online_share
                choices = mer.online if online else mer.templates
                template = choices[int(u_tmpl[k] * len(choices))]
                change = changes.get((src, mer.name))
                if change and t_ord >= change[0]:
                    template, anomaly = change[1], "descriptor_change"
                desc = fill(template, store, city, st, refs[k], restaurants[rest_idx[k]])
                if "GAS" in template:
                    mcc = 5542
                label = mer.ticker if t_ord >= mer.valid_from.toordinal() else "NONE"
                true_brand = mer.name
            else:
                store = 1 + int(any_store[k]) % 3000
                if mer.kind == "chain":
                    desc = f"{mer.name} #{store} {city}" if u[k] < 0.5 else f"{mer.name} {store}"
                elif mer.kind == "local":
                    desc = f"{mer.name} {city} {st}" if u[k] < 0.5 else mer.name
                else:
                    desc = mer.name if u[k] < 0.5 else f"{mer.name} #{store}"
                label, true_brand = "NONE", "NONE"
            desc = apply_source_format(desc, m.format, float(u[k]), last4)

            amt = float(amount[k])
            amount_str = f"{amt:.2f}"
            if unit_err[k]:
                amount_str, anomaly = f"{amt * 100:.0f}", "unit_error"
            txn_date_str = dt.date.fromordinal(t_ord).isoformat()
            if malformed[k]:
                field = int(bad_field[k])
                if field == 0:
                    amount_str = ["N/A", "12.3O", "1,234.5.6"][bad_pick[k]]
                elif field == 1:
                    txn_date_str = ["2024-13-45", "00/00/0000", "yesterday"][bad_pick[k]]
                else:
                    desc = ""
                anomaly = f"malformed_{['amount', 'date', 'descriptor'][field]}"

            p_ord = int(post_ord[k])
            delivery = (
                f"backfill-{dt.date.fromordinal(p_ord):%Y-%m}"
                if p_ord <= backfill_until
                else f"daily-{dt.date.fromordinal(p_ord).isoformat()}"
            )
            txn_id = f"{src[-1]}{m.member_idx:07d}{t_ord:07d}{merch[k]:04d}{k:05d}"
            copies = [(delivery, anomaly)]
            if dup[k]:
                last_dup = max(hi for lo, hi, _ in dupes[src] if lo <= t_ord <= hi)
                copies.append((f"redelivery-{dt.date.fromordinal(last_dup + 3).isoformat()}", "duplicate"))
            for deliv, anom in copies:
                rows["txn_id"].append(txn_id)
                rows["txn_date"].append(txn_date_str)
                rows["post_date"].append(dt.date.fromordinal(p_ord).isoformat())
                rows["amount"].append(amount_str)
                rows["descriptor"].append(desc)
                rows["mcc"].append(str(mcc))
                rows["delivery"].append(deliv)
                rows["merchant_idx"].append(int(merch[k]))
                rows["true_brand"].append(true_brand)
                rows["true_kind"].append(mer.kind)
                rows["true_ticker"].append(label)
                rows["true_amount"].append(amt)
                rows["true_txn_date"].append(dt.date.fromordinal(t_ord))
                rows["anomaly"].append(anom)
        n_out = len(rows["txn_id"])
        rows["user_id"] = [m.user_id] * n_out
        rows["source"] = [src] * n_out
        return rows

    def generate(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in batches:
            for col in ("join_date", "data_end"):
                pdf[col] = pd.to_datetime(pdf[col]).dt.date
            buffer: list[pd.DataFrame] = []
            size = 0
            for m in pdf.itertuples(index=False):
                rows = member_rows(m)
                if not rows:
                    continue
                frame = pd.DataFrame(
                    {
                        "txn_id": rows["txn_id"],
                        "user_id": rows["user_id"],
                        "txn_date": rows["txn_date"],
                        "post_date": rows["post_date"],
                        "amount": rows["amount"],
                        "currency": "USD",
                        "merchant_descriptor": rows["descriptor"],
                        "mcc": rows["mcc"],
                        "source": rows["source"],
                        "delivery": rows["delivery"],
                        "merchant_idx": np.asarray(rows["merchant_idx"], np.int32),
                        "true_brand": rows["true_brand"],
                        "true_kind": rows["true_kind"],
                        "true_ticker": rows["true_ticker"],
                        "true_amount": rows["true_amount"],
                        "true_txn_date": rows["true_txn_date"],
                        "anomaly": rows["anomaly"],
                    }
                )
                buffer.append(frame)
                size += len(frame)
                if size > 200_000:
                    yield pd.concat(buffer, ignore_index=True)
                    buffer, size = [], 0
            if buffer:
                yield pd.concat(buffer, ignore_index=True)

    return generate


def anomaly_log(sim: dict) -> pd.DataFrame:
    rows = []
    an = sim["anomalies"]
    for kind in ("outages", "duplicate_deliveries", "unit_errors"):
        for a in an[kind]:
            start = dt.date.fromisoformat(str(a["start"]))
            rows.append(
                {
                    "type": kind.rstrip("s").replace("_deliverie", "_delivery"),
                    "source": a["source"],
                    "brand": None,
                    "start": start,
                    "end": start + dt.timedelta(days=int(a["days"]) - 1),
                }
            )
    end = dt.date.fromisoformat(str(sim["end_date"]))
    for a in an["silent_departures"]:
        rows.append(
            {
                "type": "silent_departure",
                "source": a["source"],
                "brand": None,
                "start": dt.date.fromisoformat(str(a["start"])),
                "end": end,
            }
        )
    for a in an["descriptor_changes"]:
        rows.append(
            {
                "type": "descriptor_change",
                "source": a["source"],
                "brand": a["brand"],
                "start": dt.date.fromisoformat(str(a["start"])),
                "end": end,
            }
        )
    return pd.DataFrame(rows)


def run_simulation(settings: Settings, store: TableStore) -> dict:
    spark = store.spark
    sim = settings.sim
    revenue = pd.read_csv(settings.path(settings["reference"]["edgar_revenue"]))
    world = build_world(settings, revenue)
    members = build_members(settings, world)
    log.info("world: %d merchants, %d days; panel: %d members", len(world.merchants), len(world.dates), len(members))

    store.reset_landing()
    store.reset_layers(["sim_truth", "bronze", "silver", "gold", "eval"])

    gen = make_generator(world, sim, settings.seed)
    member_sdf = spark.createDataFrame(
        members[
            ["member_idx", "user_id", "source", "format", "region_idx", "cell", "join_date", "data_end", "home_city"]
        ]
    )
    n_parts = max(8, min(96, len(members) // 100))
    txns = member_sdf.repartition(n_parts, "member_idx").mapInPandas(gen, TXN_SCHEMA)
    store.write(txns, "sim_truth", "transactions")
    truth = store.read("sim_truth", "transactions")

    (
        truth.select(
            "txn_id",
            "user_id",
            "txn_date",
            "post_date",
            "amount",
            "currency",
            "merchant_descriptor",
            "mcc",
            "source",
            "delivery",
        )
        .repartition("source", "delivery")
        .write.mode("overwrite")
        .partitionBy("source", "delivery")
        .option("header", True)
        .option("compression", "gzip")
        .csv(store.landing("transactions"))
    )

    labels = truth.groupBy("merchant_descriptor", "mcc", "true_brand", "true_kind").agg(
        F.count("*").alias("n_txns"), F.sum(F.abs("true_amount")).alias("abs_spend")
    )
    store.write(labels, "sim_truth", "descriptor_labels")

    # Vendor membership file: lists everyone, including members of silently departed sources.
    vendor = members.assign(
        age_bucket=lambda d: d["age_idx"].map(dict(enumerate(sim["demographics"]["age"]["labels"]))),
        income_bucket=lambda d: d["income_idx"].map(dict(enumerate(sim["demographics"]["income"]["labels"]))),
        region=lambda d: d["region_idx"].map(dict(enumerate(world.regions))),
    )[["user_id", "source", "age_bucket", "income_bucket", "region", "join_date", "leave_date"]]
    _write_landing_csv(store, vendor, "panel", "panel_members.csv")

    margins = pd.DataFrame(
        [
            {"dimension": dim, "bucket": label, "share": share}
            for dim in ("age", "income", "region")
            for label, share in zip(
                sim["demographics"][dim]["labels"], sim["demographics"][dim]["population"], strict=True
            )
        ]
    )
    _write_landing_csv(store, margins, "reference", "population_margins.csv")

    catalog = pd.DataFrame(
        [
            {
                "merchant_idx": m.idx,
                "kind": m.kind,
                "name": m.name,
                "ticker": m.ticker,
                "valid_from": m.valid_from,
                "mcc": m.mcc,
                "ticket": m.ticket,
            }
            for m in world.merchants
        ]
    )
    store.write_pandas(catalog, "sim_truth", "merchants")
    store.write_pandas(members.drop(columns=["format"]), "sim_truth", "members")
    store.write_pandas(world.daily_truth, "sim_truth", "daily_truth")
    store.write_pandas(anomaly_log(sim), "sim_truth", "anomaly_log")

    stats = truth.agg(F.count("*").alias("rows"), F.countDistinct("merchant_descriptor").alias("descriptors")).first()
    log.info("simulated %s transaction rows, %s distinct descriptors", stats["rows"], stats["descriptors"])
    return {"rows": int(stats["rows"]), "descriptors": int(stats["descriptors"]), "members": len(members)}


def _write_landing_csv(store: TableStore, pdf: pd.DataFrame, folder: str, filename: str) -> None:
    path = Path(store.landing(folder))
    path.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(path / filename, index=False)
