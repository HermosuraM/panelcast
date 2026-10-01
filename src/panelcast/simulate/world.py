"""Driver-side construction of the synthetic world: merchants, daily per-capita spend, demographics, members."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from panelcast.config import FAR_PAST, Settings
from panelcast.simulate.descriptors import CITIES, LOCAL_WORDS_A, LOCAL_WORDS_B
from panelcast.simulate.revenue_curve import daily_revenue, visible_ratio

AGE, INCOME, REGION = "age", "income", "region"


@dataclass
class Merchant:
    idx: int
    kind: str  # brand | distractor | chain | local
    name: str  # brand id (kind=brand) or printed merchant name
    ticker: str | None  # owning ticker for brands
    valid_from: dt.date  # date the brand starts counting toward `ticker`
    mcc: int
    ticket: float
    sigma: float
    profile: str
    online_share: float = 0.0
    templates: list[str] = field(default_factory=list)
    online: list[str] = field(default_factory=list)
    stores: int = 1
    region: str | None = None  # locals are regional


@dataclass
class World:
    start: dt.date
    end: dt.date
    dates: pd.DatetimeIndex
    merchants: list[Merchant]
    spend: np.ndarray  # [days, merchants] population per-capita $ per day
    mult: np.ndarray  # [cells, merchants] demographic spend multipliers (pop-weighted mean 1)
    cell_labels: list[tuple[str, str, str]]
    cell_pop_share: np.ndarray
    daily_truth: pd.DataFrame  # date, ticker, revenue, ratio, visible_spend, per_capita
    regions: list[str]

    def merchant_index(self, kind: str) -> list[int]:
        return [m.idx for m in self.merchants if m.kind == kind]


def cell_grid(sim: dict) -> tuple[list[tuple[str, str, str]], np.ndarray, list[np.ndarray]]:
    demo = sim["demographics"]
    labels, shares = [], []
    margins = [np.asarray(demo[k]["population"], float) for k in (AGE, INCOME, REGION)]
    for a, age in enumerate(demo[AGE]["labels"]):
        for i, inc in enumerate(demo[INCOME]["labels"]):
            for r, reg in enumerate(demo[REGION]["labels"]):
                labels.append((age, inc, reg))
                shares.append(margins[0][a] * margins[1][i] * margins[2][r])
    return labels, np.asarray(shares), margins


def profile_multipliers(profile: dict, pop_share: np.ndarray) -> np.ndarray:
    a, i, r = (np.asarray(profile[k], float) for k in (AGE, INCOME, REGION))
    raw = np.einsum("a,i,r->air", a, i, r).reshape(-1)
    return raw / float(pop_share @ raw)


def _contiguous_tail(periods: pd.DataFrame) -> pd.DataFrame:
    """Longest run of back-to-back fiscal quarters ending at the latest period."""
    p = periods.sort_values("period_start").reset_index(drop=True)
    keep = len(p) - 1
    while keep > 0 and p.loc[keep, "period_start"] == p.loc[keep - 1, "period_end"] + dt.timedelta(days=1):
        keep -= 1
    return p.iloc[keep:]


def build_world(settings: Settings, revenue: pd.DataFrame) -> World:
    sim = settings.sim
    start, end = settings.sim_date("start_date"), settings.sim_date("end_date")
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(settings.seed)
    cell_labels, pop_share, _ = cell_grid(sim)
    population = float(sim["population"])

    revenue = revenue.copy()
    for col in ("period_start", "period_end"):
        revenue[col] = pd.to_datetime(revenue[col]).dt.date

    merchants: list[Merchant] = []
    spend_cols: list[np.ndarray] = []
    mult_cols: list[np.ndarray] = []
    truth_frames = []
    universe = {b.brand: b for b in settings.brands}

    for ticker in settings.tickers:
        tk = sim["tickers"][ticker]
        periods = revenue[(revenue["ticker"] == ticker) & (revenue["period_end"] >= start - dt.timedelta(days=400))]
        periods = _contiguous_tail(periods[periods["period_start"] <= end])
        daily = daily_revenue(periods, sim["dow_profiles"][tk["dow"]], tk["dow"])
        ratio = visible_ratio(daily["date"], periods, tk["ratio"], tk["drift"], tk["wobble"], rng)
        daily["ratio"] = ratio
        daily["visible_spend"] = daily["revenue"] * ratio
        daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        series = pd.Series(daily["visible_spend"].to_numpy(), index=pd.to_datetime(daily["date"]))
        visible = series.reindex(dates).fillna(0.0).to_numpy() / population
        truth_frames.append(
            pd.DataFrame(
                {
                    "date": daily["date"],
                    "ticker": ticker,
                    "revenue": daily["revenue"],
                    "ratio": daily["ratio"],
                    "visible_spend": daily["visible_spend"],
                    "per_capita": daily["visible_spend"] / population,
                }
            )
        )

        mix = {b: float(s) for b, s in tk["mix"].items()}
        mult = profile_multipliers(sim["profiles"][tk["profile"]], pop_share)
        valid_from = {b: universe[b].valid_from for b in mix}
        # Shares of brands owned on each day, renormalized; acquired brands keep trading before the close
        # (as stand-alone companies, i.e. outside the universe) at their post-close share of spend.
        owned = np.stack([(dates.date >= valid_from[b]).astype(float) for b in mix], axis=1)
        shares = np.asarray(list(mix.values()))
        norm = (owned * shares).sum(axis=1, keepdims=True)
        brand_spend = visible[:, None] * np.where(owned > 0, shares / norm, shares)
        for j, brand in enumerate(mix):
            spec = sim["brands"][brand]
            merchants.append(
                Merchant(
                    idx=len(merchants),
                    kind="brand",
                    name=brand,
                    ticker=ticker,
                    valid_from=valid_from[brand],
                    mcc=int(spec["mcc"]),
                    ticket=float(tk["ticket"]),
                    sigma=float(tk["sigma"]),
                    profile=tk["profile"],
                    online_share=float(spec["online_share"]),
                    templates=list(spec["templates"]),
                    online=list(spec["online"]),
                    stores=int(spec["stores"]),
                )
            )
            spend_cols.append(brand_spend[:, j])
            mult_cols.append(mult)

    years = (dates - dates[0]).days.to_numpy() / 365.25
    flat = profile_multipliers(sim["profiles"]["local"], pop_share)
    for d in sim["distractors"]:
        merchants.append(
            Merchant(
                idx=len(merchants),
                kind="distractor",
                name=d["name"],
                ticker=None,
                valid_from=FAR_PAST,
                mcc=int(d["mcc"]),
                ticket=float(d["ticket"]),
                sigma=0.6,
                profile="local",
            )
        )
        spend_cols.append(float(d["daily_per_capita"]) * np.exp(0.02 * years))
        mult_cols.append(flat)

    other = sim["other_merchants"]
    for name, mcc in other["chains"]:
        merchants.append(
            Merchant(
                idx=len(merchants),
                kind="chain",
                name=name,
                ticker=None,
                valid_from=FAR_PAST,
                mcc=int(mcc),
                ticket=float(rng.uniform(12, 70)),
                sigma=0.7,
                profile="local",
            )
        )
        spend_cols.append(float(other["chain_daily_per_capita"]) * np.exp(0.03 * years))
        mult_cols.append(flat)

    regions = list(sim["demographics"][REGION]["labels"])
    used = set()
    for _ in range(int(other["local_count"])):
        while True:
            name = f"{rng.choice(LOCAL_WORDS_A)} {rng.choice(LOCAL_WORDS_B)}"
            region = str(rng.choice(regions))
            if (name, region) not in used:
                used.add((name, region))
                break
        processor = str(rng.choice(other["processors"]))
        merchants.append(
            Merchant(
                idx=len(merchants),
                kind="local",
                name=f"{processor}{name}",
                ticker=None,
                valid_from=FAR_PAST,
                mcc=int(rng.choice(other["local_mcc"])),
                ticket=float(rng.uniform(8, 60)),
                sigma=0.6,
                profile="local",
                region=region,
            )
        )
        spend_cols.append(float(other["local_daily_per_capita"]) * np.ones(len(dates)))
        mult_cols.append(flat)

    return World(
        start=start,
        end=end,
        dates=dates,
        merchants=merchants,
        spend=np.stack(spend_cols, axis=1),
        mult=np.stack(mult_cols, axis=1),
        cell_labels=cell_labels,
        cell_pop_share=pop_share,
        daily_truth=pd.concat(truth_frames, ignore_index=True),
        regions=regions,
    )


def build_members(settings: Settings, world: World) -> pd.DataFrame:
    """Panel members with true enrollment dates, demographics and a 'data_end' that honors silent departures."""
    sim = settings.sim
    rng = np.random.default_rng(settings.seed + 1)
    n_total = int(sim["panel"]["n_panelists"])
    tenure_days = float(sim["panel"]["mean_tenure_years"]) * 365.25
    silent = {a["source"]: dt.date.fromisoformat(str(a["start"])) for a in sim["anomalies"]["silent_departures"]}
    cities_by_region = {r: [i for i, c in enumerate(CITIES) if c[2] == r] for r in world.regions}
    rows = []
    member_idx = 0
    for src in sim["sources"]:
        n = int(round(n_total * float(src["share"])))
        on0, on1 = (dt.date.fromisoformat(str(x)) for x in src["onboard"])
        trickle = rng.random(n) < float(src["trickle"])
        wave_days = (on1 - on0).days
        late_days = max((world.end - on1).days, 1)
        offsets = np.where(trickle, wave_days + rng.integers(0, late_days, n), rng.integers(0, wave_days + 1, n))
        ages = rng.choice(3, n, p=src["age"])
        incomes = rng.choice(3, n, p=src["income"])
        regs = rng.choice(4, n, p=src["region"])
        tenures = rng.exponential(tenure_days, n).astype(int) + 30
        for k in range(n):
            join = on0 + dt.timedelta(days=int(offsets[k]))
            leave = join + dt.timedelta(days=int(tenures[k]))
            leave = None if leave > world.end else leave
            data_end = min(leave or world.end, world.end)
            if src["id"] in silent:
                data_end = min(data_end, silent[src["id"]] - dt.timedelta(days=1))
            region = world.regions[regs[k]]
            rows.append(
                {
                    "member_idx": member_idx,
                    "user_id": f"U{member_idx:07d}",
                    "source": src["id"],
                    "format": src["format"],
                    "age_idx": int(ages[k]),
                    "income_idx": int(incomes[k]),
                    "region_idx": int(regs[k]),
                    "cell": int(ages[k] * 12 + incomes[k] * 4 + regs[k]),
                    "join_date": join,
                    "leave_date": leave,
                    "data_end": data_end,
                    "home_city": int(rng.choice(cities_by_region[region])),
                }
            )
            member_idx += 1
    return pd.DataFrame(rows)
