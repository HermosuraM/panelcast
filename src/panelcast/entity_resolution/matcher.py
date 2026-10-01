"""Two-stage entity resolution: normalized card descriptor -> brand.

Stage 1 (blocking): TF-IDF character n-grams retrieve the top-k candidate aliases per descriptor, so the
expensive pairwise scoring never runs on the full cross product.
Stage 2 (scoring): fuzzy string features + merchant-category-code (MCC) compatibility -> one score.

Decisions are precision-first, because a false positive silently corrupts a ticker's spend signal while a
false negative only shrinks the sample:
    rule      merchant-of-record rules (marketplaces, ride-hailing) win over string similarity
    exact     descriptor equals an alias or starts with one, and the MCC is plausible for the brand
    fuzzy     combined score >= auto_accept, MCC plausible, and no close runner-up from another ticker
    review    gray zone (or high-spend unmatched) -> LLM adjudication if enabled, else left unmatched
    none      everything else
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

MCC_PENALTY = 0.85
SHORT_ALIAS = 4
AMBIGUITY_MARGIN = 0.03

# Merchant-of-record rules: (token that must appear, brand or None for "not in the universe")
MARKETPLACE_RULES: list[tuple[str, str | None]] = [
    ("INSTACART", None),
    ("INSTACAR", None),
    ("GRUBHUB", None),
    ("DOORDASH", "DOORDASH"),
    ("DASHPASS", "DOORDASH"),
]
FIRST_TOKEN_RULES = {"UBER": "UBER", "LYFT": None}


@dataclass
class Match:
    brand: str | None
    score: float
    method: str
    decision: str  # MATCH | REVIEW | NONE
    mcc_ok: bool
    runner_up: str | None = None
    runner_up_score: float = 0.0


class DescriptorMatcher:
    def __init__(
        self,
        aliases: pd.DataFrame,
        brands: pd.DataFrame,
        auto_accept: float = 0.90,
        review_floor: float = 0.72,
        top_k: int = 5,
    ):
        self.alias_norm = aliases["alias_norm"].tolist()
        self.alias_brand = aliases["brand"].tolist()
        self.brand_ticker = dict(zip(brands["brand"], brands["ticker"], strict=True))
        self.brand_mcc = {
            b: {int(x) for x in str(m).split(",") if x}
            for b, m in zip(brands["brand"], brands["mcc_list"], strict=True)
        }
        self.auto_accept, self.review_floor, self.top_k = auto_accept, review_floor, top_k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.alias_matrix = self.vectorizer.fit_transform(self.alias_norm)
        self.exact: dict[str, str] = {}
        self.compact: dict[str, str] = {}  # spacing variants: YARDHOUSE == YARD HOUSE, PIZZAHUT == PIZZA HUT
        for a, b in zip(self.alias_norm, self.alias_brand, strict=True):
            self.exact.setdefault(a, b)
            self.compact.setdefault(a.replace(" ", ""), b)

    # -- features -------------------------------------------------------------------------------
    def _pair_score(self, desc: str, alias_i: int, cosine: float) -> float:
        alias = self.alias_norm[alias_i]
        tsr = max(fuzz.token_set_ratio(desc, alias), fuzz.ratio(desc.replace(" ", ""), alias.replace(" ", ""))) / 100
        tsort = fuzz.token_sort_ratio(desc, alias) / 100
        jw = JaroWinkler.similarity(desc.split()[0], alias.split()[0])
        score = 0.40 * tsr + 0.20 * tsort + 0.20 * cosine + 0.20 * jw
        if len(alias) <= SHORT_ALIAS and alias not in desc.split():
            score = min(score, 0.5)  # "ULTA" must not match "ULTRA CLEAN", "UBER" not "UBERCUTS"
        return float(score)

    def _rule(self, desc: str) -> Match | None:
        tokens = desc.split()
        if not tokens:
            return Match(None, 0.0, "rule:empty", "NONE", False)
        for token, brand in MARKETPLACE_RULES:
            if token in tokens:
                return Match(brand, 1.0, "rule:merchant_of_record", "MATCH" if brand else "NONE", True)
        if tokens[0] in FIRST_TOKEN_RULES:
            brand = FIRST_TOKEN_RULES[tokens[0]]
            return Match(brand, 1.0, "rule:merchant_of_record", "MATCH" if brand else "NONE", True)
        return None

    def _exact(self, desc: str, mcc: int) -> Match | None:
        brand = self.exact.get(desc) or self.compact.get(desc.replace(" ", "")) or self.compact.get(desc.split()[0])
        if brand is None:
            # longest alias that the descriptor starts with, on a token boundary
            hits = [
                (len(a), b) for a, b in zip(self.alias_norm, self.alias_brand, strict=True) if desc.startswith(a + " ")
            ]
            if not hits:
                return None
            brand = max(hits)[1]
        if mcc in self.brand_mcc[brand]:
            return Match(brand, 1.0 if desc in self.exact else 0.99, "exact", "MATCH", True)
        return None  # e.g. TARGET OPTICAL (MCC 8043): fall through to fuzzy scoring, which penalizes the MCC

    # -- public API -----------------------------------------------------------------------------
    def match_many(self, descs: list[str], mccs: list[int]) -> list[Match]:
        results: list[Match | None] = [None] * len(descs)
        pending = []
        for i, (d, m) in enumerate(zip(descs, mccs, strict=True)):
            results[i] = self._rule(d) or self._exact(d, int(m))
            if results[i] is None:
                pending.append(i)
        if pending:
            sims = (self.vectorizer.transform([descs[i] for i in pending]) @ self.alias_matrix.T).toarray()
            for row, i in enumerate(pending):
                results[i] = self._fuzzy(descs[i], int(mccs[i]), sims[row])
        return results  # type: ignore[return-value]

    def _fuzzy(self, desc: str, mcc: int, sims: np.ndarray) -> Match:
        top = np.argsort(-sims)[: self.top_k]
        best: dict[str, tuple[float, bool]] = {}
        for alias_i in top:
            brand = self.alias_brand[alias_i]
            score = self._pair_score(desc, int(alias_i), float(sims[alias_i]))
            mcc_ok = mcc in self.brand_mcc[brand]
            if not mcc_ok:
                score *= MCC_PENALTY
            if brand not in best or score > best[brand][0]:
                best[brand] = (score, mcc_ok)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        brand, (score, mcc_ok) = ranked[0]
        runner, runner_score = (ranked[1][0], ranked[1][1][0]) if len(ranked) > 1 else (None, 0.0)
        ambiguous = (
            runner is not None
            and self.brand_ticker[runner] != self.brand_ticker[brand]
            and score - runner_score < AMBIGUITY_MARGIN
        )
        if score >= self.auto_accept and mcc_ok and not ambiguous:
            decision = "MATCH"
        elif score >= self.review_floor:
            decision = "REVIEW"
        else:
            decision = "NONE"
        return Match(brand, round(score, 4), "fuzzy", decision, mcc_ok, runner, round(runner_score, 4))

    def match_frame(self, pdf: pd.DataFrame) -> pd.DataFrame:
        matches = self.match_many(pdf["descriptor_norm"].fillna("").tolist(), pdf["mcc"].fillna(-1).tolist())
        out = pd.DataFrame([asdict(m) for m in matches])
        out.insert(0, "descriptor_norm", pdf["descriptor_norm"].to_numpy())
        out.insert(1, "mcc", pdf["mcc"].to_numpy())
        out["ticker"] = out["brand"].map(self.brand_ticker)
        return out
