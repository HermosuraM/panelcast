# Methodology

This document explains *why* each stage works the way it does. Code references are relative to `src/panelcast/`.

## 1. What is real, what is simulated

| Real (public data) | Simulated (because real card data costs six figures) |
|---|---|
| Quarterly revenue for 20 companies from SEC EDGAR XBRL, as first reported, on each company's exact fiscal calendar | Individual card transactions from 5 data contributors (~10-20M rows) |
| Filing dates (when the market learned each number) | Merchant descriptors, MCCs, store numbers, posting lags |
| Acquisition close dates (Habit -> YUM, Ruth's Chris and Chuy's -> DRI) | Panel members, demographics, churn, onboarding waves |
| Daily English-Wikipedia pageviews (the second alt-data asset) | 13 injected data problems with a ground-truth log |

The simulator is not a toy random walk: each company's daily *card-visible spend* is interpolated from its **real
reported revenue** and scaled by a slowly drifting "visibility ratio" (US consumer share, franchise system sales,
gig-economy take rates) that the panel cannot observe. So the panel tracks the real business, the nowcast is scored
against **real reported revenue**, and the irreducible gap between what a card panel sees and what a company
reports is part of the problem, as it is in production. Because the simulator also logs the truth (true spend,
true merchant of every descriptor, every injected anomaly), every stage can be *measured*, not just run.

## 2. Reference data: SEC EDGAR (`reference/edgar.py`)

* **Periods come from dates, never from `fy`/`fp` tags.** Those tags describe the filing a fact appeared in; a
  prior-year comparative inside this year's 10-Q carries this year's labels.
* **As-first-reported.** Each fiscal period keeps the earliest-filed value (what the market saw on earnings day)
  plus the latest value, so restatements are visible but never leak into training (look-ahead bias).
* **Concept drift.** ASC 606 moved most filers from `SalesRevenueNet` to `RevenueFromContractWithCustomer...` around
  2018. The primary concept is the one with the most coverage; others back-fill only when they agree within 0.5% on
  overlapping periods.
* **Q4 = annual - (Q1+Q2+Q3).** Q4 is rarely tagged on its own.
* **52/53-week calendars.** Retailers and restaurants report 12-, 13-, 14-, 16- and 17-week quarters (Costco's Q4
  is 16 or 17 weeks; Domino's quarters are 12/12/12/16 weeks). Nothing assumes calendar quarters; the fiscal year
  label is the year in which the fiscal year ends (shifted a week so years ending Jan 1-3 keep last year's label).

## 3. The simulated vendor feed (`simulate/`)

* **Daily spend.** Cumulative revenue at fiscal-period boundaries is interpolated with a monotone cubic (PCHIP);
  differencing gives smooth daily rates whose quarterly sums reproduce the reported revenue exactly. Day-of-week
  and holiday factors (Black Friday, Thanksgiving closures, Super Bowl pizza, Mother's Day dinners) reshape spend
  inside each quarter and are renormalized so totals still tie out.
* **Population and panel.** 36 demographic cells (age x income x region). Each brand has spend multipliers by
  cell (Dollar General skews low-income and Southern; Uber and DoorDash skew young). Five data sources join at
  different times with different demographic mixes, so the raw panel composition drifts - exactly the bias
  normalization and raking must remove.
* **Members.** Each member has a home city, a few "home stores" per brand, and gamma-distributed loyalty, so a few
  heavy customers drive volume (realistic over-dispersion).
* **Descriptors.** Brand templates (`STARBUCKS STORE 01234`, `AMZN MKTP US*2K4L91XZ3`, `DD *DOORDASH PANERA`) are
  wrapped by each source's format (`POS PURCHASE ...`, `DEBIT CARD PURCHASE XXXXX1234 ...`, 22-character
  truncation, title case). Look-alike merchants (`TARGET OPTICAL`, `LOWES FOODS`, `OFFICE DEPOT`, `ULTRA CLEAN CAR
  WASH`, `IC* COSTCO BY INSTACART`) and real chains outside the universe (Lyft shares Uber's MCC, BJ's shares
  Costco's) exist to test precision.
* **Injected problems** (logged to `sim_truth.anomaly_log`): source outages, a *silent* source dropout (data stops
  but the membership file still lists the members), duplicate re-deliveries, amounts delivered in cents, processor
  format changes that rewrite one brand's descriptors on one source, and malformed rows.

## 4. Ingestion and data quality (`pipeline/bronze.py`, `pipeline/silver.py`)

* **Bronze = exactly what arrived.** All columns are read as strings with Auto Loader (Databricks) or Spark's file
  stream source (local), `trigger(availableNow=True)`, and a checkpoint, so each file is ingested once and a re-run
  is a no-op. Lineage (`_metadata.file_path`, ingest time) is kept on every row.
* **Silver = typed and trusted.** Parsing uses ANSI-safe functions (`try_cast`, `try_to_timestamp`) so a bad row
  becomes a null instead of failing the job. Nine expectations (missing ids, unparseable dates or amounts, absurd
  amounts, bad currency, empty descriptors, bad MCC...) route failures to `silver.quarantine` with the reasons.
* **De-duplication** on the vendor's transaction id, preferring the original delivery over a re-delivery, both
  within a micro-batch (window `row_number`) and against already-loaded rows (insert-only `MERGE`). Every dropped
  duplicate is logged.

## 5. Entity resolution (`entity_resolution/`)

Card descriptors are noisy strings; the research question is "which public company earned this dollar?".

1. **Normalization** (one rule list executed by both Python `re` and Spark Java regex, tested for parity): strip
   bank boilerplate, masked card numbers, payment wrappers (`SQ *`, `PAYPAL *`), domains, reference codes, store
   numbers, punctuation, stopwords, trailing state and gazetteer city. In the full run 8.8M distinct raw
   descriptors collapse to 32K (normalized descriptor, MCC) pairs, so the expensive steps run on 0.4% of the
   strings and join back to the 19M transactions with a broadcast.
2. **Merchant-of-record rules** run first: `DD *DOORDASH PANERA` is a DoorDash sale, not a Panera sale;
   `IC* COSTCO BY INSTACART` is Instacart's; `UBER *EATS` is Uber's.
3. **Blocking**: TF-IDF character n-grams retrieve the top-k candidate aliases per descriptor.
4. **Scoring**: token-set, token-sort, Jaro-Winkler and TF-IDF cosine combine into one score. Short aliases must
   match a whole token (`ULTA` never matches `ULTRA`), and an MCC the brand never uses multiplies the score by
   0.85, which is what keeps `TARGET OPTICAL` (an optometrist MCC) out of Target's revenue.
5. **Decisions are precision-first**: exact/prefix match with a plausible MCC, or a fuzzy score >= 0.90 with no close
   runner-up from another ticker, is auto-accepted; the gray zone and high-spend unmatched descriptors go to a
   review queue. A false positive silently corrupts a company's signal; a false negative only shrinks the sample.
6. **LLM adjudication** (optional, `llm_adjudicator.py`): the review queue is sent in batches to Claude with a
   JSON-schema-constrained output; every returned brand is validated against the universe, low-confidence answers
   are dropped, and decisions are cached so no descriptor is paid for twice.
7. **Effective-dated ownership**: ER resolves descriptor -> *brand*; a separate brand -> ticker table with
   `valid_from` dates attributes Chuy's to Darden only after 2024-10-11.

## 6. Anomaly detection (`anomaly/`)

The key idea: **compare each data source with the cross-source consensus on the same day.** Real-world shocks
(COVID lockdowns, Black Friday, storms) move every source together and cancel out; data problems move one
source away from the others.

* **Source level** (daily): log-ratio of the source's transactions-per-member and median ticket to the cross-source
  medians, scored against the source's own trailing 28-day median with a robust (MAD) z-score.
  * zero transactions while members are enrolled -> outage (or, if it lasts > 14 days, a dropout)
  * median ticket jumps by ~e^4.6 (100x) -> unit error -> rescale by 0.01
  * extreme *and* large (>= 40%) volume drops -> exclude that source-day
* **Source x ticker level** (tagging breaks): 7-day tagged transaction counts are compared with what the source
  would show if it behaved like the others (consensus rate x member-days x the source's typical multiplier). The
  residual is scaled by a trailing robust dispersion estimate (quasi-Poisson), a break needs a >= 35% shortfall with
  z <= -4 on 7 consecutive days, and the baseline is frozen when a break opens so a permanent format change stays
  flagged.
* **Actions** feed the estimates: excluded source-days and source-ticker spans drop out of both numerator and
  denominator; unit errors are rescaled.

## 7. Panel estimates (`pipeline/estimates.py`, `stats/raking.py`)

Four estimators, each adding one correction, so the evaluation shows what each step is worth:

| estimator | definition | fixes |
|---|---|---|
| raw | total panel spend | nothing (panel growth looks like revenue growth) |
| per_member | spend / enrolled members | panel growth and churn |
| clean | per-member after anomaly actions | outages, dropouts, unit errors, tagging breaks |
| raked | clean, with cells reweighted daily to census margins (IPF) | demographic drift as sources join and leave |

Raking uses iterative proportional fitting over age, income and region margins every day, because the panel's
composition changes whenever a source joins, leaves, or is excluded. Daily estimates are then summed over each
company's **exact fiscal days**, so a 14-week quarter is compared with its 14-week counterpart and 53-week years
need no special handling.

## 8. Nowcasting (`modeling/nowcast.py`)

* **Target**: YoY log growth of reported revenue for the fiscal quarter.
* **Timing**: the nowcast is made 7 days after quarter end (late-posting buffer), typically ~4 weeks before the
  10-Q/10-K. A model may only train on quarters whose revenue was **filed before the nowcast date**, and lag
  features must also be public - the walk-forward is point-in-time by construction (tested).
* **Models**: naive (last quarter's growth), raw-panel OLS, raked panel with ticker intercepts, ridge with
  panel growth / last reported growth / last quarter's panel-vs-reported gap, gradient boosting anchored on the
  panel signal (trees cannot extrapolate into COVID-sized moves, so they model the residual), and an ensemble.
* **Intervals**: 80% bands from each model's own earlier out-of-sample errors (empirical quantiles, no peeking).
* **Metrics**: MAE in percentage points, revenue MAPE, direction of growth acceleration vs. last quarter, share of
  quarters beating naive, interval coverage, split into COVID (2020-21) and post-COVID periods.

## 9. Alternative-data asset evaluation (`modeling/asset_eval.py`)

Any candidate asset is reduced to a fiscal-quarter YoY signal per ticker and scored on coverage, fit (correlation
with reported growth, with and without COVID), incremental out-of-sample skill vs. naive in the same
point-in-time walk-forward, timeliness, and feed quality. The real Wikipedia pageview feed needed its own data
engineering: articles get renamed (`Amazon.com` -> `Amazon (company)` in 2017, `The Home Depot` -> `Home Depot`
in 2024...), and views are counted per requested title, so the pipeline fetches old and new titles, sums them, and
a rename detector (>= 10x quarter-on-quarter jumps per article, then re-checked per ticker) verifies continuity.

## 10. Research-note agent (`agent/`)

A LangGraph state machine: `gather -> analyst <-> tools -> verify -> (revise -> analyst) -> finalize`.

* Tools are **read-only and point-in-time**: the agent sees the data as of the nowcast date, so it cannot see the
  revenue it is predicting. `run_sql` accepts a single SELECT over three tables with a row limit.
* **Verification is deterministic**: every figure in the draft must match a number a tool returned, within the
  rounding the draft itself shows ("$40.9B" may round 40.86; "$41.2B" may not). Required sections and a
  no-investment-advice rule are checked too. Failures go back to the model as specific feedback, at most twice;
  if problems remain, the always-grounded template note is published instead.
* Without an API key the same graph runs with a deterministic template writer, so the pipeline and tests never
  depend on an LLM. With a key, Claude (`claude-opus-5`, server-side refusal fallbacks enabled) drafts the note.

## 11. Evaluation against ground truth (`evaluation.py`)

The only module allowed to read `sim_truth`. It reports entity-resolution precision/recall (by transactions and by
dollars, brand- and ticker-level), quarantine recall for malformed rows, duplicates removed, event-level anomaly
precision/recall with detection delay, and each estimator's YoY error against the true card-visible spend.
