# Interview guide

How to put PanelCast on a resume honestly and talk about it under questioning. Every number below comes from
`reports/RESULTS.md` (full run: 20,000 panelists, 19.4M simulated transactions, 20 companies, 2018-2026).

## Resume entry

**PanelCast: Alternative-Data Revenue Nowcasting Lakehouse** | *Independent Project - PySpark, Delta Lake, Databricks, SQL, scikit-learn, LangGraph, Claude API*

- Built a PySpark/Delta Lake medallion pipeline (incremental Auto Loader-style ingestion, schema expectations with a
  quarantine table, MERGE-based deduplication) over 19M simulated card transactions from 5 data vendors, packaged as
  a 13-task Databricks serverless job; 56 pytest unit/Spark tests plus an end-to-end test in CI.
- Engineered merchant entity resolution (Python/Spark regex normalization, TF-IDF blocking, fuzzy + MCC scoring,
  effective-dated M&A brand ownership, LLM review queue) at 100% precision / 98% recall, and cross-source anomaly
  detection (robust z-scores, quasi-Poisson tests) that caught 13/13 injected data failures with 0 false positives.
- Nowcast quarterly revenue for 20 public companies against real SEC EDGAR filings aligned to 52/53-week fiscal
  calendars: daily census raking cut panel growth error from 28.7 to 1.9 pp, and a point-in-time ridge + gradient
  boosting ensemble reached 2.1 pp MAE vs. 6.3 pp naive ~26 days before filings; added a LangGraph agent that writes
  number-verified pre-earnings notes.

Shorter, if space is tight - keep bullets 1 and 3 and fold "entity resolution at 100% precision" into bullet 1.

Skills line additions: **PySpark, Spark SQL, Delta Lake, Databricks, LangGraph, LangChain, entity resolution,
anomaly detection, data quality testing.**

Always say "simulated card transactions anchored to real SEC revenue" when asked about the data. The honesty is
a strength here: you can explain exactly why you simulated and what is real.

## 60-second pitch

"Alt-data firms buy card-transaction panels and turn them into revenue estimates before companies report. I built
that loop end to end. Since real card data costs six figures, I simulated a 19-million-transaction panel from five
data vendors, but anchored each company's spend to its real revenue from SEC filings, so my nowcasts are scored
against what the companies actually reported. The pipeline is PySpark and Delta in a bronze/silver/gold layout that
deploys to Databricks. The interesting parts are the problems in the middle: mapping messy strings like
`SQ *STARBUCKS #1234` to tickers without false positives, catching vendor outages and unit errors automatically, and
correcting for a panel whose demographics keep shifting. Each correction is measured against ground truth - raw panel
growth is off by 29 points, the fully corrected panel by under 2 - and the final ensemble beats a naive forecast by
two-thirds, about four weeks before each 10-Q."

## Likely questions (by job-description bullet)

**"Isn't simulated data circular? You generated it from the answer."**
The simulator models the *measurement process*, not the answer: panel spend is real revenue times an unobservable,
drifting visibility ratio, plus sampling noise, demographic bias, churn and data failures. The ratio drift is
irreducible error the models must live with, as in production. Simulation buys ground truth, so I can measure
entity-resolution precision, anomaly recall and estimator bias - things you cannot measure on a vendor feed without
labels. Revenue, fiscal calendars, filing dates, M&A dates and the Wikipedia asset are all real.

**"Walk me through your ingestion pipeline."** (data ingestion with Python, SQL, Databricks, Spark)
Bronze lands every vendor file as strings with Auto Loader (`availableNow` + checkpoint), so each file is processed
exactly once and re-runs are no-ops; I keep `_metadata.file_path` for lineage. Silver streams from bronze with
`foreachBatch`: ANSI-safe casts (`try_cast`, `try_to_timestamp`) so one bad row cannot fail a job, nine expectations
that route failures to a quarantine table with reasons, window-function dedup that prefers the original delivery,
and an insert-only `MERGE` for idempotency. In the full run: 3,781 of 3,781 malformed rows quarantined, 10,908
duplicate re-deliveries removed, row counts reconcile exactly (tested).

**"How does entity resolution avoid false positives?"** (named entity resolution / NLP)
Normalize first (8.8M raw strings collapse to 32K forms, so the expensive part runs on 0.4% of the data), then
merchant-of-record rules (a DoorDash order of Chipotle is DoorDash revenue), TF-IDF character n-grams to block
candidates, and fuzzy features with two guards: the MCC must be plausible for the brand (`TARGET OPTICAL` is an
optometrist) and short aliases must match a whole token (`ULTA` vs `ULTRA CLEAN CAR WASH`). Thresholds are
precision-first because a false positive silently corrupts a company's signal while a miss only shrinks the sample.
The gray zone goes to an LLM with schema-constrained output, validated against the brand list and cached.

**"How do you find data problems without labels?"** (data testing, anomaly detection)
Compare each source with the other sources on the same day. COVID or Black Friday move all sources together; an
outage, a cents-instead-of-dollars file, or a broken descriptor feed moves one source away from the consensus. I score
that log-ratio against the source's own trailing median with a robust MAD z-score, and for per-ticker breaks I use a
quasi-Poisson count test with a frozen baseline so permanent breaks stay flagged. 13/13 injected failures detected,
0 false positives. The first version flagged dozens of false breaks on small tickers; switching from a MAD z-score
on overlapping 7-day windows to a count test fixed it - a good story about debugging a detector.

**"What is raking and why did you need it?"**
The panel's demographics drift as data sources join and leave (a young fintech panel joins, a Midwestern issuer
leaves). Raking (iterative proportional fitting) reweights cells each day to match census margins for age, income
and region. It cut panel growth error from 3.0 to 1.9 pp on top of anomaly handling. I used raking rather than full
post-stratification because only margins are assumed known and some cells are sparse.

**"How do you prevent look-ahead bias?"**
Three places: revenue is stored as first reported (restatements tracked separately), the walk-forward trains only on
quarters filed before each nowcast date (a unit test asserts it), and the agent's tools see a point-in-time view
where unfiled revenue is masked, so it cannot see the number it is predicting. I also found that companies re-tag
revenue concepts (ASC 606), which made the first-public date look later than it was for 52 quarters; the fix takes
the earliest filing across all revenue concepts.

**"Why does raw panel spend do worse than no alt data at all?"**
Raw totals grow when the panel grows, shrink when a vendor leaves, and jump 100x when amounts arrive in cents. Panel
mechanics swamp the economic signal (10.9 pp error vs. 6.3 naive). Per-member normalization, anomaly handling and
raking are what make the asset valuable (2.1 pp).

**"How did you evaluate a new data asset?"** (evaluate new alternative data assets)
Reduce it to a fiscal-quarter growth signal and score coverage, correlation with reported growth (with and without
COVID), incremental out-of-sample skill over naive in the same point-in-time walk-forward, timeliness, and feed
quality. The real Wikipedia pageview feed scored +3% skill - weak - and needed real data engineering: article
renames (`Amazon.com` -> `Amazon (company)`) fragment the history, so I built a rename detector and summed old and
new titles.

**"How does the agent avoid hallucinated numbers?"** (AI agents, LLM orchestration)
A LangGraph state machine: gather facts, draft with read-only tools, then a deterministic verifier checks every
figure in the draft against numbers the tools returned, at the precision the draft shows. Failures go back as
specific feedback, at most twice; if it still fails, the always-grounded template note is published. The model
never grades itself. Tests drive the whole loop with a scripted fake LLM.

**"How would this scale to billions of rows?"** (distributed data processing)
Already designed for it: entity resolution runs on distinct descriptors and broadcasts back; gold aggregates before
any pandas step; per-series statistics run in parallel with `applyInPandas`. Next steps would be liquid clustering on
`txn_date`/`ticker`, Auto Loader file-notification mode, incremental gold (only recompute affected dates), and
caching ER decisions across runs (already done for LLM decisions).

**"What are the weaknesses?"** (always volunteer these)
The transactions are simulated; the panel is small, so small tickers are noisy; the LLM paths are tested with a
scripted model rather than measured live; the Databricks bundle is written to spec but [deploy it before the
interview and update this line].

## Know your numbers

| | |
|---|---|
| Transactions / panelists / sources / companies | 19.4M / 20,000 / 5 / 20 |
| Raw descriptors -> normalized forms | 8.8M -> 32K |
| ER precision / recall (ticker, $) | 100% / 98.0% |
| Anomalies detected / false positives | 13 of 13 / 0 |
| Panel error raw -> per member -> clean -> raked | 28.7 -> 20.4 -> 3.0 -> 1.9 pp |
| Nowcast MAE ensemble vs. naive | 2.1 vs. 6.3 pp (COVID: 2.6 vs. 13.6) |
| Revenue MAPE / beats naive / median lead | 2.1% / 71% of quarters / 26 days |
| Wikipedia pageviews skill | +3% (weak) |
| Full local run time | 6.7 minutes (12 Spark cores) |

## Make it yours before you interview

Interviewers probe what you changed and why. Do at least two of these yourself:

1. **Deploy to Databricks Free Edition** ([DATABRICKS.md](DATABRICKS.md)), screenshot the job graph and a
   `DESCRIBE HISTORY`, and fix whatever the bundle needs. This makes "Databricks" on your resume first-hand.
2. **Turn on the LLM** (`ANTHROPIC_API_KEY`) and measure entity-resolution recall with and without the review queue
   (`panelcast run resolve evaluate`). Report the cost per resolved descriptor.
3. **Panel-size study**: run `--scale 0.5, 1, 2, 4` and plot nowcast error vs. panel size. It answers "how big
   does a panel need to be?" - a question alt-data firms ask about every new vendor.
4. **Add an asset**: FRED retail sales or Census Monthly Retail Trade (both free) as a third scorecard row.
5. **Same-store cohort estimator**: measure growth only on members active in both years and compare with raking.
