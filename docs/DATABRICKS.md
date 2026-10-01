# Running PanelCast on Databricks (Free Edition works)

Everything in the pipeline uses APIs that run on Databricks serverless compute: Auto Loader with
`trigger(availableNow=True)`, Structured Streaming `foreachBatch` + `MERGE`, Delta tables in Unity Catalog,
`mapInPandas` / `applyInPandas`, and Unity Catalog volumes for vendor files and checkpoints. Platform detection is
automatic (`DATABRICKS_RUNTIME_VERSION`): tables become `workspace.panelcast.<layer>_<name>` and files go to
`/Volumes/workspace/panelcast/landing`.

> I wrote the bundle and notebook against the documented interfaces, but I could not deploy them from this
> machine (no workspace credentials). Run `databricks bundle validate` first and expect to fix small things.

## Option A - Git folder + notebook (no CLI, ~10 minutes of clicking)

1. Sign up for [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. Push this repo to GitHub, then in the workspace: **Workspace -> Create -> Git folder** and paste the repo URL.
3. Open `notebooks/panelcast_walkthrough.py`, attach **Serverless**, and **Run all**. It installs the Python
   dependencies, runs every stage at half scale (~5M transactions), and displays the key tables.
4. Explore `workspace.panelcast` in **Catalog Explorer**; reports land in
   `/Volumes/workspace/panelcast/landing/reports`.

## Option B - Declarative Automation Bundle (CLI, the "production" path)

`databricks.yml` builds the PanelCast wheel and deploys a 13-task serverless job (one task per stage, with the
dependency graph of the medallion architecture).

Install the current Databricks CLI (the Go binary; the old `pip install databricks-cli` package cannot deploy
bundles): `winget install Databricks.DatabricksCLI` on Windows, `brew install databricks/tap/databricks` on macOS.

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks bundle validate
databricks bundle deploy
databricks bundle run panelcast_pipeline
```

Scale the simulated panel with a bundle variable: `databricks bundle deploy --var="scale=2.0"`.

## Optional: Claude for entity-resolution review and research notes

```bash
databricks secrets create-scope panelcast
databricks secrets put-secret panelcast anthropic_api_key
```

`panelcast.llm` reads that secret on Databricks; locally it uses `ANTHROPIC_API_KEY`. Without a key both LLM steps
fall back to deterministic logic. Free Edition restricts outbound internet from serverless compute, so calls to
`api.anthropic.com` may be blocked there - run `panelcast run resolve agent` locally in that case.

## What to look at (interview demo)

* **Job graph**: Workflows -> panelcast-pipeline -> a run. The DAG mirrors bronze -> silver -> gold.
* **Delta history**: `DESCRIBE HISTORY workspace.panelcast.silver_transactions` shows the streaming `MERGE`s.
* **Idempotency**: re-run the `bronze` task; Auto Loader reports zero new files.
* **Lineage**: Catalog Explorer -> `gold_ticker_quarterly` -> Lineage.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PERSIST TABLE is not supported on serverless` | PanelCast never caches; make sure you did not add `.cache()` |
| Library install fails in the job environment | Check that `dist/*.whl` was built (`databricks bundle deploy` runs the build) |
| `CREATE VOLUME` permission error | Your catalog differs: set `storage.databricks.catalog` in `conf/panelcast.yaml` |
| Agent step only writes template notes | No `panelcast/anthropic_api_key` secret, or outbound internet is blocked |
