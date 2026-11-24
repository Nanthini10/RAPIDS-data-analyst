# Workflow: RAPIDS Data Analyst Crew

**Input:** a dataset (CSV/Parquet) and a question, e.g. "What drives churn?"
**Output:** a markdown report with charts, a reproducible `analysis.py`, and a run trace.

## Stages

```
1. INTAKE     Planner
2. PROFILE    Profiler        (cuDF)
3. PREP       Cleaner         (cuDF)
4. ANALYZE    Analyst ‖ Modeler   (cuDF / cuML, parallel)      [Modeler: v2]
5. VISUALIZE  Visualizer      (matplotlib/hvplot on cuDF output)
6. VERIFY     Critic          (can loop back to 3 or 4)
7. REPORT     Reporter
```

v1 ships Planner → Profiler → Cleaner → Analyst → Visualizer → Critic → Reporter.
Modeler (cuML) and cuGraph are v2.

## Agents and contracts

| Agent | Does | RAPIDS tools | Writes to workspace |
|---|---|---|---|
| Planner | Turns the question into hypotheses and a task list | none | `plan.json` |
| Profiler | Schema, dtypes, nulls, cardinality, distributions, GPU memory fit | cuDF | `profile.json` |
| Cleaner | Fixes types, imputes/drops nulls, dedupes, encodes, engineers features | cuDF | `clean.parquet`, `transform_log.json` |
| Analyst | Groupby/aggregation, correlations, segment comparisons, stats tests | cuDF, CuPy | `findings.json` |
| Modeler (v2) | Clustering, anomaly detection, predictive baseline | cuML | `models.json` |
| Visualizer | One chart per finding | matplotlib / hvplot | `figures/*.png`, `figure_index.json` |
| Critic | Re-runs key numbers independently; checks leakage, sample size, relevance | cuDF | `critique.json` |
| Reporter | Writes narrative from verified findings only; exports code | none | `report.md`, `analysis.py` |

## Design decisions

- **Shared state is files, not chat.** Parquet and JSON in a per-run workspace.
  Agents pass references, never raw data, so context stays small.
- **Code-execution agents.** Profiler, Cleaner, Analyst (and later Modeler) write
  Python, run it in a subprocess on the GPU, and get stderr back to self-correct
  (max 3 attempts each).
- **Structured findings.** Every finding is
  `{id, claim, evidence_code, metric, effect_size, n, confidence, figure_ref}`.
- **Critic loop.** Failed findings return to the producing agent with the Critic's
  reason, at most 2 rounds; then the finding is dropped or marked *unverified*.
- **GPU fallback.** If an op is unsupported in cuDF/cuML, the agent may fall back
  to pandas/sklearn and logs it (`fallbacks.json`), which doubles as a RAPIDS
  coverage report.
- **Out-of-core.** Datasets larger than GPU memory use `dask_cudf`; the Profiler
  makes that call and records it in `profile.json`.

## Guardrails

- The Reporter may cite only findings the Critic passed.
- Every number in the report links to its `evidence_code`.
- Generated code runs in a subprocess with a timeout, inside the workspace dir.

## Evaluation

3–5 public datasets with known answers (Telco churn, NYC taxi, Titanic). Score:
1. Was the key insight found?
2. Are the numbers correct versus a reference notebook?
3. Speedup versus pandas on a larger dataset.

## Run workspace layout

```
runs/<run_id>/
  input/            copy of source dataset
  plan.json
  profile.json
  clean.parquet
  transform_log.json
  findings.json
  figures/
  critique.json
  fallbacks.json
  report.md
  analysis.py
  trace.jsonl       every LLM call + code execution, with tokens and timings
```
