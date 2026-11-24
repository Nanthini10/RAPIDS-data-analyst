# RAPIDS Data Analyst

A multi-agent data-analyst crew that answers a business question about a tabular dataset
using **GPU dataframes (RAPIDS cuDF)** instead of SQL, with an LLM (default: NVIDIA Nemotron via build.nvidia.com) as the reasoning engine.

```
Planner → Profiler → Cleaner → Analyst ⇄ Critic → Visualizer → Reporter
```

Agents write and execute Python against cuDF, self-correct from tracebacks, and pass
state through files in a per-run workspace. A **Critic** independently recomputes every
number and sends failed findings back to the Analyst (max 2 rounds); the **Reporter** may
cite only verified findings. A finding is verified only if the Critic's independent recomputation agrees **and** it clears a deterministic statistical gate computed by the Critic (omnibus p-value and smallest-group size: effect claims need p ≤ 0.05, null claims need p > 0.05, groups need n ≥ 30). See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the full design.

## Quick start

```bash
export NVIDIA_API_KEY=...         # from build.nvidia.com
# optional: RAPIDS_ANALYST_MODEL, or RAPIDS_ANALYST_PROVIDER=openai|anthropic
pip install -e ".[dev]"            # CPU: agents fall back to pandas and log it
rapids-analyst data/telco.csv "What drives customer churn?"
```

On a GPU, use the RAPIDS image:

```bash
docker build -t rapids-analyst .
docker run --gpus all -e NVIDIA_API_KEY -v $PWD:/work -w /work rapids-analyst \
  data/telco.csv "What drives customer churn?"
```

Each run writes `runs/<id>/` containing `report.md`, `analysis.py` (reproducible script),
`findings.json`, `critique.json`, `figures/`, `fallbacks.json` (ops that left the GPU) and
`trace.jsonl` (every LLM call and code execution).

## Dashboard

A Streamlit UI for uploading a CSV/Parquet file, entering an API key, watching the crew work
(live stage tracker and log) and reading the report, findings, charts and downloads.

```bash
pip install -e ".[dashboard]"
rapids-dashboard
```

The key is held in memory for the session and is never written to disk or the run trace. Past runs
can be reopened from the sidebar. Because agents write Python that runs locally, use it with models
and data you trust.

## Status

- [x] v1 pipeline: Planner, Profiler, Cleaner, Analyst, Critic loop, Visualizer, Reporter
- [x] Eval harness (`evals/`): synthetic churn data with planted ground truth, scoring, GPU-vs-pandas benchmark
- [x] Streamlit dashboard (upload, API key, live progress, report viewer)
- [ ] v2: Modeler agent (cuML), cuGraph, `dask_cudf` for out-of-core data

## Evals

The eval dataset is **synthetic churn data with planted structure** ([evals/synth.py](evals/synth.py)):
`contract` and `tenure` are true drivers, `gender` and `region` are null traps, and the data is
deliberately dirty (numeric strings, blanks, duplicates, nulls). Ground truth comes from a plain
pandas reference ([evals/reference.py](evals/reference.py)), so there is nothing to download.

```bash
python -m evals.run_evals --seeds 0 1 2      # needs NVIDIA_API_KEY
python -m evals.bench_speedup --rows 10000000   # needs RAPIDS / a GPU
```

| Score | Meaning |
|---|---|
| `insight_recall` | Share of true drivers (contract, tenure) named by a verified finding |
| `trap_precision` | 1 minus the share of null columns (gender, region) wrongly reported as drivers |
| `numeric_accuracy` | Share of verified findings whose quoted `rate (n=size)` pairs all exist in the pandas reference table (n exact, rate within 0.002); findings with no such pair are checked by `value` against named reference scalars |
| `verified_fraction` | Share of findings that survived the Critic |
| speedup | cuDF vs pandas wall time on groupby, sort, describe, corr, merge |

Results are written to `evals/results/` (git-ignored). Add your own measured numbers here once run.

## Tests

```bash
pytest   # offline; no API key or GPU required
```
