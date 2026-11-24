# RAPIDS Data Analyst

A multi-agent data-analyst crew that answers a business question about a tabular dataset
using **GPU dataframes (RAPIDS cuDF)** instead of SQL, with Claude as the reasoning engine.

```
Planner → Profiler → Cleaner → Analyst ⇄ Critic → Visualizer → Reporter
```

Agents write and execute Python against cuDF, self-correct from tracebacks, and pass
state through files in a per-run workspace. A **Critic** independently recomputes every
number and sends failed findings back to the Analyst (max 2 rounds); the **Reporter** may
cite only verified findings. See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the full design.

## Quick start

```bash
export ANTHROPIC_API_KEY=...
pip install -e ".[dev]"            # CPU: agents fall back to pandas and log it
rapids-analyst data/telco.csv "What drives customer churn?"
```

On a GPU, use the RAPIDS image:

```bash
docker build -t rapids-analyst .
docker run --gpus all -e ANTHROPIC_API_KEY -v $PWD:/work -w /work rapids-analyst \
  data/telco.csv "What drives customer churn?"
```

Each run writes `runs/<id>/` containing `report.md`, `analysis.py` (reproducible script),
`findings.json`, `critique.json`, `figures/`, `fallbacks.json` (ops that left the GPU) and
`trace.jsonl` (every LLM call and code execution).

## Status

- [x] v1 pipeline: Planner, Profiler, Cleaner, Analyst, Critic loop, Visualizer, Reporter
- [x] Eval harness (`evals/`): synthetic churn data with planted ground truth, scoring, GPU-vs-pandas benchmark
- [ ] v2: Modeler agent (cuML), cuGraph, `dask_cudf` for out-of-core data

## Evals

The eval dataset is **synthetic churn data with planted structure** ([evals/synth.py](evals/synth.py)):
`contract` and `tenure` are true drivers, `gender` and `region` are null traps, and the data is
deliberately dirty (numeric strings, blanks, duplicates, nulls). Ground truth comes from a plain
pandas reference ([evals/reference.py](evals/reference.py)), so there is nothing to download.

```bash
python -m evals.run_evals --seeds 0 1 2      # needs ANTHROPIC_API_KEY
python -m evals.bench_speedup --rows 10000000   # needs RAPIDS / a GPU
```

| Score | Meaning |
|---|---|
| `insight_recall` | Share of true drivers (contract, tenure) named by a verified finding |
| `trap_precision` | 1 minus the share of null columns (gender, region) wrongly reported as drivers |
| `numeric_accuracy` | Share of verified findings whose value matches a reference metric (2% tolerance) |
| `verified_fraction` | Share of findings that survived the Critic |
| speedup | cuDF vs pandas wall time on groupby, sort, describe, corr, merge |

Results are written to `evals/results/` (git-ignored). Add your own measured numbers here once run.

## Tests

```bash
pytest   # offline; no API key or GPU required
```
