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
- [ ] Eval harness (`evals/`): public datasets with reference answers, plus GPU-vs-pandas speedup
- [ ] v2: Modeler agent (cuML), cuGraph, `dask_cudf` for out-of-core data

## Tests

```bash
pytest   # offline; no API key or GPU required
```
