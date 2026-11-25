"""Run the crew and the single-shot baseline on a real dataset and compare them.

  python -m evals.real.run_real --months 1                         # smoke (~3M rows), crew + baseline
  python -m evals.real.run_real --months 12 --compare-backends     # scale test on a GPU box
  python -m evals.real.run_real --questions payment zones --systems crew

Writes evals/results/real_latest.json. Needs NVIDIA_API_KEY (or the provider env vars in README).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .datasets import fetch_nyc_taxi
from .questions import QUESTIONS, build_checks
from .score import score_answer

RESULTS = Path(__file__).resolve().parent.parent / "results"


def trace_stats(run_dir: Path) -> dict:
    recs = [json.loads(l) for l in (run_dir / "trace.jsonl").read_text().splitlines() if l.strip()] \
        if (run_dir / "trace.jsonl").exists() else []
    llm = [r for r in recs if r["event"] == "llm_call"]
    ex = [r for r in recs if r["event"] == "exec"]
    return {
        "llm_calls": len(llm),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in llm),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in llm),
        "llm_seconds": round(sum(r.get("seconds") or 0 for r in llm), 1),
        "exec_seconds": round(sum(r.get("seconds") or 0 for r in ex), 1),   # compute time of generated code
        "exec_failures": sum(1 for r in ex if not r.get("ok")),
        "wall_seconds": round(recs[-1]["t"] - recs[0]["t"], 1) if recs else 0.0,
    }


def load_findings(run_dir: Path, verified_only: bool) -> list[dict]:
    f = run_dir / "findings.json"
    if not f.exists():
        return []
    found = json.loads(f.read_text())
    return [x for x in found if x.get("status") == "verified"] if verified_only else found


def time_backends(run_dir: Path) -> dict | None:
    """Re-run the exported analysis.py on the GPU backend and again forced onto pandas; compare wall time."""
    script = run_dir / "analysis.py"
    if not script.exists():
        return None
    out = {}
    for name, extra in (("default", {}), ("pandas", {"RA_FORCE_PANDAS": "1"})):
        env = {**os.environ, "RA_WORKSPACE": str(run_dir), **extra}
        t = time.perf_counter()
        r = subprocess.run([sys.executable, str(script)], cwd=run_dir, env=env, capture_output=True, text=True)
        out[name] = {"ok": r.returncode == 0, "seconds": round(time.perf_counter() - t, 1)}
    out["gpu_available"] = not (run_dir / "_no_gpu").exists()
    if out["default"]["ok"] and out["pandas"]["ok"] and out["default"]["seconds"] > 0:
        out["speedup"] = round(out["pandas"]["seconds"] / out["default"]["seconds"], 2)
    return out


def run_system(system: str, q, dataset: Path, out: Path, model, key: list[dict], compare: bool) -> dict:
    from rapids_analyst.orchestrator import PipelineError, run_pipeline
    from .baseline import run_baseline

    err, run_dir = None, None
    try:
        if system == "crew":
            ws = run_pipeline(dataset, q.text, out / "runs", model)
        else:
            ws = run_baseline(dataset, q.text, out / "runs", model)
        run_dir = ws.root
    except PipelineError as e:
        err, run_dir = str(e), e.run_dir
    except Exception as e:  # noqa: BLE001  one bad run must not kill the whole comparison
        err = f"{type(e).__name__}: {e}"
    res = {"question": q.id, "system": system, "run_dir": str(run_dir) if run_dir else None, "error": err}
    findings = load_findings(run_dir, verified_only=(system == "crew")) if run_dir else []
    res.update(score_answer(key, findings) if findings else {"score": 0.0, "fully_correct": False, "checks": {}})
    res["completed"] = err is None and bool(findings)
    if run_dir:
        res.update(trace_stats(run_dir))
        if compare and system == "crew" and err is None:
            res["backends"] = time_backends(run_dir)
    return res


def summarise(runs: list[dict]) -> dict:
    out = {}
    for system in sorted({r["system"] for r in runs}):
        rs = [r for r in runs if r["system"] == system]
        out[system] = {
            "mean_check_score": round(statistics.mean(r["score"] for r in rs), 3),
            "fully_correct": f"{sum(r['fully_correct'] for r in rs)}/{len(rs)}",
            "completed": f"{sum(r['completed'] for r in rs)}/{len(rs)}",
            "mean_output_tokens": round(statistics.mean(r.get("output_tokens", 0) for r in rs)),
            "mean_wall_seconds": round(statistics.mean(r.get("wall_seconds", 0) for r in rs), 1),
            "mean_exec_seconds": round(statistics.mean(r.get("exec_seconds", 0) for r in rs), 1),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=1, help="NYC taxi months to use (1 ~ 3M rows, 12 ~ 38M)")
    p.add_argument("--data", type=Path, help="use this parquet instead of downloading")
    p.add_argument("--systems", nargs="+", default=["crew", "baseline"], choices=["crew", "baseline"])
    p.add_argument("--questions", nargs="+", default=[q.id for q in QUESTIONS])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("eval_runs_real"))
    p.add_argument("--model", default=None)
    p.add_argument("--compare-backends", action="store_true", help="time analysis.py on cuDF vs forced pandas (GPU box)")
    a = p.parse_args()

    dataset = a.data or fetch_nyc_taxi(a.months)
    qs = [q for q in QUESTIONS if q.id in a.questions]
    keys = {q.id: build_checks(q, dataset) for q in qs}   # answer keys first: independent of any model output
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "real_keys.json").write_text(json.dumps(keys, indent=2))

    runs = []
    for q in qs:
        for system in a.systems:
            for rep in range(a.repeats):
                print(f"== {q.id} / {system} / run {rep + 1}", file=sys.stderr, flush=True)
                r = run_system(system, q, dataset, a.out, a.model, keys[q.id], a.compare_backends)
                r["repeat"] = rep
                runs.append(r)
                print(f"   score={r['score']:.2f} completed={r['completed']} wall={r.get('wall_seconds')}s", file=sys.stderr)
                (RESULTS / "real_latest.json").write_text(json.dumps({"runs": runs, "summary": summarise(runs)}, indent=2))
    print(json.dumps(summarise(runs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
