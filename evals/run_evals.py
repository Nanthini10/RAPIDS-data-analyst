"""Run the crew on the synthetic churn dataset and score it.

  python -m evals.run_evals                      # generate data, run pipeline, score
  python -m evals.run_evals --seeds 0 1 2        # repeat for variance
  python -m evals.run_evals --score-only runs/<id> --seed 0   # re-score an existing run
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .score import score_run
from .synth import make_churn

QUESTION = "What drives customer churn, and which factors do not?"
RESULTS = Path(__file__).parent / "results"


def run_one(seed: int, n: int, out: Path, model: str | None) -> dict:
    from rapids_analyst.orchestrator import run_pipeline  # needs API key

    raw = make_churn(n=n, seed=seed)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / f"churn_seed{seed}.csv"
    raw.to_csv(csv, index=False)
    ws = run_pipeline(csv, QUESTION, out / "runs", model)
    return {"seed": seed, "run_dir": str(ws.root), **score_run(ws.root, raw)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--out", type=Path, default=Path("eval_runs"))
    p.add_argument("--model", default=None)
    p.add_argument("--score-only", type=Path)
    p.add_argument("--seed", type=int, default=0, help="seed used for --score-only")
    args = p.parse_args()

    if args.score_only:
        res = [{"seed": args.seed, **score_run(args.score_only, make_churn(args.n, args.seed))}]
    else:
        res = [run_one(s, args.n, args.out, args.model) for s in args.seeds]

    summary = {
        k: statistics.mean(r[k] for r in res)
        for k in ("insight_recall", "trap_precision", "numeric_accuracy", "verified_fraction")
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps({"summary": summary, "runs": res}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
