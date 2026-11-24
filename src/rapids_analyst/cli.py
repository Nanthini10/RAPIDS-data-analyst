from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .orchestrator import PipelineError, run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rapids-analyst", description="Multi-agent RAPIDS data analyst crew")
    p.add_argument("dataset", type=Path, help="CSV or Parquet file")
    p.add_argument("question", help="Question to answer, e.g. 'What drives churn?'")
    p.add_argument("--out", type=Path, default=Path("runs"), help="Directory for run workspaces")
    p.add_argument("--model", default=None, help="Claude model id (default: $RAPIDS_ANALYST_MODEL)")
    args = p.parse_args(argv)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    try:
        ws = run_pipeline(args.dataset, args.question, args.out, args.model)
    except PipelineError as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 1
    print(f"Done. Report: {ws.path('report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
