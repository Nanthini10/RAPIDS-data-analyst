"""GPU (cuDF) vs CPU (pandas) speedup on typical analyst operations. Requires RAPIDS.

  python -m evals.bench_speedup --rows 10000000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .synth import make_bench


def _time(fn, repeats: int = 3) -> float:
    fn()  # warm-up (JIT / allocator)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def ops(df):
    return {
        "groupby_agg": lambda: df.groupby(["k1", "k2"]).agg({"x": "mean", "y": "std", "z": "sum"}),
        "sort": lambda: df.sort_values(["k1", "z"]),
        "describe": lambda: df.describe(),
        "corr": lambda: df[["x", "y", "z"]].corr(),
        "merge": lambda: df.merge(df.groupby("k1").x.mean().reset_index(), on="k1"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5_000_000)
    args = p.parse_args()

    try:
        import cudf
    except ImportError:
        print("cuDF not installed; speedup benchmark needs a RAPIDS environment.")
        return 2

    pdf = make_bench(args.rows)
    gdf = cudf.from_pandas(pdf)
    cpu = {k: _time(f) for k, f in ops(pdf).items()}
    gpu = {k: _time(f) for k, f in ops(gdf).items()}
    out = {k: {"pandas_s": cpu[k], "cudf_s": gpu[k], "speedup": cpu[k] / gpu[k]} for k in cpu}
    out["_rows"] = args.rows

    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    (res / "speedup.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if k != "_rows":
            print(f"{k:12s} pandas {v['pandas_s']:.3f}s  cudf {v['cudf_s']:.3f}s  {v['speedup']:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
