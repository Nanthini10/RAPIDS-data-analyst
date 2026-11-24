"""Run agent-generated Python in a subprocess with a timeout.

The prelude gives generated code `xdf`, `to_pd` (cuDF if available, else pandas), `GPU`, and `WS`
(the workspace dir). Falling back to pandas is logged so we can report RAPIDS coverage.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PRELUDE = '''\
import os, json, warnings
warnings.filterwarnings("ignore")
WS = os.environ["RA_WORKSPACE"]
try:
    import cudf as xdf
    GPU = True
except Exception:  # no GPU / RAPIDS not installed
    import pandas as xdf
    GPU = False
    open(os.path.join(WS, "_no_gpu"), "w").close()
    # CPU shim: code written for cuDF often calls cuDF-only methods. Make the common ones no-ops on pandas
    # so the same script runs on both backends.
    for _cls in (xdf.DataFrame, xdf.Series, xdf.Index):
        if not hasattr(_cls, "to_pandas"):
            _cls.to_pandas = lambda self, *a, **k: self
        if not hasattr(_cls, "to_cupy"):
            _cls.to_cupy = lambda self, *a, **k: self.to_numpy()


def to_pd(x):
    """cuDF -> pandas; no-op if already pandas/numpy. Use instead of calling .to_pandas() directly."""
    return x.to_pandas() if hasattr(x, "to_pandas") else x
'''


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    script: Path
    used_gpu: bool | None = None


def run_python(code: str, script_path: Path, workspace_root: Path, timeout: int = 300) -> ExecResult:
    script_path.write_text(PRELUDE + "\n" + code)
    env = {**os.environ, "RA_WORKSPACE": str(workspace_root)}
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=workspace_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(False, e.stdout or "", f"Timed out after {timeout}s", script_path)
    return ExecResult(proc.returncode == 0, proc.stdout[-4000:], proc.stderr[-4000:], script_path)
