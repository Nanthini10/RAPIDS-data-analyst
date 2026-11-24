"""Score a finished run workspace against planted ground truth.

Three scores (all in [0,1]):
  insight_recall    - fraction of true drivers mentioned by >=1 verified finding
  trap_precision    - 1 - fraction of null-trap columns reported as a driver
  numeric_accuracy  - fraction of verified findings whose value matches a reference metric
Also reported: verified_fraction (how many findings survived the Critic).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .synth import DRIVERS, NULL_TRAPS

REL_TOL = 0.02
ABS_TOL = 0.01

# A claim that merely says "no effect" is the CORRECT behaviour for a trap column.
_NEGATION = re.compile(r"\b(no|not|little|negligible|minimal|neither|insignificant|similar|nearly identical|almost identical)\b", re.I)


def _mentions(claim: str, col: str) -> bool:
    return col.replace("_", " ") in claim.lower().replace("_", " ")


def _matches(value: float, ref: dict[str, float]) -> bool:
    for r in ref.values():
        for cand in (r, -r, r * 100):  # allow sign flips and percentages
            if abs(value - cand) <= max(ABS_TOL, REL_TOL * abs(cand)):
                return True
    return False


def score_run(run_dir: Path, ref: dict[str, float]) -> dict:
    findings = json.loads((run_dir / "findings.json").read_text())
    verified = [f for f in findings if f.get("status") == "verified"]

    found = {d: any(_mentions(f["claim"], d) for f in verified) for d in DRIVERS}
    insight_recall = sum(found.values()) / len(DRIVERS)

    false_driver = {}
    for t in NULL_TRAPS:
        claims = [f["claim"] for f in verified if _mentions(f["claim"], t)]
        false_driver[t] = any(not _NEGATION.search(c) for c in claims)
    trap_precision = 1 - sum(false_driver.values()) / len(NULL_TRAPS)

    numeric_hits = [_matches(f["value"], ref) for f in verified]
    numeric_accuracy = sum(numeric_hits) / len(numeric_hits) if numeric_hits else 0.0

    return {
        "n_findings": len(findings),
        "n_verified": len(verified),
        "verified_fraction": len(verified) / len(findings) if findings else 0.0,
        "drivers_found": found,
        "insight_recall": insight_recall,
        "traps_misreported": false_driver,
        "trap_precision": trap_precision,
        "numeric_accuracy": numeric_accuracy,
        "unmatched_values": [f["id"] for f, ok in zip(verified, numeric_hits) if not ok],
    }
