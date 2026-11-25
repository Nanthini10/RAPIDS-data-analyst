"""Score a finished run workspace against planted ground truth.

Scores (all in [0,1]):
  insight_recall    - fraction of true drivers mentioned by >=1 verified finding
  trap_precision    - 1 - fraction of null-trap columns reported as a driver
  numeric_accuracy  - fraction of verified findings whose numbers are correct (see below)
  verified_fraction - fraction of all findings that survived the Critic

A finding's numbers are correct if EVERY "<rate> (n=<size>)" pair quoted in its claim exists in the
reference group table (rate within 0.002, n exact). Findings that quote no such pair (e.g. a
correlation) are checked by matching `value` against the small named-scalar reference instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .reference import reference_groups, reference_metrics
from .synth import DRIVERS, NULL_TRAPS

RATE_TOL = 0.002
SCALAR_REL_TOL = 0.01

# "0.330, n=10913)", "0.424 (n=3,302)", "33.0% (n=10913)"
# also "43.5% (0-12 mo, n=3311)": a short label may sit between the rate and n, but no "=" (so "r=0.06, p=1e-9, n=20000" is skipped)
_PAIR = re.compile(r"(?<![\w.=\-])(\d+(?:\.\d+)?)(%?)(?:\s*\(\s*(?:[^=()]{0,25}?[,\s])?|\)?,\s*)n\s*=\s*([\d,]+)")
# metrics that are test statistics, not rates/means: the scorer has no reference for them, so they are not scored
_TEST_METRIC = re.compile(r"chi|p_value|p-value|cramer|interaction|anova|f_stat", re.I)
# a claim that merely says "no effect" is CORRECT behaviour for a trap column
_NEGATION = re.compile(
    r"\b(no|not|little|negligible|minimal|neither|insignificant|similar|nearly identical|almost identical|null)\b", re.I
)


def _mentions(claim: str, col: str) -> bool:
    return col.replace("_", " ") in claim.lower().replace("_", " ")


def quoted_pairs(claim: str) -> list[tuple[float, int]]:
    pairs = []
    for num, pct, n in _PAIR.findall(claim):
        rate = float(num) / 100 if pct else float(num)
        pairs.append((rate, int(n.replace(",", ""))))
    return pairs


def _pair_ok(pair: tuple[float, int], groups: list[tuple[float, int]]) -> bool:
    rate, n = pair
    return any(pn == n and abs(rate - pr) <= RATE_TOL for pr, pn in groups)


def _scalar_ok(value: float, scalars: dict[str, float]) -> bool:
    for r in scalars.values():
        for cand in (r, -r):
            if abs(value - cand) <= max(RATE_TOL, SCALAR_REL_TOL * abs(cand)):
                return True
    return False


def finding_numbers_correct(f: dict, groups, scalars) -> bool | None:
    """True/False, or None when the finding quotes nothing the scorer can check."""
    pairs = quoted_pairs(f["claim"])
    if pairs:
        return all(_pair_ok(p, groups) for p in pairs)
    if _scalar_ok(f["value"], scalars):
        return True
    return None if _TEST_METRIC.search(f.get("metric", "")) else False


def score_run(run_dir: Path, raw: pd.DataFrame) -> dict:
    groups, scalars = reference_groups(raw), reference_metrics(raw)
    findings = json.loads((run_dir / "findings.json").read_text())
    verified = [f for f in findings if f.get("status") == "verified"]

    found = {d: any(_mentions(f["claim"], d) for f in verified) for d in DRIVERS}
    false_driver = {
        t: any(not _NEGATION.search(f["claim"]) for f in verified if _mentions(f["claim"], t)) for t in NULL_TRAPS
    }
    checks = [finding_numbers_correct(f, groups, scalars) for f in verified]
    scored = [(f, c) for f, c in zip(verified, checks) if c is not None]
    ok = [c for _, c in scored]

    return {
        "n_findings": len(findings),
        "n_verified": len(verified),
        "verified_fraction": len(verified) / len(findings) if findings else 0.0,
        "drivers_found": found,
        "insight_recall": sum(found.values()) / len(DRIVERS),
        "traps_misreported": false_driver,
        "trap_precision": 1 - sum(false_driver.values()) / len(NULL_TRAPS),
        "numeric_accuracy": sum(ok) / len(ok) if ok else 0.0,
        "incorrect_numbers": [f["id"] for f, g in scored if not g],
    }
