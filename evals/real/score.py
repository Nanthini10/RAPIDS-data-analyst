"""Score an answer (list of findings) against an answer key from questions.py."""
from __future__ import annotations

import re

_NUM = re.compile(r"(?<![\w.])-?\$?(\d[\d,]*(?:\.\d+)?)\s*(%|k\b|m\b|million|thousand)?", re.I)
_NEG = re.compile(r"\b(no|not|negligible|minimal|minor|trivial|tiny|small|slight|weak|little|does ?n.t|doesn.t|"
                  r"practically (?:no|zero)|insignificant|unimportant|immaterial)\b", re.I)
_STRONG = re.compile(r"\b(strong|large|substantial|major|significant(?:ly)? (?:affect|impact|driv|influenc))", re.I)


def numbers(text: str) -> list[float]:
    out = []
    for raw, suf in _NUM.findall(text):
        try:
            x = float(raw.replace(",", ""))
        except ValueError:
            continue
        s = (suf or "").lower()
        if s in ("k", "thousand"):
            x *= 1e3
        elif s in ("m", "million"):
            x *= 1e6
        out.append(x)
    return out


def number_matches(value: float, nums: list[float], tol: float) -> bool:
    cands = {value, value * 100, value / 100}
    return any(abs(x - c) <= tol * abs(c) for x in nums for c in cands if c != 0)


def check_one(check: dict, findings: list[dict]) -> bool:
    text = " ".join(str(f.get("claim", "")) for f in findings)
    if check["kind"] == "number":
        nums = numbers(text) + [float(f["value"]) for f in findings if isinstance(f.get("value"), (int, float))]
        return number_matches(check["value"], nums, check["tol"])
    topical = [f for f in findings if re.search(check["topic"], str(f.get("claim", "")), re.I)]
    if not topical:
        return False
    # judge the sentences about the topic: negligible language and no "strong/significant effect" language
    claims = " ".join(str(f["claim"]) for f in topical)
    says_null = bool(_NEG.search(claims)) and not _STRONG.search(claims)
    return says_null if check["kind"] == "null" else not says_null


def score_answer(checks: list[dict], findings: list[dict]) -> dict:
    results = {c["label"]: check_one(c, findings) for c in checks}
    passed = sum(results.values())
    return {"checks": results, "score": passed / len(checks) if checks else 0.0,
            "fully_correct": passed == len(checks) and bool(checks)}
