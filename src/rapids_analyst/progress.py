"""Progress tracking: live log lines during a run, and a status table from trace.jsonl.

  rapids-status runs/<id>            # snapshot
  rapids-status runs/<id> --watch    # refresh every 2s (use from a second terminal)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STAGES = ["planner", "profiler", "cleaner", "analyst", "critic", "visualizer", "reporter"]


def _ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def describe(rec: dict, elapsed: float) -> str | None:
    """One human-readable line per notable trace event (None = stay quiet)."""
    ev, a = rec["event"], rec.get("agent", "")
    pre = f"[{_ts(elapsed)}]"
    if ev == "run_start":
        return f"{pre} ▶ run started: {rec.get('dataset')}"
    if ev == "stage_start":
        return f"{pre} ▶ {a}"
    if ev == "llm_start":
        return f"{pre}     … {a}: waiting on model"
    if ev == "llm_call":
        flag = "  ⚠ EMPTY reply" if rec.get("empty") else ""
        return (f"{pre}     ← {a}: {rec.get('output_tokens')} tok in {rec.get('seconds')}s "
                f"(finish={rec.get('finish')}){flag}")
    if ev == "exec":
        if rec.get("ok"):
            return f"{pre}     ✓ {a}: attempt {rec.get('attempt')} ran OK"
        last = (rec.get("stderr") or "").strip().splitlines()[-1:] or ["failed"]
        return f"{pre}     ✗ {a}: attempt {rec.get('attempt')} failed: {last[0][:140]}"
    if ev == "agent_done":
        return f"{pre} ✓ {a} done ({rec.get('attempts')} attempt(s))"
    if ev == "agent_failed":
        return f"{pre} ✗ {a} FAILED after retries"
    if ev == "critic_round":
        n = len(rec.get("failed", []))
        return f"{pre} ◆ critic round {rec.get('round')}: {n} finding(s) sent back" if n else f"{pre} ◆ critic round {rec.get('round')}: all findings verified"
    if ev == "run_done":
        return f"{pre} ■ run complete"
    return None


def load_trace(run_dir: Path) -> list[dict]:
    f = run_dir / "trace.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def status_table(run_dir: Path) -> str:
    recs = load_trace(run_dir)
    if not recs:
        return f"No trace yet in {run_dir}"
    t0 = recs[0]["t"]
    rows = {s: {"state": "pending", "attempts": 0, "tok": 0, "secs": 0.0, "last": ""} for s in STAGES}
    for r in recs:
        a = r.get("agent")
        if a not in rows:
            continue
        row = rows[a]
        if r["event"] == "stage_start":
            row["state"] = "running"
        elif r["event"] == "llm_call":
            row["tok"] += r.get("output_tokens") or 0
            row["secs"] += r.get("seconds") or 0
        elif r["event"] == "exec":
            row["attempts"] = max(row["attempts"], r.get("attempt", 0))
            if not r.get("ok"):
                row["last"] = ((r.get("stderr") or "").strip().splitlines() or [""])[-1][:70]
        elif r["event"] == "agent_done":
            row["state"] = "done"
            row["last"] = ""
        elif r["event"] == "agent_failed":
            row["state"] = "FAILED"
    icon = {"pending": "·", "running": "▶", "done": "✓", "FAILED": "✗"}
    done = any(r["event"] == "run_done" for r in recs)
    end = recs[-1]["t"] if done else time.time()
    out = [f"Run {run_dir.name}   elapsed {_ts(end - t0)}   {'COMPLETE' if done else 'in progress'}",
           f"{"stage":<13}{"state":<9}{'tries':>6}{'tokens':>8}{'model s':>9}  last error"]
    for s, r in rows.items():
        out.append(f"{icon[r["state"]]} {s:<11}{r["state"]:<9}{r['attempts']:>6}{r['tok']:>8}{r['secs']:>9.0f}  {r['last']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rapids-status")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args(argv)
    if not args.watch:
        print(status_table(args.run_dir))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + status_table(args.run_dir), flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
