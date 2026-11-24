"""Headless smoke tests for the Streamlit dashboard (no model calls, no API key)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "rapids_analyst" / "dashboard.py")


def _fake_run(base: Path) -> Path:
    root = base / "20260101-000000-abc123"
    (root / "figures").mkdir(parents=True)
    t = 1_000_000.0
    events = [
        {"t": t, "event": "run_start", "dataset": "d.csv"},
        {"t": t + 1, "event": "stage_start", "agent": "planner"},
        {"t": t + 5, "event": "llm_call", "agent": "planner", "output_tokens": 200, "seconds": 4.0, "finish": "stop", "empty": False},
        {"t": t + 6, "event": "agent_done", "agent": "planner", "attempts": 1},
        {"t": t + 9, "event": "run_done"},
    ]
    (root / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (root / "findings.json").write_text(json.dumps([
        {"id": "f1", "claim": "Contract drives churn (0.330, n=10913)", "value": 0.28, "n": 20000, "kind": "effect", "status": "verified"},
        {"id": "f2", "claim": "Region differs", "value": 0.05, "n": 20000, "kind": "effect", "status": "unverified"},
    ]))
    (root / "critique.json").write_text(json.dumps([
        {"id": "f1", "verdict": "pass", "reason": "ok", "p_value": 1e-9, "min_group_n": 4000},
        {"id": "f2", "verdict": "pass", "reason": "arithmetic ok", "p_value": 0.4, "min_group_n": 428},
    ]))
    (root / "report.md").write_text("# Q\n\nAnswer.\n\n![](figures/missing.png)\n\n## Caveats\n- none\n")
    return root


def test_dashboard_renders_without_error(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPIDS_ANALYST_RUNS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    run_btn = next(b for b in at.button if b.label == "Run analysis")
    assert run_btn.disabled  # no key / dataset / question yet


def test_dashboard_opens_a_previous_run(monkeypatch, tmp_path):
    root = _fake_run(tmp_path)
    monkeypatch.setenv("RAPIDS_ANALYST_RUNS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["view_run"] = root.name
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Verified findings"] == "1 / 2"


def test_dashboard_never_stores_key_on_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPIDS_ANALYST_RUNS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.text_input[0].set_value("nvapi-SECRET-123").run()
    assert not any("SECRET-123" in p.read_text(errors="ignore") for p in tmp_path.rglob("*") if p.is_file())
