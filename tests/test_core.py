"""Offline tests: no API key or GPU needed."""
from pathlib import Path

from rapids_analyst.executor import run_python
from rapids_analyst.llm import extract_code, extract_json
from rapids_analyst.orchestrator import apply_critique, merge_revisions
from rapids_analyst.schemas import Critique
from rapids_analyst.workspace import Workspace


def _ws(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("a,b\n1,2\n3,4\n")
    return Workspace.create(tmp_path / "runs", data)


def test_extractors():
    assert extract_code("x\n```python\nprint(1)\n```") == "print(1)"
    assert extract_json('here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('[{"a": 1}]') == [{"a": 1}]


def test_executor_runs_with_prelude_and_fallback(tmp_path):
    ws = _ws(tmp_path)
    res = run_python(
        "df = xdf.read_csv(os.path.join(WS, 'input', 'd.csv'))\n"
        "json.dump({'rows': int(len(df))}, open(os.path.join(WS, 'out.json'), 'w'))",
        ws.next_step_path("t"), ws.root,
    )
    assert res.ok, res.stderr
    assert ws.read_json("out.json") == {"rows": 2}


def test_executor_reports_errors(tmp_path):
    ws = _ws(tmp_path)
    res = run_python("raise ValueError('boom')", ws.next_step_path("t"), ws.root)
    assert not res.ok and "boom" in res.stderr


def test_critic_loop_helpers():
    findings = [{"id": "f1", "status": "pending"}, {"id": "f2", "status": "pending"}]
    crit = [Critique(id="f1", verdict="pass", reason="ok"), Critique(id="f2", verdict="fail", reason="n too small")]
    failed = apply_critique(findings, crit)
    assert findings[0]["status"] == "verified"
    assert [x["finding"]["id"] for x in failed] == ["f2"]
    merged = merge_revisions(findings, [{"id": "f2", "status": "pending", "claim": "fixed"}], {"f2"})
    assert merged[1]["claim"] == "fixed"
    assert merge_revisions(findings, [], {"f2"}) == [findings[0]]


def test_progress_status_table(tmp_path):
    from rapids_analyst.progress import describe, status_table

    ws = _ws(tmp_path)
    ws.verbose = False
    ws.trace("run_start", dataset="d.csv")
    ws.trace("stage_start", agent="profiler")
    ws.trace("llm_call", agent="profiler", output_tokens=900, seconds=12.0, finish="stop", empty=False)
    ws.trace("exec", agent="profiler", attempt=1, ok=False, stderr="Traceback\nKeyError: 'x'")
    table = status_table(ws.root)
    assert "profiler" in table and "running" in table and "KeyError" in table
    assert "EMPTY" in describe({"event": "llm_call", "agent": "a", "empty": True}, 5)


def test_workspace_root_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "d.csv"
    data.write_text("a\n1\n")
    ws = Workspace.create(Path("rel_runs"), data)
    assert ws.root.is_absolute()


def test_to_pd_helper_works_without_gpu(tmp_path):
    ws = _ws(tmp_path)
    res = run_python(
        "df = xdf.read_csv(os.path.join(WS, 'input', 'd.csv'))\n"
        "assert to_pd(df) is df or hasattr(to_pd(df), 'iloc')\n"
        "assert to_pd(df['a']).sum() == 4",
        ws.next_step_path("t"), ws.root,
    )
    assert res.ok, res.stderr


def test_analyst_rejects_too_many_findings(tmp_path):
    from rapids_analyst.agents import Analyst

    ws = _ws(tmp_path)
    f = lambda i: {"id": f"f{i}", "claim": "c", "evidence_code": "x", "metric": "m", "value": 1.0, "n": 50}
    ws.write_json("findings.json", [f(i) for i in range(11)])
    assert "limit" in Analyst(None, ws, []).validate()
    ws.write_json("findings.json", [f(i) for i in range(5)])
    assert Analyst(None, ws, []).validate() is None
    ws.write_json("findings.json", [f(1), f(1)])
    assert "unique" in Analyst(None, ws, []).validate()
