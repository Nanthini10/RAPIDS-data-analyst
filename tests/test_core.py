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
