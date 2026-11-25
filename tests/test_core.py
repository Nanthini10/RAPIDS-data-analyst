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
    findings = [{"id": "f1", "status": "pending", "kind": "descriptive"}, {"id": "f2", "status": "pending", "kind": "descriptive"}]
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


def test_stat_gate_rules():
    from rapids_analyst.orchestrator import stat_gate

    eff = {"kind": "effect"}
    assert stat_gate({"kind": "descriptive"}, Critique(id="a", verdict="pass", reason="")) is None
    assert stat_gate(eff, Critique(id="a", verdict="pass", reason="", p_value=0.001, min_group_n=500)) is None
    assert "p=0.4" in stat_gate(eff, Critique(id="a", verdict="pass", reason="", p_value=0.4, min_group_n=500))
    assert "no significance" in stat_gate(eff, Critique(id="a", verdict="pass", reason=""))
    assert "n=12" in stat_gate(eff, Critique(id="a", verdict="pass", reason="", p_value=0.001, min_group_n=12))
    null = {"kind": "null"}
    assert stat_gate(null, Critique(id="a", verdict="pass", reason="", p_value=0.36, min_group_n=900)) is None
    assert "no effect" in stat_gate(null, Critique(id="a", verdict="pass", reason="", p_value=0.001, min_group_n=900))


def test_apply_critique_enforces_gate_even_when_critic_passes():
    findings = [{"id": "r", "status": "pending", "kind": "effect"}]
    crit = [Critique(id="r", verdict="pass", reason="arithmetic ok", p_value=0.4, min_group_n=428)]
    failed = apply_critique(findings, crit)
    assert findings[0]["status"] == "pending" and "reframe" in failed[0]["critic_reason"]


def test_critic_requires_p_values(tmp_path):
    from rapids_analyst.agents import Critic

    ws = _ws(tmp_path)
    ws.write_json("findings.json", [{"id": "a", "kind": "effect"}, {"id": "b", "kind": "descriptive"}])
    ws.write_json("critique.json", [{"id": "a", "verdict": "pass", "reason": ""}, {"id": "b", "verdict": "pass", "reason": ""}])
    assert "p_value" in Critic(None, ws, []).validate()
    ws.write_json("critique.json", [{"id": "a", "verdict": "pass", "reason": "", "p_value": 0.01, "min_group_n": 50, "effect_size": 0.3},
                                    {"id": "b", "verdict": "pass", "reason": ""}])
    assert Critic(None, ws, []).validate() is None


def test_cpu_shim_makes_cudf_methods_work_on_pandas(tmp_path):
    ws = _ws(tmp_path)
    res = run_python(
        "df = xdf.read_csv(os.path.join(WS, 'input', 'd.csv'))\n"
        "g = df.groupby('a').agg({'b': ['mean', 'count']}).to_pandas()\n"   # cuDF idiom, direct call
        "assert df['a'].to_pandas().sum() == 4 and len(df.to_cupy()) == 2\n"
        "assert xdf.Index([1, 2]).to_pandas() is not None",
        ws.next_step_path("t"), ws.root,
    )
    assert res.ok, res.stderr


def test_cleaner_finalize_normalises_categories_and_writes_schema(tmp_path):
    import pandas as pd
    from rapids_analyst.agents import Cleaner

    ws = _ws(tmp_path)
    df = pd.DataFrame({"contract": pd.Categorical(["Month-to-month", "Two year", "Two year"]), "tenure": [1, 2, 3]})
    df.to_parquet(ws.path("clean.parquet"), index=False)
    Cleaner(None, ws, []).finalize()
    out = pd.read_parquet(ws.path("clean.parquet"))
    assert str(out["contract"].dtype) == "object"
    out["contract"] = out["contract"].fillna("missing")  # would raise on a Categorical
    schema = ws.read_json("clean_schema.json")
    contract = next(c for c in schema["columns"] if c["name"] == "contract")
    assert contract["levels"] == {"Two year": 2, "Month-to-month": 1}
    assert next(c for c in schema["columns"] if c["name"] == "tenure")["max"] == 3.0


def test_prelude_provides_pd_and_np(tmp_path):
    ws = _ws(tmp_path)
    res = run_python("assert pd.__name__ == 'pandas' and np.__name__ == 'numpy'", ws.next_step_path("t"), ws.root)
    assert res.ok, res.stderr


def test_gate_rejects_negligible_effect_size():
    from rapids_analyst.orchestrator import stat_gate

    c = Critique(id="a", verdict="pass", reason="", p_value=0.043, min_group_n=428, effect_size=0.022)
    assert "negligible" in stat_gate({"kind": "effect"}, c)
    c2 = Critique(id="a", verdict="pass", reason="", p_value=0.001, min_group_n=428, effect_size=0.3)
    assert stat_gate({"kind": "effect"}, c2) is None
