"""Pipeline: Planner -> Profiler -> Cleaner -> Analyst <-> Critic (bounded) -> Visualizer -> Reporter."""
from __future__ import annotations

from pathlib import Path

from .agents import Analyst, Cleaner, Critic, Planner, Profiler, Reporter, Visualizer
from .llm import LLM
from .schemas import Critique
from .workspace import Workspace

MAX_REVISIONS = 2
ALPHA = 0.05
MIN_GROUP_N = 30


class PipelineError(RuntimeError):
    def __init__(self, msg: str, run_dir: Path | None = None):
        super().__init__(msg)
        self.run_dir = run_dir


def _preview(path: Path, n: int = 6) -> str:
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        with open(path, errors="replace") as f:
            return "".join(f.readline() for _ in range(n))
    import pandas as pd  # parquet etc.

    return pd.read_parquet(path).head(n).to_csv(index=False)


def stat_gate(finding: dict, c: Critique) -> str | None:
    """Deterministic statistical rule, applied in code so it cannot be talked out of by the LLM.

    Returns a failure reason, or None if the finding clears the gate.
    """
    kind = finding.get("kind", "effect")
    if kind == "descriptive":
        return None
    if c.min_group_n is not None and c.min_group_n < MIN_GROUP_N:
        return f"smallest group has n={c.min_group_n} (< {MIN_GROUP_N}); too small to support a comparison"
    if c.p_value is None:
        return "no significance test was computed for a comparative claim"
    if kind == "effect" and c.p_value > ALPHA:
        return (f"claims an effect but the omnibus test gives p={c.p_value:.3g} (> {ALPHA}). "
                "Either reframe it as a null result (kind='null'; say no reliable difference was found) or drop it. "
                "Do not cherry-pick the best and worst group.")
    if kind == "null" and c.p_value <= ALPHA:
        return f"claims no effect but the test gives p={c.p_value:.3g} (<= {ALPHA}); an effect is present"
    return None


def apply_critique(findings: list[dict], critiques: list[Critique]) -> list[dict]:
    """Mark findings verified only if the Critic passes them AND they clear the statistical gate."""
    by_id = {c.id: c for c in critiques}
    failed = []
    for f in findings:
        c = by_id.get(f["id"])
        reason = None
        if c is None:
            reason = "not audited"
        elif c.verdict != "pass":
            reason = c.reason
        else:
            reason = stat_gate(f, c)
        if reason is None:
            f["status"] = "verified"
        else:
            f["status"] = "pending"
            failed.append({"finding": f, "critic_reason": reason})
    return failed


def merge_revisions(findings: list[dict], revised: list[dict], failed_ids: set[str]) -> list[dict]:
    revised_by_id = {r["id"]: r for r in revised}
    out = []
    for f in findings:
        if f["id"] in failed_ids:
            if f["id"] in revised_by_id:
                out.append(revised_by_id[f["id"]])
            # else: Analyst dropped it as unsupported
        else:
            out.append(f)
    return out


def run_pipeline(dataset: Path, question: str, out_dir: Path, model: str | None = None) -> Workspace:
    ws = Workspace.create(out_dir, dataset)
    llm = LLM(ws, model) if model else LLM(ws)
    ledger: list[Path] = []
    ws.trace("run_start", question=question, dataset=dataset.name)
    print(f"        run dir: {ws.root}\n        watch:   rapids-status {ws.root} --watch", file=__import__("sys").stderr, flush=True)

    Planner(llm, ws).run(question, _preview(ws.dataset_path()))
    for cls in (Profiler, Cleaner):
        if not cls(llm, ws, ledger).run():
            raise PipelineError(f"{cls.name} failed after retries; see {ws.path('trace.jsonl')}", ws.root)

    if not Analyst(llm, ws, ledger).run():
        raise PipelineError("analyst failed after retries", ws.root)
    findings = ws.read_json("findings.json")

    for round_no in range(MAX_REVISIONS + 1):
        ws.write_json("findings.json", findings)
        if not Critic(llm, ws, ledger).run():
            raise PipelineError("critic failed after retries", ws.root)
        critiques = [Critique(**c) for c in ws.read_json("critique.json")]
        failed = apply_critique(findings, critiques)
        ws.trace("critic_round", round=round_no, failed=[x["finding"]["id"] for x in failed])
        if not failed:
            break
        if round_no == MAX_REVISIONS:
            for f in findings:
                if f["status"] == "pending":
                    f["status"] = "unverified"
            break
        revise = Analyst(llm, ws, ledger, output_file="findings_revised.json")
        if not revise.run({"revise": failed}):
            for x in failed:
                x["finding"]["status"] = "unverified"
            break
        findings = merge_revisions(
            findings, ws.read_json("findings_revised.json"), {x["finding"]["id"] for x in failed}
        )

    ws.write_json("findings.json", findings)
    if any(f["status"] == "verified" for f in findings):
        if Visualizer(llm, ws, ledger).run():
            figs = ws.read_json("figure_index.json")
            for f in findings:
                f["figure_ref"] = figs.get(f["id"])
            ws.write_json("findings.json", findings)

    Reporter(llm, ws).run()
    ws.export_analysis_py(ledger)
    ws.trace("run_done")
    return ws
