"""Single-shot baseline: one prompt, one script, same LLM, same executor and retries as the crew's agents.

Isolates what the crew's extra stages (plan, profile, clean, critique, gate) actually buy.
"""
from __future__ import annotations

from pathlib import Path

from rapids_analyst.agents.base import CodeAgent
from rapids_analyst.llm import LLM
from rapids_analyst.workspace import Workspace


class Baseline(CodeAgent):
    name = "baseline"

    def __init__(self, llm, ws, ledger, question: str, preview: str, schema: str):
        super().__init__(llm, ws, ledger)
        self.question, self.preview, self.schema = question, preview, schema

    def required_outputs(self):
        return ["findings.json"]

    def validate(self):
        try:
            f = self.ws.read_json("findings.json")
        except ValueError as e:
            return str(e)[:300]
        ok = isinstance(f, list) and f and all(isinstance(x, dict) and "claim" in x and "value" in x for x in f)
        return None if ok else 'findings.json must be a non-empty list of {"claim": str, "value": float}'

    def task(self, ctx):
        return (f"Question: {self.question}\n\nDataset file: {self.dataset_rel()}\n"
                f"Schema:\n{self.schema}\n\nFirst rows (CSV):\n{self.preview}\n\nWrite findings.json to WS.")


def run_baseline(dataset: Path, question: str, out_dir: Path, model: str | None = None, **llm_kw) -> Workspace:
    import pyarrow.parquet as pq
    from rapids_analyst.orchestrator import _preview

    ws = Workspace.create(out_dir, dataset)
    llm = LLM(ws, model, **llm_kw)
    ws.trace("run_start", question=question, dataset=dataset.name, system="baseline")
    schema = str(pq.ParquetFile(ws.dataset_path()).schema_arrow)
    ok = Baseline(llm, ws, [], question, _preview(ws.dataset_path()), schema).run()
    if ok:
        ws.trace("run_done")
    return ws
