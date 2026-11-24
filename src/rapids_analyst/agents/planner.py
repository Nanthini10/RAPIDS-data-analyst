from __future__ import annotations

from ..llm import LLM, extract_json
from ..schemas import Plan
from ..workspace import Workspace
from .base import load_prompt


class Planner:
    name = "planner"

    def __init__(self, llm: LLM, ws: Workspace):
        self.llm, self.ws = llm, ws

    def run(self, question: str, columns_hint: str) -> Plan:
        self.ws.trace("stage_start", agent=self.name)
        user = f"Question: {question}\n\nDataset preview (header + first rows):\n{columns_hint}"
        for attempt in (1, 2):
            text = self.llm.complete(self.name, load_prompt("planner"), user)
            try:
                plan = Plan.model_validate({**extract_json(text), "question": question})
                break
            except Exception as e:  # empty reply or malformed JSON
                self.ws.trace("exec", agent=self.name, attempt=attempt, ok=False, stderr=f"bad plan JSON: {e}")
                if attempt == 2:
                    raise
        self.ws.write_json("plan.json", plan.model_dump())
        self.ws.trace("agent_done", agent=self.name, attempts=1)
        return plan
