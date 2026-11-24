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
        user = f"Question: {question}\n\nDataset preview (header + first rows):\n{columns_hint}"
        text = self.llm.complete(self.name, load_prompt("planner"), user)
        plan = Plan.model_validate({**extract_json(text), "question": question})
        self.ws.write_json("plan.json", plan.model_dump())
        return plan
