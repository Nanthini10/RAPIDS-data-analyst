from __future__ import annotations

import json

from ..llm import LLM
from ..workspace import Workspace
from .base import load_prompt


class Reporter:
    name = "reporter"

    def __init__(self, llm: LLM, ws: Workspace):
        self.llm, self.ws = llm, ws

    def run(self) -> str:
        findings = self.ws.read_json("findings.json")
        verified = [f for f in findings if f["status"] == "verified"]
        unverified = [f for f in findings if f["status"] == "unverified"]
        user = json.dumps(
            {
                "plan": self.ws.read_json("plan.json"),
                "transform_log": self.ws.read_json("transform_log.json"),
                "verified_findings": verified,
                "unverified_findings": [{"id": f["id"], "claim": f["claim"]} for f in unverified],
                "fallbacks": self.ws.read_json("fallbacks.json") if self.ws.exists("fallbacks.json") else [],
            },
            indent=2,
            default=str,
        )
        report = self.llm.complete(self.name, load_prompt("reporter"), user, max_tokens=6000)
        self.ws.path("report.md").write_text(report)
        return report
