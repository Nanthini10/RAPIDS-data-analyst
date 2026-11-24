"""Base class for agents that write + execute Python and self-correct from stderr."""
from __future__ import annotations

import json
from pathlib import Path

from ..executor import run_python
from ..llm import LLM, extract_code
from ..workspace import Workspace

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


class CodeAgent:
    name: str = "agent"
    max_attempts: int = 3
    timeout: int = 300

    def __init__(self, llm: LLM, ws: Workspace, ledger: list[Path]):
        self.llm = llm
        self.ws = ws
        self.ledger = ledger  # successful step scripts, exported as analysis.py

    # -- subclass hooks ------------------------------------------------
    def required_outputs(self) -> list[str]:
        return []

    def task(self, ctx: dict) -> str:
        raise NotImplementedError

    def validate(self) -> str | None:
        """Return an error message if outputs are malformed, else None."""
        return None

    # -- main loop -----------------------------------------------------
    def run(self, ctx: dict | None = None) -> bool:
        ctx = ctx or {}
        self.ws.trace("stage_start", agent=self.name)
        system = load_prompt("common") + "\n\n" + load_prompt(self.name)
        error = ""
        for attempt in range(1, self.max_attempts + 1):
            user = self.task(ctx)
            if error:
                user += f"\n\n## Previous attempt failed\n{error}\nFix the problem and return the full corrected script."
            reply = self.llm.complete(self.name, system, user, max_tokens=6000)
            if not reply.strip():
                error = (f"The model returned no content (finish_reason={self.llm.last_finish}). "
                         "Reply with ONE python code block only; do not think at length.")
                self.ws.trace("exec", agent=self.name, attempt=attempt, ok=False, stderr=error, script="-")
                continue
            code = extract_code(reply)
            res = run_python(code, self.ws.next_step_path(self.name), self.ws.root, self.timeout)
            self.ws.trace(
                "exec", agent=self.name, attempt=attempt, ok=res.ok,
                stderr=res.stderr[-500:], script=res.script.name,
            )
            if res.ok:
                missing = [f for f in self.required_outputs() if not self.ws.exists(f)]
                if missing:
                    error = f"Script ran but did not write: {missing}"
                    continue
                bad = self.validate()
                if bad:
                    error = f"Output validation failed: {bad}"
                    continue
                self.ledger.append(res.script)
                if self.ws.exists("_no_gpu"):
                    self.ws.record_fallback(self.name, "cuDF unavailable; executed on pandas")
                self.ws.trace("agent_done", agent=self.name, attempts=attempt)
                return True
            error = f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
        self.ws.trace("agent_failed", agent=self.name, error=error[-500:])
        return False

    # -- context helpers ----------------------------------------------
    def dataset_rel(self) -> str:
        return str(self.ws.dataset_path().relative_to(self.ws.root))

    @staticmethod
    def dump(obj, limit: int = 6000) -> str:
        s = json.dumps(obj, indent=2, default=str)
        return s if len(s) <= limit else s[:limit] + "\n... [truncated]"
