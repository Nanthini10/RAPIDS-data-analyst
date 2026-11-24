"""Thin Claude client wrapper: text in, text out, with tracing and JSON/code extraction."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .workspace import Workspace

DEFAULT_MODEL = os.environ.get("RAPIDS_ANALYST_MODEL", "claude-sonnet-5-5")


class LLM:
    def __init__(self, ws: Workspace, model: str = DEFAULT_MODEL):
        import anthropic  # imported lazily so tests can run without the SDK configured

        self.client = anthropic.Anthropic()
        self.model = model
        self.ws = ws

    def complete(self, agent: str, system: str, user: str, max_tokens: int = 4096) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        self.ws.trace(
            "llm_call",
            agent=agent,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        return text


_CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_JSON_RE = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    return (m.group(1) if m else text).strip()


def extract_json(text: str) -> Any:
    m = _JSON_RE.search(text)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("{"), raw.find("[")) if i != -1], default=0)
    return json.loads(raw[start:])
