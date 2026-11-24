"""Thin LLM wrapper: text in, text out, with tracing and JSON/code extraction.

Providers (env RAPIDS_ANALYST_PROVIDER):
  nvidia    (default) NVIDIA NIM / build.nvidia.com via the OpenAI-compatible API. Key: NVIDIA_API_KEY
  openai    any OpenAI-compatible endpoint. Set OPENAI_API_KEY and RAPIDS_ANALYST_BASE_URL
  anthropic Claude. Key: ANTHROPIC_API_KEY   (pip install anthropic)

Model: RAPIDS_ANALYST_MODEL or --model.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .workspace import Workspace

PROVIDER = os.environ.get("RAPIDS_ANALYST_PROVIDER", "nvidia").lower()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = {
    "nvidia": "nvidia/nemotron-3.5-super-vl-120b-a12b",  # confirm the exact id on build.nvidia.com
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-5-5",
}
DEFAULT_MODEL = os.environ.get("RAPIDS_ANALYST_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))

# Reasoning models can burn the whole token budget thinking. Default: thinking off.
THINKING = os.environ.get("RAPIDS_ANALYST_THINKING", "0") == "1"
MAX_TOKENS = int(os.environ.get("RAPIDS_ANALYST_MAX_TOKENS", "16000"))

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLM:
    def __init__(self, ws: Workspace, model: str = DEFAULT_MODEL, provider: str = PROVIDER):
        self.ws, self.model, self.provider = ws, model, provider
        if provider in ("nvidia", "openai"):
            from openai import OpenAI  # lazy import so offline tests need no SDK/key

            if provider == "nvidia":
                key = os.environ.get("NVIDIA_API_KEY")
                base = os.environ.get("RAPIDS_ANALYST_BASE_URL", NVIDIA_BASE_URL)
            else:
                key = os.environ.get("OPENAI_API_KEY")
                base = os.environ.get("RAPIDS_ANALYST_BASE_URL")
            if not key:
                raise RuntimeError(f"API key env var not set for provider '{provider}'")
            self.client = OpenAI(api_key=key, base_url=base, timeout=300)
        elif provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, agent: str, system: str, user: str, max_tokens: int = 4096) -> str:
        max_tokens = max(max_tokens, MAX_TOKENS)
        self.last_finish = None
        self.ws.trace("llm_start", agent=agent)
        t0 = time.time()
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=0.2,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body={"chat_template_kwargs": {"enable_thinking": THINKING}},
            )
            msg = resp.choices[0].message
            self.last_finish = resp.choices[0].finish_reason
            text = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            tin = getattr(resp.usage, "prompt_tokens", None)
            tout = getattr(resp.usage, "completion_tokens", None)
        text = _THINK_RE.sub("", text).strip()  # drop reasoning traces if the model emits them
        self.ws.trace("llm_call", agent=agent, provider=self.provider, model=self.model,
                      input_tokens=tin, output_tokens=tout, seconds=round(time.time() - t0, 1),
                      finish=self.last_finish, empty=not text, reasoning_chars=len(locals().get("reasoning", "")))
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
