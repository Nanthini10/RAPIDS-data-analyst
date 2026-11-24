"""Structured contracts passed between agents (serialised as JSON in the run workspace)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Plan(BaseModel):
    question: str
    target_column: Optional[str] = None
    hypotheses: list[str] = Field(default_factory=list)
    analysis_tasks: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    id: str
    claim: str
    evidence_code: str = Field(description="Self-contained snippet that recomputes `value`")
    metric: str
    value: float
    effect_size: Optional[float] = None
    n: int
    confidence: Literal["low", "medium", "high"] = "medium"
    figure_ref: Optional[str] = None
    status: Literal["pending", "verified", "unverified"] = "pending"


class Critique(BaseModel):
    id: str
    verdict: Literal["pass", "fail"]
    reason: str
    recomputed_value: Optional[float] = None
