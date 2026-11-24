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
    # effect: asserts a difference/association exists; null: asserts there is none; descriptive: just reports numbers
    kind: Literal["effect", "null", "descriptive"] = "effect"
    figure_ref: Optional[str] = None
    status: Literal["pending", "verified", "unverified"] = "pending"


class Critique(BaseModel):
    id: str
    verdict: Literal["pass", "fail"]
    reason: str
    recomputed_value: Optional[float] = None
    p_value: Optional[float] = None       # Critic's own significance test (omnibus for >2 groups)
    min_group_n: Optional[int] = None     # size of the smallest group in the comparison
