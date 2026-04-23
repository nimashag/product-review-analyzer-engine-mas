"""Pydantic output schemas for all MAS stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Stage1Output(BaseModel):
    """Schema for Review Data Collector output."""

    model_config = ConfigDict(extra="forbid")

    review_count: int
    review_ids: list[str]


class Stage2Output(BaseModel):
    """Schema for Statistical Review Analyst output."""

    model_config = ConfigDict(extra="forbid")

    review_count: int
    average_rating: float
    rating_distribution: dict[str, int]
    verified_ratio: float
    keyword_negative_hits: int
    keyword_positive_hits: int


class Stage3Flag(BaseModel):
    """Schema for an individual fraud flag object."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    fraud_score: float
    reasons: list[str]


class Stage3Output(BaseModel):
    """Schema for Fraud Pattern Inspector output."""

    model_config = ConfigDict(extra="forbid")

    flagged_review_count: int
    suspicious_ratio: float
    flags: list[Stage3Flag]
    summary: str


class Stage4Output(BaseModel):
    """Schema for Product Recommendation Strategist output."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["BUY", "CONSIDER", "AVOID"]
    confidence: float
    pros: list[str]
    cons: list[str]
    red_flags: list[str]
    summary: str

