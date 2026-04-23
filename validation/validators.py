"""Stage validators and safe fallback helpers for deterministic pipeline outputs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from validation.schemas import Stage1Output, Stage2Output, Stage3Output, Stage4Output


def _as_dict(payload: Any, stage: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{stage} output must be a JSON object")
    return payload


def validate_stage1(output: Any, reviews: list[dict[str, Any]]) -> Stage1Output:
    data = _as_dict(output, "stage1")
    model = Stage1Output.model_validate(data)
    if model.review_count != len(reviews):
        raise ValueError("stage1 review_count does not match reviews length")
    if len(model.review_ids) != len(reviews):
        raise ValueError("stage1 review_ids length does not match reviews length")
    expected_ids = [str(r.get("id", "")) for r in reviews]
    if model.review_ids != expected_ids:
        raise ValueError("stage1 review_ids order/content mismatch")
    return model


def validate_stage2(output: Any, reviews: list[dict[str, Any]]) -> Stage2Output:
    data = _as_dict(output, "stage2")
    model = Stage2Output.model_validate(data)
    if model.review_count != len(reviews):
        raise ValueError("stage2 review_count does not match reviews length")
    return model


def validate_stage3(output: Any, reviews: list[dict[str, Any]]) -> Stage3Output:
    data = _as_dict(output, "stage3")
    model = Stage3Output.model_validate(data)
    if model.flagged_review_count != len(model.flags):
        raise ValueError("stage3 flagged_review_count does not match flags length")
    valid_ids = {str(r.get("id", "")) for r in reviews}
    if any(flag.review_id not in valid_ids for flag in model.flags):
        raise ValueError("stage3 flags contains unknown review_id")
    return model


def validate_stage4(output: Any) -> Stage4Output:
    data = _as_dict(output, "stage4")
    model = Stage4Output.model_validate(data)
    if not 0.0 <= model.confidence <= 1.0:
        raise ValueError("stage4 confidence must be between 0.0 and 1.0")
    if not model.summary.strip():
        raise ValueError("stage4 summary cannot be empty")
    return model


def as_validation_error_text(exc: Exception) -> str:
    """Normalize validation exceptions for structured logging."""
    if isinstance(exc, ValidationError):
        return exc.json()
    return str(exc)

