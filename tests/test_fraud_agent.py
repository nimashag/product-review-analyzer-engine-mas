"""
Focused evaluation script for Member 2 fraud component.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from tools.fraud_heuristics import detect_review_fraud


def _row(rows: list[dict[str, Any]], review_id: str) -> dict[str, Any]:
    for r in rows:
        if r.get("id") == review_id:
            return r
    raise AssertionError(f"missing scored row for id={review_id!r}")


def test_duplicate_reviews_raise_score() -> None:
    reviews = [
        {"id": "a", "text": "Great product!!!", "rating": 5, "date": "2026-03-10", "verified": False},
        {"id": "b", "text": "Great product!!!", "rating": 5, "date": "2026-03-10", "verified": False},
    ]
    scored = detect_review_fraud(reviews)
    assert _row(scored, "a")["fraud_score"] >= 0.6
    assert "duplicate_review_text" in _row(scored, "a")["fraud_flags"]


def test_generic_short_extreme_review_is_suspicious() -> None:
    reviews = [{"id": "g1", "text": "Awesome!", "rating": 5, "date": "2026-03-11", "verified": False}]
    scored = detect_review_fraud(reviews)
    row = _row(scored, "g1")
    assert row["fraud_score"] >= 0.45
    assert "very_short_or_generic_review" in row["fraud_flags"]
    assert "unverified_extreme_rating" in row["fraud_flags"]


def test_genuine_detailed_review_stays_low() -> None:
    reviews = [
        {
            "id": "real1",
            "text": "Packaging was solid, setup took 12 minutes, and battery lasted all week.",
            "rating": 4,
            "date": "2026-03-12",
            "verified": True,
        }
    ]
    scored = detect_review_fraud(reviews)
    assert _row(scored, "real1")["fraud_score"] <= 0.2


def test_burst_activity_marks_multiple_reviews() -> None:
    reviews = [
        {"id": "u1", "text": "Nice product", "rating": 5, "date": "2026-03-13", "verified": False},
        {"id": "u2", "text": "Nice product", "rating": 5, "date": "2026-03-13", "verified": False},
        {"id": "u3", "text": "Nice product", "rating": 5, "date": "2026-03-13", "verified": False},
        {"id": "u4", "text": "Nice product", "rating": 5, "date": "2026-03-13", "verified": False},
    ]
    scored = detect_review_fraud(reviews)
    assert all("review_burst_activity" in r["fraud_flags"] for r in scored)
    assert mean(float(r["fraud_score"]) for r in scored) >= 0.7


def test_invalid_input_raises_type_error() -> None:
    try:
        detect_review_fraud("bad-input")  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("expected TypeError for non-list input")
