"""
Deterministic fraud scoring heuristics for review moderation.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any

ReviewRecord = dict[str, Any]

_GENERIC_PHRASES = {
    "great product",
    "nice product",
    "good product",
    "awesome",
    "excellent",
    "very good",
    "love it",
    "bad product",
}


def _as_rating(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_iso_day(date_text: Any) -> str | None:
    if not isinstance(date_text, str) or not date_text.strip():
        return None
    try:
        return datetime.fromisoformat(date_text.strip()).date().isoformat()
    except ValueError:
        return None


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def detect_review_fraud(reviews: list[ReviewRecord]) -> list[ReviewRecord]:
    """
    Score each review for fraud likelihood with deterministic local rules.

    Parameters
    ----------
    reviews:
        List of review dictionaries. Expected keys: ``id``, ``text``, ``rating``,
        ``date``, and ``verified``.

    Returns
    -------
    list[dict[str, Any]]
        Per-review fraud rows containing:
        ``id`` (str), ``fraud_score`` (0..1), ``fraud_flags`` (list[str]).

    Raises
    ------
    TypeError
        If ``reviews`` is not a list.
    """
    if not isinstance(reviews, list):
        raise TypeError("reviews must be a list of dict objects")

    normalized_texts: list[str] = [_normalize_text(r.get("text", "")) for r in reviews if isinstance(r, dict)]
    text_freq = Counter(t for t in normalized_texts if t)
    day_freq = Counter(
        d for d in (_parse_iso_day(r.get("date")) for r in reviews if isinstance(r, dict)) if d is not None
    )

    scored: list[ReviewRecord] = []
    for review in reviews:
        if not isinstance(review, dict):
            scored.append({"id": "", "fraud_score": 1.0, "fraud_flags": ["invalid_review_record"]})
            continue

        review_id = str(review.get("id", ""))
        text = str(review.get("text", ""))
        normalized = _normalize_text(text)
        rating = _as_rating(review.get("rating"))
        verified = bool(review.get("verified"))
        review_day = _parse_iso_day(review.get("date"))
        word_count = len(normalized.split()) if normalized else 0

        score = 0.0
        flags: list[str] = []

        if normalized and text_freq[normalized] > 1:
            flags.append("duplicate_review_text")
            score += 0.45

        if normalized in _GENERIC_PHRASES or (word_count <= 4 and len(normalized) <= 28):
            flags.append("very_short_or_generic_review")
            score += 0.25

        if (rating <= 1 or rating >= 5) and word_count <= 5:
            flags.append("extreme_rating_low_content_mismatch")
            score += 0.25

        if review_day is not None and day_freq[review_day] >= 4:
            flags.append("review_burst_activity")
            score += 0.20

        if not verified and (rating <= 1 or rating >= 5):
            flags.append("unverified_extreme_rating")
            score += 0.20

        if not verified and "duplicate_review_text" in flags:
            flags.append("unverified_duplicate_pattern")
            score += 0.10

        scored.append(
            {
                "id": review_id,
                "fraud_score": round(min(1.0, score), 3),
                "fraud_flags": sorted(set(flags)),
            }
        )

    return scored


def detect_fraud_signals(
    reviews: list[ReviewRecord],
    analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build fraud summary report from ``detect_review_fraud`` output.
    """
    scored = detect_review_fraud(reviews)
    suspicious = [r for r in scored if float(r.get("fraud_score", 0.0)) >= 0.5]
    flagged_rows = [
        {
            "review_id": row["id"],
            "fraud_score": row["fraud_score"],
            "reasons": row["fraud_flags"],
        }
        for row in suspicious
        if row.get("fraud_flags")
    ]
    avg = (analysis or {}).get("average_rating")
    avg_fraud = round(mean(float(r.get("fraud_score", 0.0)) for r in scored), 3) if scored else 0.0
    suspicious_ratio = round((len(suspicious) / len(scored)), 3) if scored else 0.0
    return {
        "flagged_review_count": len(flagged_rows),
        "flags": flagged_rows,
        "avg_fraud_score": avg_fraud,
        "suspicious_ratio": suspicious_ratio,
        "context_average_rating": avg,
        "scored_reviews": scored,
    }
