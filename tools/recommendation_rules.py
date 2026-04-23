"""Local recommendation aggregation and decision rules (deterministic, no APIs)."""

from __future__ import annotations

from typing import Any, Dict, List

ReviewRecord = dict[str, Any]

_POSITIVE_TERMS = (
    "great",
    "excellent",
    "amazing",
    "love",
    "perfect",
    "fast",
    "durable",
    "quality",
    "easy",
    "recommend",
)
_NEGATIVE_TERMS = (
    "terrible",
    "awful",
    "worst",
    "refund",
    "return",
    "broken",
    "late",
    "bad",
    "scam",
    "poor",
)
_RED_FLAG_TERMS = ("fake", "scam", "counterfeit", "duplicate", "fraud", "misleading", "broken")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_sentiment_score(row: Dict[str, Any]) -> float:
    sentiment = row.get("sentiment")
    if isinstance(sentiment, (int, float)):
        return max(-1.0, min(1.0, float(sentiment)))
    rating = _to_float(row.get("rating"), default=3.0)
    return max(-1.0, min(1.0, (rating - 3.0) / 2.0))


def _fraud_weight(fraud_score: float) -> float:
    if fraud_score >= 0.8:
        return 0.1
    if fraud_score >= 0.6:
        return 0.25
    if fraud_score >= 0.4:
        return 0.5
    return 1.0


def generate_recommendation_summary(data: List[Dict], average_rating: float) -> Dict:
    """
    Aggregate fraud-aware review evidence for recommendation decisions.

    Parameters
    ----------
    data:
        List of scored review dictionaries with ``fraud_score`` and optional ``fraud_flags``.
    average_rating:
        Aggregate rating (1..5) from upstream analysis stage.

    Returns
    -------
    Dict
        Deterministic summary payload with weighted sentiment, top pros/cons, red flags, and stats.

    Raises
    ------
    TypeError
        If ``data`` is not a list of dictionaries.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of dict records")
    if any(not isinstance(row, dict) for row in data):
        raise TypeError("each item in data must be a dict")

    total_reviews = len(data)
    if total_reviews == 0:
        return {
            "weighted_score": 0.0,
            "top_pros": [],
            "top_cons": [],
            "red_flags": ["no_reviews_available"],
            "stats": {
                "total_reviews": 0,
                "average_rating": round(average_rating, 3),
                "fraud_ratio": 0.0,
            },
        }

    weighted_sum = 0.0
    weight_total = 0.0
    raw_sentiment_sum = 0.0
    fraud_count = 0
    pros_hits: dict[str, int] = {}
    cons_hits: dict[str, int] = {}
    red_flag_hits: dict[str, int] = {}

    for row in data:
        text = str(row.get("text", "")).lower()
        fraud_score = max(0.0, min(1.0, _to_float(row.get("fraud_score"), 0.0)))
        rating = _to_float(row.get("rating"), average_rating)
        normalized_rating = max(0.0, min(1.0, rating / 5.0))
        weight = _fraud_weight(fraud_score)
        raw_sentiment_sum += _extract_sentiment_score(row)
        weighted_sum += normalized_rating * weight
        weight_total += weight

        if fraud_score >= 0.5:
            fraud_count += 1
        for flag in row.get("fraud_flags", []) if isinstance(row.get("fraud_flags"), list) else []:
            red_flag_hits[str(flag)] = red_flag_hits.get(str(flag), 0) + 1

        for token in _POSITIVE_TERMS:
            if token in text:
                pros_hits[token] = pros_hits.get(token, 0) + 1
        for token in _NEGATIVE_TERMS:
            if token in text:
                cons_hits[token] = cons_hits.get(token, 0) + 1
        for token in _RED_FLAG_TERMS:
            if token in text:
                red_flag_hits[token] = red_flag_hits.get(token, 0) + 1

    weighted_score = round(weighted_sum / weight_total, 3) if weight_total else round(average_rating / 5.0, 3)
    avg_sentiment = round(raw_sentiment_sum / total_reviews, 3)
    fraud_ratio = round(fraud_count / total_reviews, 3)
    fraud_percentage = round(fraud_ratio * 100.0, 1)

    top_pros = [k for k, _ in sorted(pros_hits.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    top_cons = [k for k, _ in sorted(cons_hits.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    red_flags = [k for k, _ in sorted(red_flag_hits.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]

    if fraud_ratio >= 0.4 and "high_fraud_ratio" not in red_flags:
        red_flags.insert(0, "high_fraud_ratio")
    if weighted_score < 0.4 and "low_quality_signal" not in red_flags:
        red_flags.append("low_quality_signal")

    return {
        "weighted_score": weighted_score,
        "top_pros": top_pros,
        "top_cons": top_cons,
        "red_flags": red_flags,
        "stats": {
            "total_reviews": total_reviews,
            "average_rating": round(average_rating, 3),
            "fraud_ratio": fraud_ratio,
            "avg_sentiment": avg_sentiment,
            "fraud_percentage": fraud_percentage,
        },
    }


def build_recommendations(
    reviews: list[ReviewRecord],
    analysis: dict[str, Any] | None,
    fraud_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Produce the final purchase recommendation from pre-scored review evidence.

    This wrapper keeps backward compatibility with existing pipeline calls.
    """
    analysis = analysis or {}
    fraud_report = fraud_report or {}
    average_rating = max(0.0, min(5.0, _to_float(analysis.get("average_rating"), 0.0)))
    summary = generate_recommendation_summary(reviews, average_rating)
    fraud_ratio = _to_float(summary.get("stats", {}).get("fraud_ratio"), _to_float(fraud_report.get("suspicious_ratio"), 0.0))
    score = round((average_rating / 5.0) * (1.0 - fraud_ratio), 3)
    fraud_ratio = _to_float(summary.get("stats", {}).get("fraud_ratio"), 0.0)
    cons = list(summary.get("top_cons", []))
    pros = list(summary.get("top_pros", []))
    red_flags = list(summary.get("red_flags", []))

    if score >= 0.7:
        decision = "BUY"
    elif score >= 0.4:
        decision = "CONSIDER"
    else:
        decision = "AVOID"

    confidence = round(max(0.0, min(1.0, score)), 2)

    if decision == "BUY":
        summary_text = "Strong average rating with low fraud risk supports a purchase."
    elif decision == "AVOID":
        summary_text = "Low quality score after fraud adjustment indicates high purchase risk."
    else:
        summary_text = "Moderate quality after fraud adjustment; consider with caution."

    if not pros:
        pros = ["acceptable_rating_signal"] if average_rating >= 3.0 else []
    if not cons:
        cons = ["quality_or_trust_uncertainty"] if decision != "BUY" else []
    if fraud_ratio >= 0.4 and "high_fraud_ratio" not in red_flags:
        red_flags.insert(0, "high_fraud_ratio")

    return {
        "decision": decision,
        "confidence": confidence,
        "pros": pros,
        "cons": cons,
        "red_flags": red_flags,
        "summary": summary_text,
    }
