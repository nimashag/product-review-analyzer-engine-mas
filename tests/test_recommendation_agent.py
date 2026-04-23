"""Evaluation tests for recommendation aggregation and final decision rules."""

from __future__ import annotations

from tools.recommendation_rules import build_recommendations, generate_recommendation_summary


def test_generate_recommendation_summary_positive_low_fraud() -> None:
    data = [
        {"id": "1", "text": "great quality and durable design", "rating": 5, "fraud_score": 0.1, "fraud_flags": []},
        {"id": "2", "text": "excellent and easy to use", "rating": 5, "fraud_score": 0.0, "fraud_flags": []},
        {"id": "3", "text": "love it, fast shipping", "rating": 4, "fraud_score": 0.2, "fraud_flags": []},
    ]
    summary = generate_recommendation_summary(data, average_rating=4.67)
    assert summary["weighted_score"] > 0.7
    assert summary["stats"]["fraud_ratio"] < 0.3
    assert isinstance(summary["top_pros"], list)


def test_recommendation_buy_case() -> None:
    scored = [
        {"id": "1", "text": "great quality and durable design", "rating": 5, "fraud_score": 0.1, "fraud_flags": []},
        {"id": "2", "text": "excellent and easy to use", "rating": 5, "fraud_score": 0.0, "fraud_flags": []},
        {"id": "3", "text": "love it, fast shipping", "rating": 4, "fraud_score": 0.2, "fraud_flags": []},
    ]
    result = build_recommendations(scored, {"average_rating": 4.67}, None)
    assert result["decision"] == "BUY"
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["pros"]) >= 1
    assert isinstance(result["cons"], list)


def test_recommendation_consider_case() -> None:
    scored = [
        {"id": "1", "text": "good value but late shipping", "rating": 3, "fraud_score": 0.2, "fraud_flags": []},
        {"id": "2", "text": "easy setup but poor packaging", "rating": 3, "fraud_score": 0.3, "fraud_flags": []},
        {"id": "3", "text": "decent quality, not amazing", "rating": 3, "fraud_score": 0.1, "fraud_flags": []},
    ]
    result = build_recommendations(scored, {"average_rating": 3.0}, None)
    assert result["decision"] == "CONSIDER"
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["pros"], list)
    assert isinstance(result["cons"], list)


def test_recommendation_avoid_negative_high_fraud() -> None:
    scored = [
        {
            "id": "1",
            "text": "terrible product and broken within days",
            "rating": 1,
            "fraud_score": 0.7,
            "fraud_flags": ["duplicate_review_text"],
        },
        {
            "id": "2",
            "text": "awful support, refund requested, scam",
            "rating": 1,
            "fraud_score": 0.8,
            "fraud_flags": ["unverified_extreme_rating"],
        },
        {
            "id": "3",
            "text": "bad quality and very poor durability",
            "rating": 2,
            "fraud_score": 0.6,
            "fraud_flags": ["very_short_or_generic_review"],
        },
    ]
    result = build_recommendations(scored, {"average_rating": 1.33}, None)
    assert result["decision"] == "AVOID"
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["cons"]) >= 1
    assert len(result["red_flags"]) >= 1
