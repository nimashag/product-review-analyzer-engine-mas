"""Agent definitions for the Product Review Analysis MAS."""

from agents.analysis_agent import build_analysis_agent
from agents.fraud_agent import build_fraud_agent
from agents.recommendation_agent import build_recommendation_agent
from agents.scraper_agent import build_scraper_agent

__all__ = [
    "build_scraper_agent",
    "build_analysis_agent",
    "build_fraud_agent",
    "build_recommendation_agent",
]
