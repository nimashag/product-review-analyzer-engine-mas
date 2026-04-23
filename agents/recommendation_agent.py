"""
Recommendation agent for the Product Review Analysis MAS (local Ollama; reasoning only, no tools).
"""

from __future__ import annotations

import logging
from typing import Any

from crewai import Agent

logger = logging.getLogger(__name__)


def build_recommendation_agent(
    *,
    llm: Any | None = None,
) -> Agent:
    """
    Product Recommendation Strategist — interprets the coordinator's recommendation JSON.

    Parameters
    ----------
    llm:
        Local Ollama-backed CrewAI LLM string (text generation only; no tool calling).
    """
    logger.info("Building agent: Product Recommendation Strategist (reasoning-only)")

    return Agent(
        role="Product Recommendation Strategist",
        goal=(
            "Given authoritative recommendation JSON from the coordinator, output only strict JSON "
            "with decision, confidence, pros, cons, red_flags, and summary aligned to that payload."
        ),
        backstory=(
            "You translate deterministic purchase guidance into clear shopper language. "
            "You never invoke tools; you only reconcile your JSON with the PRECOMPUTED recommendation block."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


__all__ = ["build_recommendation_agent"]
