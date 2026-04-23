"""
Fraud detection agent for the Product Review Analysis MAS (local Ollama; reasoning only, no tools).
"""

from __future__ import annotations

import logging
from typing import Any

from crewai import Agent

logger = logging.getLogger(__name__)


def build_fraud_agent(
    *,
    llm: Any | None = None,
) -> Agent:
    """
    Fraud Pattern Inspector — interprets precomputed fraud signals from the coordinator.

    Parameters
    ----------
    llm:
        Local Ollama-backed CrewAI LLM string (text generation only; no tool calling).
    """
    logger.info("Building agent: Fraud Pattern Inspector (reasoning-only)")

    return Agent(
        role="Fraud Pattern Inspector",
        goal=(
            "Given authoritative fraud JSON from the coordinator, output only strict JSON "
            "with flagged_review_count, suspicious_ratio, flags, and summary consistent with that data."
        ),
        backstory=(
            "You specialize in explaining heuristic fraud signals already computed upstream. "
            "You do not run tools or rescore reviews; you read the PRECOMPUTED summary and emit one JSON object."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )


__all__ = ["build_fraud_agent"]
