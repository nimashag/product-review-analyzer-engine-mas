"""Unified integration harness for end-to-end pipeline validation."""

from __future__ import annotations

import pytest

crewai = pytest.importorskip("crewai")

from coordinator import PipelineCoordinator, verify_ollama_running


@pytest.mark.integration
def test_system_end_to_end_pipeline() -> None:
    try:
        verify_ollama_running()
    except RuntimeError:
        pytest.skip("Ollama not reachable; start Ollama and pull the configured model.")

    coordinator = PipelineCoordinator()
    state, _ = coordinator.run(
        "sample_reviews.json",
        run_llm_summary=True,
        parallel_analysis_fraud=False,
        assignment_mode=True,
    )

    assert state["reviews"], "reviews should be loaded"
    assert state["analyzed"], "analysis should be populated"
    assert state["scored"], "fraud scoring should be populated"
    assert state["report"], "final report should exist"
    assert all("fraud_score" in row for row in state["scored"])
    assert state["report"]["decision"] in {"BUY", "AVOID", "CONSIDER"}
    tools = {tc.get("tool") for tc in (state.get("tool_calls") or []) if isinstance(tc, dict)}
    assert {"load_reviews", "statistical_analysis", "fraud_detection", "build_recommendations"}.issubset(tools)


def test_strict_merge_discards_mismatched_crew_output() -> None:
    coordinator = PipelineCoordinator()
    pipeline_report = {
        "decision": "CONSIDER",
        "confidence": 0.67,
        "pros": ["acceptable_rating_signal"],
        "cons": ["quality_or_trust_uncertainty"],
        "red_flags": [],
        "summary": "Moderate quality after fraud adjustment; consider with caution.",
    }
    crew_raw = (
        '{"decision":"BUY","confidence":0.99,"pros":["x"],'
        '"cons":[],"red_flags":[],"summary":"mismatch"}'
    )
    merged = coordinator.merge_pipeline_and_crew(
        pipeline_report=pipeline_report,
        crew_raw=crew_raw,
        strict_mode=True,
    )
    assert merged == pipeline_report
