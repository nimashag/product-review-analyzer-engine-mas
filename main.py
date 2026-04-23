"""Main entrypoint for orchestrated local CrewAI pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, MutableMapping

from coordinator import PipelineCoordinator, configure_logging

logger = logging.getLogger("mas.main")
_ROOT = Path(__file__).resolve().parent


def run_pipeline(
    review_file: str,
    *,
    run_llm_summary: bool = True,
    strict_mode: bool = True,
    assignment_mode: bool = True,
) -> tuple[MutableMapping[str, Any], str]:
    """Run full orchestrated pipeline."""
    coordinator = PipelineCoordinator()
    return coordinator.run(
        review_file,
        run_llm_summary=run_llm_summary,
        parallel_analysis_fraud=False,
        strict_mode=strict_mode,
        assignment_mode=assignment_mode,
    )


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run local product review MAS pipeline.")
    parser.add_argument("--review-file", default="sample_reviews.json", help="Path to local reviews JSON file.")
    parser.add_argument("--product", default="Product", help="Product name for demo output header.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run tools-only path without Crew LLM (for headless regression; not assignment submission mode).",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow valid crew JSON override even if it differs from pipeline output.",
    )
    parser.add_argument(
        "--no-assignment-mode",
        action="store_true",
        help="Disable assignment mode safeguards (not recommended for submission).",
    )
    args = parser.parse_args()

    review_file = args.review_file
    logger.info("[MAS][Ollama][Pipeline] main() — review_file=%r", review_file)

    try:
        # Deterministic/tools-only runs should not force assignment safeguards that re-enable the LLM crew.
        assignment_mode = not args.no_assignment_mode and not args.deterministic
        state, output = run_pipeline(
            review_file,
            run_llm_summary=not args.deterministic,
            strict_mode=not args.non_strict,
            assignment_mode=assignment_mode,
        )
    except RuntimeError as exc:
        logger.error("[MAS][Ollama][Pipeline] %s", exc)
        return 1

    snapshot_path = _ROOT / "last_pipeline_state.json"
    PipelineCoordinator.persist_execution_snapshot(dict(state), snapshot_path)
    logger.info("[MAS][Ollama][Pipeline] Wrote full execution snapshot to %s", snapshot_path)
    logger.info("[MAS][Ollama][Pipeline] Final JSON output length=%d", len(output))

    if state.get("report"):
        logger.info("[MAS][Ollama][Pipeline] Final recommendation state['report']=%s", state["report"])
        report = state.get("report", {})
        logger.info("=== FINAL RECOMMENDATION ===")
        logger.info("Product: %s", args.product)
        logger.info("Decision: %s", report.get("decision"))
        logger.info("Confidence: %s", report.get("confidence"))
        logger.info("Pros: %s", ", ".join(report.get("pros", [])))
        logger.info("Cons: %s", ", ".join(report.get("cons", [])))
        logger.info("Red Flags: %s", ", ".join(report.get("red_flags", [])))
        logger.info("Summary: %s", report.get("summary"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
