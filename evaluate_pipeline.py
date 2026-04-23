"""Assignment evaluation runner for deterministic MAS stage validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from coordinator import PipelineCoordinator, configure_logging
from validation.validators import validate_stage1, validate_stage2, validate_stage3, validate_stage4

logger = logging.getLogger("mas.evaluate")


def _log_result(stage: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    logger.info("[%s] %s: %s", status, stage, detail)


def _parse_json_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("review file must contain a JSON array")
    rows = [row for row in payload if isinstance(row, dict)]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run assignment evaluation checks.")
    parser.add_argument("--review-file", default="sample_reviews.json", help="Path to review JSON array.")
    args = parser.parse_args()

    configure_logging(log_file="logs/evaluation.log")
    review_path = Path(args.review_file)
    reviews = _parse_json_file(review_path)

    coordinator = PipelineCoordinator()
    state, output_json = coordinator.run(
        str(review_path),
        run_llm_summary=True,
        strict_mode=True,
        assignment_mode=True,
    )
    report = json.loads(output_json)

    stage1 = {"review_count": len(reviews), "review_ids": [str(r.get("id", "")) for r in reviews]}
    stage_outputs = state.get("stage_outputs") or {}
    stage2 = stage_outputs.get("stage2")
    stage3 = stage_outputs.get("stage3")
    if stage2 is None or stage3 is None:
        raise AssertionError("stage_outputs missing stage2/stage3 outputs")

    checks: list[tuple[str, bool, str]] = []
    try:
        validate_stage1(stage1, reviews)
        checks.append(("stage1_schema_and_correctness", True, "count and id extraction valid"))
    except Exception as exc:
        checks.append(("stage1_schema_and_correctness", False, str(exc)))

    try:
        validate_stage2(stage2, reviews)
        expected_count = len(reviews)
        if stage2["review_count"] != expected_count:
            raise AssertionError("stage2 review_count mismatch")
        if stage2["average_rating"] != (state.get("authoritative", {}).get("stage2", {}).get("average_rating")):
            raise AssertionError("stage2 average_rating mismatch with authoritative")
        if stage_outputs.get("stage2") is None:
            raise AssertionError("state stage_outputs stage2 must not be None")
        checks.append(("stage2_schema_and_correctness", True, "statistical fields valid"))
    except Exception as exc:
        checks.append(("stage2_schema_and_correctness", False, str(exc)))

    try:
        validate_stage3(stage3, reviews)
        checks.append(("stage3_schema_and_correctness", True, "fraud flags format valid"))
    except Exception as exc:
        checks.append(("stage3_schema_and_correctness", False, str(exc)))

    try:
        validate_stage4(report)
        checks.append(("stage4_schema_and_correctness", True, "decision payload valid"))
    except Exception as exc:
        checks.append(("stage4_schema_and_correctness", False, str(exc)))

    tool_calls = state.get("tool_calls") or []
    expected_tools = {"load_reviews", "statistical_analysis", "fraud_detection", "build_recommendations"}
    seen = {tc.get("tool") for tc in tool_calls if isinstance(tc, dict)}
    tools_ok = expected_tools.issubset(seen)
    checks.append(
        (
            "agent_tool_usage_recorded",
            tools_ok,
            f"tool_calls={sorted(seen)}" if tools_ok else f"missing tools; saw={sorted(seen)}",
        ),
    )

    hallucination_found = any(
        token in output_json for token in ("```", "int(", "float(", "str(")
    )
    checks.append(
        (
            "no_hallucinated_format_noise",
            not hallucination_found,
            "no markdown/wrappers detected" if not hallucination_found else "invalid wrapper or markdown detected",
        ),
    )

    failures = 0
    for stage, ok, detail in checks:
        _log_result(stage, ok, detail)
        failures += 0 if ok else 1

    overall = "PASS" if failures == 0 else "FAIL"
    logger.info("Overall: %s (%d/%d checks)", overall, len(checks) - failures, len(checks))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
