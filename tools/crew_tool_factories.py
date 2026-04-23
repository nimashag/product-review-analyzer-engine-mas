"""
CrewAI ``@tool`` factories for the local-only review pipeline.

Each factory returns a tool bound to the shared ``state`` mapping. Tools perform
real file I/O and deterministic computation (no cloud APIs). Every invocation is
appended to ``state["tool_calls"]`` for observability.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, MutableMapping

from crewai.tools import tool

from tools.fraud_heuristics import detect_review_fraud
from tools.load_reviews import ReviewRecord
from tools.load_reviews import load_reviews as load_reviews_from_json
from tools.recommendation_rules import build_recommendations
from tools.review_analysis import analyze_reviews_local

logger = logging.getLogger(__name__)


def _append_tool_call(
    state: MutableMapping[str, Any],
    *,
    agent_name: str,
    tool_name: str,
    tool_input: str,
    output_preview: str,
    duration_sec: float,
    error: str | None = None,
) -> None:
    """Record a single tool invocation on shared pipeline state."""
    preview = output_preview if len(output_preview) <= 500 else output_preview[:500] + "…"
    record: dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "tool": tool_name,
        "input": tool_input,
        "output_preview": preview,
        "duration_sec": round(duration_sec, 4),
        "error": error,
    }
    state.setdefault("tool_calls", []).append(record)
    logger.info(
        "[MAS][tool_call] agent=%s tool=%s duration_sec=%.4f error=%s input=%s",
        agent_name,
        tool_name,
        duration_sec,
        error,
        tool_input,
    )


def _log_tool_io(
    agent_label: str,
    tool_name: str,
    tool_input: str,
    output_preview: str,
    *,
    output_len: int,
) -> None:
    preview = output_preview if len(output_preview) <= 800 else output_preview[:800] + "…"
    logger.info(
        "[MAS][tool] agent=%s tool=%s input=%s output_chars=%d output_preview=%r",
        agent_label,
        tool_name,
        tool_input,
        output_len,
        preview,
    )


def create_load_reviews_crew_tool(
    state: MutableMapping[str, Any],
    *,
    on_loaded: Callable[[list[ReviewRecord]], None] | None = None,
) -> Any:
    """
    Build the Review Data Collector tool: ``load_reviews``.

    Enforces at most one successful disk read per pipeline run; later calls return
    cached ``state[\"reviews\"]``.
    """

    _load_invoked_ok: list[bool] = []

    @tool("load_reviews")
    def load_reviews_tool(file_path: str) -> str:
        """
        Load product reviews from a local JSON file (array of review objects).

        Updates ``state[\"reviews\"]``. Only the first successful load reads disk;
        later calls return the cached list as JSON text.
        """
        t0 = time.perf_counter()
        agent_name = "Review Data Collector"
        tool_name = "load_reviews"
        if _load_invoked_ok:
            cached = state.get("reviews", [])
            logger.info(
                "[MAS][tool] load_reviews: single-call cache hit (%d reviews), skipping disk",
                len(cached) if isinstance(cached, list) else 0,
            )
            out = json.dumps(cached if isinstance(cached, list) else [], ensure_ascii=True)
            _append_tool_call(
                state,
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=f"file_path={file_path!r} (cache)",
                output_preview=out,
                duration_sec=time.perf_counter() - t0,
                error=None,
            )
            return out

        logger.info("[MAS][tool] load_reviews: reading file_path=%r", file_path)
        try:
            reviews = load_reviews_from_json(file_path)
            state["reviews"] = reviews
            _load_invoked_ok.append(True)
            if on_loaded is not None:
                on_loaded(reviews)
            out = json.dumps(reviews, ensure_ascii=True)
            _log_tool_io(
                agent_name,
                tool_name,
                f"file_path={file_path!r}",
                out,
                output_len=len(out),
            )
            logger.info("[MAS][tool] load_reviews: stored %d review(s) in state[\"reviews\"]", len(reviews))
            _append_tool_call(
                state,
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=f"file_path={file_path!r}",
                output_preview=out,
                duration_sec=time.perf_counter() - t0,
                error=None,
            )
            return out
        except Exception as exc:
            err = str(exc)
            logger.exception("[MAS][tool] load_reviews failed: %s", err)
            _append_tool_call(
                state,
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=f"file_path={file_path!r}",
                output_preview="",
                duration_sec=time.perf_counter() - t0,
                error=err,
            )
            raise

    return load_reviews_tool


def create_run_analysis_crew_tool(state: MutableMapping[str, Any]) -> Any:
    """Build the Statistical Review Analyst tool: ``statistical_analysis``."""

    @tool("statistical_analysis")
    def statistical_analysis(stage_note: str = "") -> str:
        """
        Run deterministic statistics on ``state[\"reviews\"]``.

        Persists a one-element list in ``state[\"analyzed\"]`` and returns JSON text.
        Optional ``stage_note`` is for logging only.
        """
        t0 = time.perf_counter()
        agent_name = "Statistical Review Analyst"
        tool_name = "statistical_analysis"
        logger.info("[MAS][tool] statistical_analysis input=%r", stage_note)
        reviews = state.get("reviews")
        if not isinstance(reviews, list) or not reviews:
            payload: dict[str, Any] = {"error": "No reviews in state; call load_reviews first."}
            state["analyzed"] = [payload]
            out = json.dumps(payload, ensure_ascii=True)
            _log_tool_io(agent_name, tool_name, stage_note, out, output_len=len(out))
            _append_tool_call(
                state,
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=stage_note or "(empty)",
                output_preview=out,
                duration_sec=time.perf_counter() - t0,
                error=payload.get("error"),
            )
            return out
        analysis = analyze_reviews_local(reviews)
        state["analyzed"] = [analysis]
        out = json.dumps(analysis, ensure_ascii=True)
        _log_tool_io(agent_name, tool_name, stage_note, out, output_len=len(out))
        _append_tool_call(
            state,
            agent_name=agent_name,
            tool_name=tool_name,
            tool_input=stage_note or "(empty)",
            output_preview=out,
            duration_sec=time.perf_counter() - t0,
            error=None,
        )
        return out

    return statistical_analysis


def create_run_fraud_crew_tool(state: MutableMapping[str, Any]) -> Any:
    """Build the Fraud Pattern Inspector tool: ``fraud_detection``."""

    @tool("fraud_detection")
    def fraud_detection(stage_note: str = "") -> str:
        """
        Run local fraud heuristics on ``state[\"reviews\"]``.

        Writes scored rows to ``state[\"scored\"]`` and a stage-3-shaped summary to
        ``state[\"fraud_report\"]``.
        """
        t0 = time.perf_counter()
        agent_name = "Fraud Pattern Inspector"
        tool_name = "fraud_detection"
        logger.info("[MAS][tool] fraud_detection input=%r", stage_note)
        reviews = state.get("reviews")
        if not isinstance(reviews, list) or not reviews:
            state["scored"] = []
            state["fraud_scored"] = []
            report: dict[str, Any] = {
                "error": "No reviews in state; call load_reviews first.",
                "flagged_review_count": 0,
                "flags": [],
                "suspicious_ratio": 0.0,
                "summary": "No reviews loaded; fraud detection skipped.",
            }
            state["fraud_report"] = report
            out = json.dumps(report, ensure_ascii=True)
            _log_tool_io(agent_name, tool_name, stage_note, out, output_len=len(out))
            _append_tool_call(
                state,
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=stage_note or "(empty)",
                output_preview=out,
                duration_sec=time.perf_counter() - t0,
                error=report["error"],
            )
            return out

        scored_reviews = detect_review_fraud(reviews)
        flagged = [r for r in scored_reviews if float(r.get("fraud_score", 0.0)) >= 0.5]
        report = {
            "flagged_review_count": len(flagged),
            "flags": [
                {
                    "review_id": str(r.get("id", "")),
                    "fraud_score": float(r.get("fraud_score", 0.0)),
                    "reasons": [str(x) for x in list(r.get("fraud_flags", []))],
                }
                for r in flagged
            ],
            "suspicious_ratio": round(len(flagged) / len(scored_reviews), 3) if scored_reviews else 0.0,
            "summary": f"Flagged {len(flagged)} suspicious review(s) out of {len(scored_reviews)} reviewed.",
        }
        state["scored"] = scored_reviews
        state["fraud_scored"] = report["flags"]
        state["fraud_report"] = report
        logger.info(
            "[MAS][tool] fraud_detection: scored=%d flagged=%d",
            len(scored_reviews),
            len(flagged),
        )
        out = json.dumps(report, ensure_ascii=True)
        _log_tool_io(agent_name, tool_name, stage_note, out, output_len=len(out))
        _append_tool_call(
            state,
            agent_name=agent_name,
            tool_name=tool_name,
            tool_input=stage_note or "(empty)",
            output_preview=out,
            duration_sec=time.perf_counter() - t0,
            error=None,
        )
        return out

    return fraud_detection


def create_run_recommendations_crew_tool(state: MutableMapping[str, Any]) -> Any:
    """Build the Product Recommendation Strategist tool: ``build_recommendations``."""

    @tool("build_recommendations")
    def build_recommendations_tool(stage_note: str = "") -> str:
        """
        Build the final recommendation from analysis and fraud signals in ``state``.

        Writes the decision dict to ``state[\"report\"]`` and returns JSON text.
        """
        t0 = time.perf_counter()
        agent_name = "Product Recommendation Strategist"
        tool_name = "build_recommendations"
        logger.info("[MAS][tool] build_recommendations input=%r", stage_note)
        scored = state.get("scored")
        if not isinstance(scored, list):
            scored = []
        analyzed = state.get("analyzed") or []
        analysis: dict[str, Any] = analyzed[0] if analyzed and isinstance(analyzed[0], dict) else {}
        flagged_count = len([r for r in scored if float(r.get("fraud_score", 0.0)) >= 0.5])
        fraud_report = {
            "flagged_review_count": flagged_count,
            "suspicious_ratio": round(flagged_count / len(scored), 3) if scored else 0.0,
        }
        logger.info("[MAS][tool] build_recommendations_start scored_count=%d", len(scored))
        rec = build_recommendations(scored, analysis, fraud_report)
        text = json.dumps(rec, indent=2, ensure_ascii=False)
        state["report"] = rec
        logger.info(
            "[MAS][tool] build_recommendations_result decision=%s confidence=%.3f",
            rec.get("decision"),
            float(rec.get("confidence", 0.0)),
        )
        _log_tool_io(agent_name, tool_name, stage_note, text, output_len=len(text))
        _append_tool_call(
            state,
            agent_name=agent_name,
            tool_name=tool_name,
            tool_input=stage_note or "(empty)",
            output_preview=text,
            duration_sec=time.perf_counter() - t0,
            error=None,
        )
        return text

    return build_recommendations_tool
