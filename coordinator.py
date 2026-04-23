"""Coordinator/orchestrator layer for the local CrewAI review pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from crewai import Crew, Process, Task

from agents.analysis_agent import build_analysis_agent
from agents.fraud_agent import build_fraud_agent
from agents.recommendation_agent import build_recommendation_agent
from agents.scraper_agent import build_scraper_agent
from tools.crew_tool_factories import (
    create_load_reviews_crew_tool,
    create_run_analysis_crew_tool,
    create_run_fraud_crew_tool,
    create_run_recommendations_crew_tool,
)
from tools.review_analysis import analyze_reviews_local
from validation.validators import (
    as_validation_error_text,
    validate_stage1,
    validate_stage2,
    validate_stage3,
    validate_stage4,
)

logger = logging.getLogger("mas.coordinator")

OLLAMA_BASE_URL = "http://localhost:11434"
# Local chat model for agent reasoning only (no LLM tool/function calling). Override with MAS_OLLAMA_MODEL.
OLLAMA_MODEL = os.environ.get("MAS_OLLAMA_MODEL", "phi3:mini")

AGENT_REASONING_ONLY_RULES = """
CRITICAL RULES (coordinator already ran all Python tools; you have NO tools):
- Do NOT call tools, functions, or external APIs. You only reason over the PRECOMPUTED JSON in this task.
- Output ONLY one JSON object parseable by Python json.loads().
- No markdown, no code fences, no commentary before or after the JSON.
- Do not wrap values in function calls (no int(), float(), str() in output).
- TOOL_OUTPUT_JSON blocks are authoritative for counts, IDs, scores, flags, decision, and confidence; mirror them exactly.
- You may rephrase only where the task explicitly allows natural language (e.g. summary, pros, cons on stage 4; summary on stage 3).
""".strip()


@dataclass
class PipelineState:
    """Shared state object used across all pipeline stages."""

    reviews: list[dict[str, Any]] = field(default_factory=list)
    analyzed: list[dict[str, Any]] = field(default_factory=list)
    scored: list[dict[str, Any]] = field(default_factory=list)
    fraud_scored: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    run_id: str = ""
    authoritative: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    timestamps: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass state into mutable dict for tools/serialization."""
        return asdict(self)


def verify_ollama_running(base_url: str = OLLAMA_BASE_URL) -> None:
    """Fail fast if Ollama is not reachable."""
    logger.info("[Coordinator] Checking Ollama connection")
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama returned HTTP {resp.status} for {url!r}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        msg = (
            "Ollama is not reachable. Start the Ollama app or `ollama serve`, confirm it "
            f"listens on {base_url}, run `ollama pull {OLLAMA_MODEL}`, then retry. "
            f"Last error: {exc}"
        )
        raise RuntimeError(msg) from exc


def build_crewai_ollama_llm() -> str:
    """Return CrewAI Ollama LLM selector (local inference only)."""
    os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_TRACING_ENABLED"] = "false"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return f"ollama/{OLLAMA_MODEL}"


def configure_logging(log_file: str | None = "logs/system.log") -> None:
    """Configure console+file logging for coordinator observability."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )


class PipelineCoordinator:
    """Controls end-to-end execution and shared state transitions."""

    def __init__(self, *, llm: str | None = None) -> None:
        self.llm = llm or build_crewai_ollama_llm()

    def _ollama_model_id(self) -> str:
        sel = str(self.llm or "")
        if sel.lower().startswith("ollama/"):
            return sel.split("/", 1)[1]
        return OLLAMA_MODEL

    def _ollama_json_repair_once(self, broken: str) -> dict[str, Any] | None:
        """Single local Ollama call to recover one JSON object from noisy agent text."""
        snippet = (broken or "").strip()
        if len(snippet) > 12_000:
            snippet = snippet[:12_000] + "\n... [truncated]"
        prompt = (
            "Convert the following assistant output into exactly one valid JSON object. "
            "Output ONLY raw JSON — no markdown fences, no commentary, no function wrappers.\n\n"
            f"{snippet}"
        )
        payload = json.dumps(
            {"model": self._ollama_model_id(), "prompt": prompt, "stream": False},
        ).encode("utf-8")
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = str(body.get("response") or "").strip()
            return PipelineCoordinator._coerce_json_object_from_text(text, allow_repair_slices=True)
        except Exception as exc:
            logger.warning("[Coordinator] Ollama JSON repair pass failed: %s", exc)
            return None

    @staticmethod
    def _coerce_json_object_from_text(raw_text: str, *, allow_repair_slices: bool = True) -> dict[str, Any]:
        """Parse the first JSON object from model text (strict load, fenced blocks, or brace slice)."""
        attempts: list[str] = []
        text = raw_text.strip()
        if text:
            attempts.append(text)
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text, flags=re.IGNORECASE)
        if fence:
            attempts.append(fence.group(1).strip())
        for candidate in attempts:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        if allow_repair_slices:
            start = raw_text.find("{")
            if start >= 0:
                depth = 0
                for i, ch in enumerate(raw_text[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(raw_text[start : i + 1])
                                if isinstance(obj, dict):
                                    return obj
                            except json.JSONDecodeError:
                                pass
                            break
        raise ValueError("Could not parse a JSON object from LLM text")

    def _parse_llm_task_output(self, raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return PipelineCoordinator._coerce_json_object_from_text(
                str(raw_text or ""),
                allow_repair_slices=True,
            ), None
        except ValueError as first_exc:
            repaired = self._ollama_json_repair_once(str(raw_text or ""))
            if repaired is not None:
                return repaired, None
            return None, str(first_exc)

    @staticmethod
    def resolve_with_tool_priority(
        *,
        llm_payload: dict[str, Any] | None,
        tool_payload: dict[str, Any],
        stage_validate: Callable[..., Any],
        reviews: list[dict[str, Any]] | None,
        stage_name: str = "",
    ) -> tuple[dict[str, Any], str, bool]:
        """
        Merge tool-backed truth with validated LLM JSON.

        Numeric / structural fields stay tool-authoritative. When the LLM validates
        but disagrees, interpretive fields (stage 3–4 summaries; stage 4 pros/cons)
        may be merged when the combined object still validates.
        """
        tool_final = dict(tool_payload)
        if not llm_payload:
            return tool_final, "tool_only", False
        try:
            if reviews is not None:
                model = stage_validate(llm_payload, reviews)
            else:
                model = stage_validate(llm_payload)
            llm_norm = model.model_dump()
            if llm_norm == tool_final:
                return tool_final, "llm_tool_agreement", True

            merged = dict(tool_final)
            if stage_name == "stage3":
                summary_llm = str(llm_norm.get("summary") or "").strip()
                if summary_llm:
                    merged["summary"] = summary_llm
            elif stage_name == "stage4":
                summary_llm = str(llm_norm.get("summary") or "").strip()
                if summary_llm:
                    merged["summary"] = summary_llm
                pros = llm_norm.get("pros")
                cons = llm_norm.get("cons")
                if isinstance(pros, list) and pros:
                    merged["pros"] = [str(x) for x in pros]
                if isinstance(cons, list) and cons:
                    merged["cons"] = [str(x) for x in cons]

            try:
                if reviews is not None:
                    stage_validate(merged, reviews)
                else:
                    stage_validate(merged)
                if merged != tool_final:
                    logger.info(
                        "[Coordinator] Merged tool output with LLM interpretive fields (stage=%s).",
                        stage_name,
                    )
                    return merged, "llm_tool_merged", True
            except Exception as merge_exc:
                logger.warning(
                    "[Coordinator] LLM/tool merge failed validation; using tool fields only: %s",
                    as_validation_error_text(merge_exc),
                )

            logger.warning(
                "[Coordinator] LLM output differs from tool truth; using tool-authoritative fields (stage=%s).",
                stage_name,
            )
            return tool_final, "tool_authority_numeric", True
        except Exception as exc:
            logger.error(
                "[Coordinator] LLM output failed validation; using tool truth (LLM retained in trace): %s",
                as_validation_error_text(exc),
            )
            return tool_final, "tool_fallback_invalid_llm", False

    def _run_stage(
        self,
        state: dict[str, Any],
        stage_name: str,
        fn: Any,
        *args: Any,
        continue_on_error: bool = False,
        **kwargs: Any,
    ) -> Any:
        start = time.perf_counter()
        before = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in state.items()}
        logger.info("[Coordinator] Running %s...", stage_name)
        logger.info("[State][before:%s] %s", stage_name, before)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive for demo runtime
            msg = f"{stage_name} failed: {exc}"
            state.setdefault("errors", []).append(msg)
            logger.exception("[Coordinator] %s", msg)
            if continue_on_error:
                return ""
            raise
        finally:
            elapsed = round(time.perf_counter() - start, 3)
            state.setdefault("timings", {})[stage_name] = elapsed
            after = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in state.items()}
            logger.info("[State][after:%s] %s (%.3fs)", stage_name, after, elapsed)

    def run_tool_phase(
        self,
        state: dict[str, Any],
        review_file: str,
        tool_load: Any,
        tool_analysis: Any,
        tool_fraud: Any,
        tool_rec: Any,
        *,
        continue_on_error: bool,
    ) -> None:
        """
        Execute the four pipeline tools sequentially in-process (coordinator-driven).

        This is the authoritative tool path: the LLM never invokes these tools.
        """
        logger.info(
            "[Coordinator][tools] Coordinator invoking: load_reviews -> statistical_analysis -> "
            "fraud_detection -> build_recommendations (deterministic Python layer).",
        )
        self._run_stage(
            state,
            "loader",
            tool_load.run,
            file_path=review_file,
            continue_on_error=continue_on_error,
        )
        self._run_stage(
            state,
            "analysis",
            tool_analysis.run,
            continue_on_error=continue_on_error,
            stage_note="",
        )
        self._run_stage(
            state,
            "fraud",
            tool_fraud.run,
            continue_on_error=continue_on_error,
            stage_note="",
        )
        self._run_stage(
            state,
            "recommendation_tool",
            tool_rec.run,
            continue_on_error=continue_on_error,
            stage_note="",
        )

    def run_authoritative_pipeline(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build tool-backed authoritative stage payloads from ``state`` after tools ran."""
        reviews_payload = state.get("reviews", [])
        if not isinstance(reviews_payload, list):
            reviews_payload = []

        stage1_authoritative = self._build_stage1_authoritative(reviews_payload)

        analyzed = state.get("analyzed") or []
        analysis_payload: dict[str, Any] = {}
        if analyzed and isinstance(analyzed[0], dict) and "error" not in analyzed[0]:
            analysis_payload = dict(analyzed[0])
        elif reviews_payload:
            # Deterministic repair if the analysis tool did not populate state (agent/tool failure).
            analysis_payload = analyze_reviews_local(reviews_payload)
        stage2_authoritative = self._build_stage2_authoritative(analysis_payload, reviews_payload)

        fraud_raw = state.get("fraud_report")
        if isinstance(fraud_raw, dict) and fraud_raw.get("error"):
            stage3_authoritative = self._stage3_fallback_from_reviews(reviews_payload)
        elif isinstance(fraud_raw, dict) and fraud_raw:
            stage3_authoritative = self._build_stage3_authoritative(fraud_raw)
        else:
            stage3_authoritative = self._stage3_fallback_from_reviews(reviews_payload)

        rec_payload = state.get("report") if isinstance(state.get("report"), dict) else {}
        stage4_authoritative = self._build_stage4_authoritative(rec_payload)

        return {
            "review_count": len(reviews_payload),
            "stage1": dict(stage1_authoritative),
            "stage2": dict(stage2_authoritative),
            "stage3": dict(stage3_authoritative),
            "stage4": dict(stage4_authoritative),
        }

    def run_reasoning_crew_phase(
        self,
        state: dict[str, Any],
        review_file: str,
        *,
        continue_on_error: bool,
    ) -> tuple[Any, str]:
        """
        Run the CrewAI crew for LLM reasoning only: agents have no tools.

        Requires :meth:`run_tool_phase` and validated ``state[\"authoritative\"]`` beforehand.
        """
        auth = state.get("authoritative") or {}
        s1 = dict(auth.get("stage1") or {})
        s2 = dict(auth.get("stage2") or {})
        s3 = dict(auth.get("stage3") or {})
        s4 = dict(auth.get("stage4") or {})

        def _compact(obj: Any) -> str:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

        reviews_live = state.get("reviews") or []
        analyzed_rows = state.get("analyzed") or []
        analyzed_tool = analyzed_rows[0] if analyzed_rows and isinstance(analyzed_rows[0], dict) else {}
        fraud_tool = state.get("fraud_report") if isinstance(state.get("fraud_report"), dict) else {}
        rec_tool = state.get("report") if isinstance(state.get("report"), dict) else {}

        scraper = build_scraper_agent(llm=self.llm)
        analyst = build_analysis_agent(llm=self.llm)
        fraud = build_fraud_agent(llm=self.llm)
        recommender = build_recommendation_agent(llm=self.llm)

        task_scrape = Task(
            description=(
                f"{AGENT_REASONING_ONLY_RULES}\n\n"
                "Stage 1 — Review Data Collector.\n"
                "The coordinator already loaded JSON from the path given by kickoff input review_file.\n"
                "TOOL_OUTPUT_JSON (load_reviews / coordinator; authoritative):\n"
                f"{_compact(s1)}\n"
                f"RAW_TOOL_STATE reviews_loaded_count={len(reviews_live) if isinstance(reviews_live, list) else 0}\n\n"
                "Output ONLY one JSON object: "
                '{"review_count": <int>, "review_ids": [<string ids in file order>]}\n'
            ),
            expected_output="Single JSON object only.",
            agent=scraper,
        )
        task_analyze = Task(
            description=(
                f"{AGENT_REASONING_ONLY_RULES}\n\n"
                "Stage 2 — Statistical Review Analyst.\n"
                "TOOL_OUTPUT_JSON (statistical_analysis tool row; authoritative):\n"
                f"{_compact(s2)}\n"
                f"RAW_TOOL_OUTPUT statistical_analysis:\n{_compact(analyzed_tool)}\n\n"
                "Output ONLY one JSON object with this exact shape; field values must match the blocks above:\n"
                '{"review_count": 0, "average_rating": 0, "rating_distribution": {}, '
                '"verified_ratio": 0, "keyword_negative_hits": 0, "keyword_positive_hits": 0}\n'
                "rating_distribution keys must be strings.\n"
            ),
            expected_output="Single JSON object only.",
            agent=analyst,
        )
        task_fraud = Task(
            description=(
                f"{AGENT_REASONING_ONLY_RULES}\n\n"
                "Stage 3 — Fraud Pattern Inspector.\n"
                "TOOL_OUTPUT_JSON (fraud_detection; authoritative):\n"
                f"{_compact(s3)}\n"
                f"RAW_TOOL_OUTPUT fraud_report:\n{_compact(fraud_tool)}\n\n"
                "Output ONLY one JSON object:\n"
                '{"flagged_review_count": 0, "suspicious_ratio": 0, "flags": [], "summary": ""}\n'
                "Each flag: review_id, fraud_score, reasons (array of strings). "
                "flagged_review_count, suspicious_ratio, and flags must match the block above exactly; "
                "you may replace summary with a clearer analyst explanation if content remains consistent.\n"
            ),
            expected_output="Single JSON object only.",
            agent=fraud,
        )
        task_rec = Task(
            description=(
                f"{AGENT_REASONING_ONLY_RULES}\n\n"
                "Stage 4 — Product Recommendation Strategist.\n"
                "TOOL_OUTPUT_JSON (build_recommendations; authoritative for decision/confidence/red_flags):\n"
                f"{_compact(s4)}\n"
                f"RAW_TOOL_OUTPUT report:\n{_compact(rec_tool)}\n\n"
                "Output ONLY one JSON object:\n"
                '{"decision": "BUY", "confidence": 0, "pros": [], "cons": [], "red_flags": [], "summary": ""}\n'
                'decision must be one of: "BUY", "CONSIDER", "AVOID". '
                "Structured fields must match the authoritative JSON; you may refine summary for readability.\n"
            ),
            expected_output="Single JSON object only.",
            agent=recommender,
        )

        crew = Crew(
            agents=[scraper, analyst, fraud, recommender],
            tasks=[task_scrape, task_analyze, task_fraud, task_rec],
            process=Process.sequential,
            verbose=True,
        )
        logger.info(
            "[Coordinator][run_id=%s] reasoning crew kickoff (no agent tools) review_file=%r",
            state.get("run_id"),
            review_file,
        )
        output = self._run_stage(
            state,
            "crew_reasoning",
            crew.kickoff,
            inputs={"review_file": review_file},
            continue_on_error=continue_on_error,
        )
        crew_text = str(output or "")
        return output, crew_text

    def merge_pipeline_and_crew(
        self,
        *,
        pipeline_report: dict[str, Any],
        crew_raw: str,
        strict_mode: bool,
    ) -> dict[str, Any]:
        """Prefer validated pipeline (tool truth); accept crew only if it matches exactly in strict mode."""
        if not strict_mode:
            try:
                return self._validate_report_or_raise(crew_raw)
            except Exception as exc:
                logger.warning("[Coordinator] Non-strict mode crew output rejected: %s", exc)
                return pipeline_report

        try:
            crew_report = self._validate_report_or_raise(crew_raw)
        except Exception as exc:
            logger.warning("[Coordinator] Strict mode discarded crew output (invalid JSON/schema): %s", exc)
            return pipeline_report

        if crew_report != pipeline_report:
            logger.warning(
                "[Coordinator] Strict mode discarded crew output (mismatch with authoritative pipeline result).",
            )
            return pipeline_report

        logger.info("[Coordinator] Crew output accepted (exact pipeline match).")
        return crew_report

    @staticmethod
    def persist_execution_snapshot(state: dict[str, Any], path: Path) -> None:
        """Write full pipeline observability snapshot for grading."""
        snapshot: dict[str, Any] = {
            "run_id": state.get("run_id", ""),
            "timestamps": dict(state.get("timestamps") or {}),
            "reviews": state.get("reviews", []),
            "analyzed": state.get("analyzed", []),
            "scored": state.get("scored", []),
            "fraud_scored": state.get("fraud_scored", []),
            "report": state.get("report", {}),
            "authoritative": state.get("authoritative", {}),
            "stage_outputs": state.get("stage_outputs", {}),
            "execution_trace": state.get("execution_trace", []),
            "tool_calls": state.get("tool_calls", []),
            "validation_results": state.get("validation_results", []),
            "timings": state.get("timings", {}),
            "errors": state.get("errors", []),
            "crew_output_raw": state.get("crew_output_raw", ""),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[Coordinator] Wrote execution snapshot to %s", path)

    def run(
        self,
        review_file: str,
        *,
        run_llm_summary: bool = True,
        parallel_analysis_fraud: bool = False,
        continue_on_error: bool = False,
        strict_mode: bool = True,
        assignment_mode: bool = True,
    ) -> tuple[dict[str, Any], str]:
        """Run the full pipeline: coordinator executes tools; optional Crew does LLM reasoning only (no tools)."""
        if parallel_analysis_fraud:
            logger.info("[Coordinator] parallel_analysis_fraud is deprecated; using sequential stages.")

        if run_llm_summary:
            verify_ollama_running()

        state_obj = PipelineState()
        state = state_obj.to_dict()
        state["run_id"] = str(uuid4())
        started = datetime.now(timezone.utc).isoformat()
        state["timestamps"] = {"started_at": started}

        if assignment_mode:
            strict_mode = True
            run_llm_summary = True

        logger.info(
            "[Coordinator][run_id=%s] Pipeline start review_file=%r assignment_mode=%s strict_mode=%s llm=%s",
            state["run_id"],
            review_file,
            assignment_mode,
            strict_mode,
            self.llm,
        )

        def _on_loaded(rows: list[Any]) -> None:
            logger.info("[State] reviews loaded: %d item(s)", len(rows))

        tool_load = create_load_reviews_crew_tool(state, on_loaded=_on_loaded)
        tool_analysis = create_run_analysis_crew_tool(state)
        tool_fraud = create_run_fraud_crew_tool(state)
        tool_rec = create_run_recommendations_crew_tool(state)

        self.run_tool_phase(
            state,
            review_file,
            tool_load,
            tool_analysis,
            tool_fraud,
            tool_rec,
            continue_on_error=continue_on_error,
        )

        authoritative_bundle = self.run_authoritative_pipeline(state)
        state["authoritative"] = authoritative_bundle

        reviews_payload = state.get("reviews", [])
        if not isinstance(reviews_payload, list):
            reviews_payload = []

        stage1_authoritative = self._validate_or_fallback(
            state=state,
            stage_name="stage1_pre",
            validator=lambda: validate_stage1(authoritative_bundle["stage1"], reviews_payload),
            fallback=self._build_stage1_authoritative(reviews_payload),
        )
        stage2_authoritative = self._validate_or_fallback(
            state=state,
            stage_name="stage2_pre",
            validator=lambda: validate_stage2(authoritative_bundle["stage2"], reviews_payload),
            fallback=self._stage2_fallback_from_reviews(reviews_payload),
        )
        stage3_authoritative = self._validate_or_fallback(
            state=state,
            stage_name="stage3_pre",
            validator=lambda: validate_stage3(authoritative_bundle["stage3"], reviews_payload),
            fallback=self._stage3_fallback_from_reviews(reviews_payload),
        )
        stage4_authoritative = self._validate_or_fallback(
            state=state,
            stage_name="stage4_pre",
            validator=lambda: validate_stage4(authoritative_bundle["stage4"]),
            fallback=self._stage4_fallback_from_upstream(stage2_authoritative, stage3_authoritative),
        )

        state["authoritative"] = {
            "review_count": len(reviews_payload),
            "stage1": stage1_authoritative,
            "stage2": dict(stage2_authoritative),
            "stage3": dict(stage3_authoritative),
            "stage4": dict(stage4_authoritative),
        }

        if run_llm_summary:
            output, crew_text = self.run_reasoning_crew_phase(
                state,
                review_file,
                continue_on_error=continue_on_error,
            )
        else:
            logger.info("[Coordinator] run_llm_summary=False: skipping reasoning crew (coordinator tools only).")
            output, crew_text = None, ""

        task_outputs = self._extract_task_outputs(output) if output is not None else []
        llm_stage_outputs, llm_task_parse_errors = self._parse_stage_outputs_from_tasks(task_outputs)

        stage1_final = self._validate_stage_output(
            state=state,
            stage_name="stage1",
            agent_name="Review Data Collector",
            tool_name="load_reviews",
            raw_agent_output=llm_stage_outputs.get("stage1"),
            authoritative=state["authoritative"]["stage1"],
            stage_validate=validate_stage1,
            reviews=reviews_payload,
            input_summary={"review_file": review_file, "reviews_count": len(reviews_payload)},
            llm_task_parse_error=llm_task_parse_errors.get("stage1"),
        )
        stage2_final = self._validate_stage_output(
            state=state,
            stage_name="stage2",
            agent_name="Statistical Review Analyst",
            tool_name="statistical_analysis",
            raw_agent_output=llm_stage_outputs.get("stage2"),
            authoritative=state["authoritative"]["stage2"],
            stage_validate=validate_stage2,
            reviews=reviews_payload,
            input_summary={"reviews_count": len(reviews_payload)},
            llm_task_parse_error=llm_task_parse_errors.get("stage2"),
        )
        stage3_final = self._validate_stage_output(
            state=state,
            stage_name="stage3",
            agent_name="Fraud Pattern Inspector",
            tool_name="fraud_detection",
            raw_agent_output=llm_stage_outputs.get("stage3"),
            authoritative=state["authoritative"]["stage3"],
            stage_validate=validate_stage3,
            reviews=reviews_payload,
            input_summary={"reviews_count": len(reviews_payload)},
            llm_task_parse_error=llm_task_parse_errors.get("stage3"),
        )
        stage4_final = self._validate_stage_output(
            state=state,
            stage_name="stage4",
            agent_name="Product Recommendation Strategist",
            tool_name="build_recommendations",
            raw_agent_output=llm_stage_outputs.get("stage4"),
            authoritative=state["authoritative"]["stage4"],
            stage_validate=validate_stage4,
            reviews=None,
            input_summary={"reviews_count": len(reviews_payload)},
            llm_task_parse_error=llm_task_parse_errors.get("stage4"),
        )

        state["stage_outputs"] = {
            "stage1": stage1_final,
            "stage2": stage2_final,
            "stage3": stage3_final,
            "stage4": stage4_final,
        }
        state["report"] = stage4_final
        pipeline_report = stage4_final

        logger.info(
            "[Coordinator][run_id=%s] crew_output_summary chars=%d preview=%r",
            state["run_id"],
            len(crew_text),
            crew_text[:240],
        )
        logger.info("[Coordinator] pipeline_result=%s", pipeline_report)
        logger.info("[Coordinator] crew_result_raw=%r", crew_text[:500])

        final_report = self.merge_pipeline_and_crew(
            pipeline_report=pipeline_report,
            crew_raw=crew_text,
            strict_mode=strict_mode,
        )
        state["report"] = final_report
        state["crew_output_raw"] = crew_text
        state["timestamps"]["finished_at"] = datetime.now(timezone.utc).isoformat()

        logger.info("[Coordinator] final_merged_result=%s", final_report)
        return state, json.dumps(final_report, ensure_ascii=False)

    def _validate_stage_output(
        self,
        *,
        state: dict[str, Any],
        stage_name: str,
        agent_name: str,
        tool_name: str,
        raw_agent_output: dict[str, Any] | None,
        authoritative: dict[str, Any],
        stage_validate: Callable[..., Any],
        reviews: list[dict[str, Any]] | None,
        input_summary: dict[str, Any],
        llm_task_parse_error: str | None = None,
    ) -> dict[str, Any]:
        """Validate LLM stage JSON against tool truth; log structured AgentOps fields."""
        t0 = time.perf_counter()
        final_output, output_source, llm_ok = self.resolve_with_tool_priority(
            llm_payload=raw_agent_output,
            tool_payload=authoritative,
            stage_validate=stage_validate,
            reviews=reviews,
            stage_name=stage_name,
        )
        validation_passed = output_source != "tool_fallback_invalid_llm"

        elapsed = round(time.perf_counter() - t0, 4)
        logger.info(
            "[MAS][stage] agent=%s tool=%s stage=%s source=%s validation_passed=%s time_sec=%s input=%s output=%s",
            agent_name,
            tool_name,
            stage_name,
            output_source,
            validation_passed,
            elapsed,
            input_summary,
            final_output,
        )

        trace_entry: dict[str, Any] = {
            "stage": stage_name,
            "agent": agent_name,
            "tool": tool_name,
            "input": input_summary,
            "tool_output": authoritative,
            "llm_output": raw_agent_output,
            "final_output": final_output,
            "validation_passed": validation_passed,
            "source": output_source,
            "llm_parse_ok": llm_ok,
            "llm_task_parse_error": llm_task_parse_error,
            "elapsed_sec": elapsed,
        }
        state.setdefault("execution_trace", []).append(trace_entry)
        state.setdefault("validation_results", []).append(
            {
                "stage": stage_name,
                "agent": agent_name,
                "tool": tool_name,
                "validation_passed": validation_passed,
                "source": output_source,
                "elapsed_sec": elapsed,
            },
        )
        return final_output

    @staticmethod
    def _extract_task_outputs(output: Any) -> list[Any]:
        maybe = getattr(output, "tasks_output", None)
        return list(maybe) if isinstance(maybe, list) else []

    def _parse_stage_outputs_from_tasks(
        self,
        task_outputs: list[Any],
    ) -> tuple[dict[str, dict[str, Any] | None], dict[str, str | None]]:
        stage_keys = ["stage1", "stage2", "stage3", "stage4"]
        parsed: dict[str, dict[str, Any] | None] = {k: None for k in stage_keys}
        errors: dict[str, str | None] = {k: None for k in stage_keys}
        for idx, stage in enumerate(stage_keys):
            if idx >= len(task_outputs):
                continue
            raw = getattr(task_outputs[idx], "raw", task_outputs[idx])
            obj, err = self._parse_llm_task_output(str(raw))
            parsed[stage] = obj
            errors[stage] = err
        return parsed, errors

    def _validate_report_or_raise(self, raw_text: str) -> dict[str, Any]:
        """Parse and validate final report JSON from crew output."""
        payload = PipelineCoordinator._coerce_json_object_from_text(raw_text, allow_repair_slices=True)
        model = validate_stage4(payload)
        return model.model_dump()

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any]:
        """Parse a JSON object from raw crew or CLI text (lenient)."""
        return PipelineCoordinator._coerce_json_object_from_text(raw_text, allow_repair_slices=True)

    def _validate_or_fallback(
        self,
        *,
        state: dict[str, Any],
        stage_name: str,
        validator: Callable[[], Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            model = validator()
            return model.model_dump() if hasattr(model, "model_dump") else fallback
        except Exception as exc:
            msg = f"{stage_name} validation failed; using deterministic fallback: {as_validation_error_text(exc)}"
            state.setdefault("errors", []).append(msg)
            logger.error("[Coordinator] %s", msg)
            return fallback

    @staticmethod
    def _build_stage1_authoritative(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        return {"review_count": len(reviews), "review_ids": [str(r.get("id", "")) for r in reviews]}

    @staticmethod
    def _build_stage2_authoritative(
        analysis_payload: dict[str, Any],
        reviews: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "review_count": PipelineCoordinator._safe_int(analysis_payload.get("review_count"), len(reviews)),
            "average_rating": PipelineCoordinator._safe_float(analysis_payload.get("average_rating"), 0.0),
            "rating_distribution": {
                str(k): PipelineCoordinator._safe_int(v, 0)
                for k, v in dict(analysis_payload.get("rating_distribution", {})).items()
            },
            "verified_ratio": PipelineCoordinator._safe_float(analysis_payload.get("verified_ratio"), 0.0),
            "keyword_negative_hits": PipelineCoordinator._safe_int(
                analysis_payload.get("keyword_negative_hits"),
                0,
            ),
            "keyword_positive_hits": PipelineCoordinator._safe_int(
                analysis_payload.get("keyword_positive_hits"),
                0,
            ),
        }

    @staticmethod
    def _build_stage3_authoritative(fraud_payload: dict[str, Any]) -> dict[str, Any]:
        flagged_count = PipelineCoordinator._safe_int(fraud_payload.get("flagged_review_count"), 0)
        return {
            "flagged_review_count": flagged_count,
            "suspicious_ratio": PipelineCoordinator._safe_float(fraud_payload.get("suspicious_ratio"), 0.0),
            "flags": [
                {
                    "review_id": str(flag.get("review_id", "")),
                    "fraud_score": PipelineCoordinator._safe_float(flag.get("fraud_score"), 0.0),
                    "reasons": [str(x) for x in list(flag.get("reasons", []))],
                }
                for flag in list(fraud_payload.get("flags", []))
                if isinstance(flag, dict)
            ],
            "summary": str(
                fraud_payload.get("summary")
                or f"Flagged {flagged_count} suspicious review(s).",
            ),
        }

    @staticmethod
    def _build_stage4_authoritative(rec_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": str(rec_payload.get("decision", "CONSIDER")).upper(),
            "confidence": round(PipelineCoordinator._safe_float(rec_payload.get("confidence"), 0.0), 2),
            "pros": [str(x) for x in list(rec_payload.get("pros", []))],
            "cons": [str(x) for x in list(rec_payload.get("cons", []))],
            "red_flags": [str(x) for x in list(rec_payload.get("red_flags", []))],
            "summary": str(rec_payload.get("summary", "")).strip(),
        }

    @staticmethod
    def _stage2_fallback_from_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "review_count": len(reviews),
            "average_rating": 0.0,
            "rating_distribution": {},
            "verified_ratio": 0.0,
            "keyword_negative_hits": 0,
            "keyword_positive_hits": 0,
        }

    @staticmethod
    def _stage3_fallback_from_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "flagged_review_count": 0,
            "suspicious_ratio": 0.0,
            "flags": [],
            "summary": f"No suspicious reviews flagged out of {len(reviews)} review(s).",
        }

    @staticmethod
    def _stage4_fallback_from_upstream(
        stage2: dict[str, Any],
        stage3: dict[str, Any],
    ) -> dict[str, Any]:
        avg_rating = PipelineCoordinator._safe_float(stage2.get("average_rating"), 0.0)
        suspicious_ratio = PipelineCoordinator._safe_float(stage3.get("suspicious_ratio"), 0.0)
        decision = (
            "BUY"
            if avg_rating >= 4.0 and suspicious_ratio < 0.3
            else "AVOID"
            if avg_rating < 2.5
            else "CONSIDER"
        )
        confidence = max(0.0, min(1.0, round((avg_rating / 5.0) * (1.0 - suspicious_ratio), 2)))
        return {
            "decision": decision,
            "confidence": confidence,
            "pros": [],
            "cons": ["fallback_recommendation_used"],
            "red_flags": ["stage4_validation_failed"],
            "summary": "Fallback recommendation generated from validated upstream stages.",
        }

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if value is None:
                return default
            if isinstance(value, bool):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, bool):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default