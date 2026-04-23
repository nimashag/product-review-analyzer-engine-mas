"""Local Streamlit demo for the product-review MAS (read-only wrapper around the real pipeline)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from coordinator import PipelineCoordinator, configure_logging
from main import run_pipeline

_ROOT = Path(__file__).resolve().parent
_SAMPLE = _ROOT / "sample_reviews.json"

logger = logging.getLogger("mas.demo_ui")


def _ensure_logging() -> None:
    if "_mas_logging_configured" not in st.session_state:
        configure_logging()
        st.session_state["_mas_logging_configured"] = True


def _write_upload_to_temp(upload_bytes: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="mas_reviews_",
        suffix=suffix or ".json",
        delete=False,
        dir=None,
    ) as tmp:
        tmp.write(upload_bytes)
        return Path(tmp.name)


def main() -> None:
    st.set_page_config(page_title="Product Review MAS", layout="wide")
    _ensure_logging()

    st.title("Product Review MAS (local)")
    st.caption("Same orchestration as `python main.py` — this app only drives and displays the pipeline.")

    with st.sidebar:
        st.header("Input")
        use_upload = st.radio("Reviews source", ("Bundled sample", "Upload JSON"), horizontal=False)
        uploaded = None
        if use_upload == "Upload JSON":
            uploaded = st.file_uploader("Reviews file (.json)", type=["json"])

        product_label = st.text_input("Product label (display only)", value="Product")

        st.divider()
        st.subheader("Advanced")
        deterministic = st.checkbox(
            "Tools-only run (no Crew LLM)",
            value=False,
            help="Same as `python main.py --deterministic`.",
        )
        non_strict = st.checkbox(
            "Non-strict validation",
            value=False,
            help="Same as `python main.py --non-strict`.",
        )
        no_assignment = st.checkbox(
            "Disable assignment safeguards",
            value=False,
            help="Same as `python main.py --no-assignment-mode`.",
        )

    assignment_mode = not no_assignment and not deterministic
    strict_mode = not non_strict

    run_clicked = st.button("Run pipeline", type="primary")

    if not run_clicked:
        st.info("Configure the sidebar and click **Run pipeline**.")
        st.stop()

    temp_path: Path | None = None
    review_file_str: str

    if use_upload == "Bundled sample":
        if not _SAMPLE.is_file():
            st.error(f"Sample file not found: {_SAMPLE}")
            st.stop()
        review_file_str = str(_SAMPLE.resolve())
    else:
        if uploaded is None:
            st.error("Upload a reviews JSON file before running.")
            st.stop()
        suffix = Path(uploaded.name).suffix or ".json"
        temp_path = _write_upload_to_temp(uploaded.getvalue(), suffix)
        review_file_str = str(temp_path.resolve())

    try:
        with st.spinner("Running pipeline (Ollama + CrewAI)…"):
            state, output_json = run_pipeline(
                review_file_str,
                run_llm_summary=not deterministic,
                strict_mode=strict_mode,
                assignment_mode=assignment_mode,
            )
        st.session_state.pop("_mas_last_error", None)
    except RuntimeError as exc:
        st.session_state["_mas_last_error"] = str(exc)
        st.error(str(exc))
        st.stop()
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temp upload %s", temp_path)

    snapshot_path = _ROOT / "last_pipeline_state.json"
    PipelineCoordinator.persist_execution_snapshot(dict(state), snapshot_path)
    logger.info("[demo_ui] Wrote execution snapshot to %s", snapshot_path)
    full_snapshot = snapshot_path.read_text(encoding="utf-8")

    report = state.get("report") or {}
    st.subheader(product_label)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Decision", report.get("decision", "—"))
    with c2:
        conf = report.get("confidence")
        st.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else "—")
    with c3:
        st.metric("Reviews", len(state.get("reviews") or []))

    st.markdown("**Summary**")
    st.write(report.get("summary") or "—")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Pros**")
        for p in report.get("pros") or []:
            st.markdown(f"- {p}")
        if not (report.get("pros") or []):
            st.caption("—")
    with col_b:
        st.markdown("**Cons**")
        for c in report.get("cons") or []:
            st.markdown(f"- {c}")
        if not (report.get("cons") or []):
            st.caption("—")

    st.markdown("**Red flags**")
    for r in report.get("red_flags") or []:
        st.markdown(f"- {r}")
    if not (report.get("red_flags") or []):
        st.caption("—")

    errs = state.get("errors") or []
    if errs:
        st.warning("Pipeline errors")
        for e in errs:
            st.text(e)

    st.download_button(
        "Download final report JSON",
        data=output_json.encode("utf-8"),
        file_name="final_report.json",
        mime="application/json",
    )

    st.download_button(
        "Download full execution snapshot (same as last_pipeline_state.json)",
        data=full_snapshot.encode("utf-8"),
        file_name="last_pipeline_state.json",
        mime="application/json",
    )

    with st.expander("Execution trace", expanded=False):
        st.json(state.get("execution_trace") or [])

    with st.expander("Tool calls", expanded=False):
        st.json(state.get("tool_calls") or [])

    with st.expander("Validation results", expanded=False):
        st.json(state.get("validation_results") or [])

    with st.expander("Stage timings", expanded=False):
        st.json(state.get("timings") or {})

    with st.expander("Authoritative / stage outputs (raw)", expanded=False):
        st.json(
            {
                "authoritative": state.get("authoritative", {}),
                "stage_outputs": state.get("stage_outputs", {}),
            }
        )


if __name__ == "__main__":
    main()
