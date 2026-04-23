"""
Tests for ``load_reviews`` and the CrewAI ``@tool`` wrapper used by the Scraper Agent.

Deterministic tests require **no** LLM and **no** network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.crew_tool_factories import create_load_reviews_crew_tool
from tools.load_reviews import (
    ReviewsFileNotFoundError,
    ReviewsJSONError,
    ReviewsSchemaError,
    load_reviews,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_reviews_success_sample_file() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "sample_reviews.json"
    reviews = load_reviews(str(sample))
    assert isinstance(reviews, list)
    assert len(reviews) >= 5
    for rev in reviews:
        assert set(rev.keys()) == {"id", "text", "rating", "date", "verified"}
        assert isinstance(rev["id"], str) and rev["id"]
        assert isinstance(rev["text"], str) and rev["text"].strip()
        assert isinstance(rev["rating"], int)
        assert 1 <= rev["rating"] <= 5
        assert isinstance(rev["date"], str) and rev["date"]
        assert isinstance(rev["verified"], bool)


def test_load_reviews_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    _write_json(p, [])
    assert load_reviews(str(p)) == []


def test_load_reviews_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ReviewsFileNotFoundError):
        load_reviews(str(missing))


def test_load_reviews_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReviewsJSONError):
        load_reviews(str(p))


def test_load_reviews_root_not_list(tmp_path: Path) -> None:
    p = tmp_path / "obj.json"
    _write_json(p, {"reviews": []})
    with pytest.raises(ReviewsJSONError):
        load_reviews(str(p))


def test_load_reviews_missing_required_key(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    _write_json(
        p,
        [
            {
                "id": "x",
                "text": "ok",
                "rating": 3,
                "date": "2026-01-01",
            }
        ],
    )
    with pytest.raises(ReviewsSchemaError):
        load_reviews(str(p))


def test_load_reviews_extra_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "extra.json"
    _write_json(
        p,
        [
            {
                "id": "x",
                "text": "ok",
                "rating": 3,
                "date": "2026-01-01",
                "verified": True,
                "unexpected": 123,
            }
        ],
    )
    with pytest.raises(ReviewsSchemaError):
        load_reviews(str(p))


def test_load_reviews_rating_out_of_range(tmp_path: Path) -> None:
    p = tmp_path / "rating.json"
    _write_json(
        p,
        [
            {
                "id": "x",
                "text": "ok",
                "rating": 6,
                "date": "2026-01-01",
                "verified": True,
            }
        ],
    )
    with pytest.raises(ReviewsSchemaError):
        load_reviews(str(p))


def test_load_reviews_bool_is_not_int_rating(tmp_path: Path) -> None:
    p = tmp_path / "bool_rating.json"
    _write_json(
        p,
        [
            {
                "id": "x",
                "text": "ok",
                "rating": True,
                "date": "2026-01-01",
                "verified": True,
            }
        ],
    )
    with pytest.raises(ReviewsSchemaError):
        load_reviews(str(p))


def test_load_reviews_verified_must_be_bool(tmp_path: Path) -> None:
    p = tmp_path / "verified_str.json"
    _write_json(
        p,
        [
            {
                "id": "x",
                "text": "ok",
                "rating": 3,
                "date": "2026-01-01",
                "verified": "true",
            }
        ],
    )
    with pytest.raises(ReviewsSchemaError):
        load_reviews(str(p))


def test_tool_wrapper_updates_state_and_returns_json(tmp_path: Path) -> None:
    p = tmp_path / "ok.json"
    _write_json(
        p,
        [
            {
                "id": "a1",
                "text": "hello",
                "rating": 2,
                "date": "2026-04-01",
                "verified": False,
            }
        ],
    )

    state: dict[str, object] = {}
    tool = create_load_reviews_crew_tool(state)
    payload = tool.run(file_path=str(p))
    assert isinstance(payload, str)
    decoded = json.loads(payload)
    assert decoded == state["reviews"]
    assert decoded[0]["id"] == "a1"


def test_load_reviews_tool_second_call_uses_cache(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    _write_json(
        p,
        [{"id": "z", "text": "x", "rating": 3, "date": "2026-01-01", "verified": True}],
    )
    state: dict[str, object] = {"reviews": [], "analyzed": [], "scored": [], "fraud_scored": [], "report": ""}
    tool = create_load_reviews_crew_tool(state)
    first = json.loads(tool.run(file_path=str(p)))
    second = json.loads(tool.run(file_path=str(p)))
    assert first == second == state["reviews"]


@pytest.mark.integration
def test_full_crew_local_ollama_smoke() -> None:
    """
    End-to-end CrewAI run against a local Ollama server.

    Enable with ``RUN_OLLAMA_INTEGRATION=1`` and ensure ``ollama serve`` is running.
    """
    if os.environ.get("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set RUN_OLLAMA_INTEGRATION=1 to run local Ollama crew smoke test")

    try:
        import urllib.request

        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
    except OSError:
        pytest.skip("Ollama does not appear to be reachable on http://127.0.0.1:11434")

    from main import run_pipeline

    root = Path(__file__).resolve().parents[1]
    sample = root / "sample_reviews.json"
    state, output = run_pipeline(str(sample))
    assert isinstance(output, str) and output.strip()
    assert isinstance(state.get("reviews"), list)
    assert len(state["reviews"]) >= 1
    assert isinstance(state.get("analyzed"), list)
    assert isinstance(state.get("scored"), list)
    assert isinstance(state.get("fraud_scored"), list)
    assert isinstance(state.get("report"), str)
