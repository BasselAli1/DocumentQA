from __future__ import annotations

from pathlib import Path

import pytest

from langchain_openai_rag.config import DEFAULT_SOURCE_URL, Settings


def test_defaults_are_not_tutorial_flavored():
    settings = Settings.from_env()

    assert settings.collection_name == "documentqa"
    assert "tutorial" not in settings.collection_name


def test_default_numeric_settings():
    settings = Settings.from_env()

    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_k == 4
    assert settings.source_url == DEFAULT_SOURCE_URL
    assert settings.persist_dir == Path(".rag/chroma")


def test_env_overrides_are_applied(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_COLLECTION_NAME", "my_docs")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "512")
    monkeypatch.setenv("RAG_RETRIEVAL_K", "8")
    monkeypatch.setenv("RAG_SOURCE_URL", "https://example.com/article")

    settings = Settings.from_env()

    assert settings.collection_name == "my_docs"
    assert settings.chunk_size == 512
    assert settings.retrieval_k == 8
    assert settings.source_url == "https://example.com/article"


@pytest.mark.parametrize("value", ["0", "-5"])
def test_non_positive_chunk_size_rejected(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("RAG_CHUNK_SIZE", value)

    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE"):
        Settings.from_env()


def test_negative_chunk_overlap_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "-1")

    with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
        Settings.from_env()


def test_zero_chunk_overlap_is_allowed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "0")

    assert Settings.from_env().chunk_overlap == 0


def test_settings_is_immutable(make_settings):
    from dataclasses import FrozenInstanceError

    settings = make_settings()

    with pytest.raises(FrozenInstanceError):
        settings.chunk_size = 1  # type: ignore[misc]
