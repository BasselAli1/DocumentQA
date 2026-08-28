from __future__ import annotations

from pathlib import Path

import pytest

from langchain_openai_rag.config import DEFAULT_SOURCE_URL, Settings

# Environment variables that Settings.from_env() reads. Cleared before every
# test so a developer's real .env / shell can't leak into assertions.
_RAG_ENV_VARS = (
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "RAG_COLLECTION_NAME",
    "RAG_PERSIST_DIR",
    "RAG_UPLOAD_DIR",
    "RAG_SOURCE_URL",
    "DATABASE_URL",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_RETRIEVAL_K",
)


@pytest.fixture(autouse=True)
def _clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _RAG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_settings(tmp_path: Path):
    """Factory for a fully-populated Settings without touching the environment."""

    def _make(**overrides) -> Settings:
        defaults = dict(
            openai_chat_model="test-chat-model",
            openai_embedding_model="test-embedding-model",
            openai_api_key="test-key",
            collection_name="documentqa",
            persist_dir=tmp_path / "chroma",
            upload_dir=tmp_path / "uploads",
            source_url=DEFAULT_SOURCE_URL,
            database_url="",
            chunk_size=200,
            chunk_overlap=20,
            retrieval_k=4,
        )
        defaults.update(overrides)
        return Settings(**defaults)

    return _make
