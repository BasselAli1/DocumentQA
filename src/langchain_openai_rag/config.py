from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


DEFAULT_SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# Kept as a tuple (not a set) so the error message in api.py/loaders.py can
# join it in a stable, predictable order.
SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".docx", ".html", ".htm", ".txt", ".md")


@dataclass(frozen=True)
class Settings:
    openai_chat_model: str
    openai_embedding_model: str
    openai_api_key: str
    collection_name: str
    persist_dir: Path
    upload_dir: Path
    source_url: str
    database_url: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(override=False)
        return cls(
            openai_chat_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "rag_tutorial"),
            persist_dir=Path(os.getenv("RAG_PERSIST_DIR", ".rag/chroma")),
            upload_dir=Path(os.getenv("RAG_UPLOAD_DIR", ".rag/uploads")),
            source_url=os.getenv("RAG_SOURCE_URL", DEFAULT_SOURCE_URL),
            database_url=os.getenv("DATABASE_URL", ""),
            chunk_size=_get_positive_int("RAG_CHUNK_SIZE", 1000),
            chunk_overlap=_get_non_negative_int("RAG_CHUNK_OVERLAP", 200),
            retrieval_k=_get_positive_int("RAG_RETRIEVAL_K", 4),
        )


def _get_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _get_non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")
    return value
