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

# Extensions we know how to parse when a user uploads a document.
SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".docx", ".html", ".htm", ".txt", ".md")


@dataclass(frozen=True)
class Settings:
    chat_model: str
    embedding_model: str
    online_chat_model: str
    online_embedding_model: str
    api_key: str
    mode: str
    ollama_base_url: str
    openrouter_base_url: str
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
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "snowflake-arctic-embed:22m"),
            online_chat_model=os.getenv("OPENROUTER_CHAT_MODEL", ""),
            online_embedding_model=os.getenv("OPENROUTER_EMBEDDING_MODEL", ""), 
            api_key=os.getenv("OPENROUTER_API_KEY", ""), 
            mode=os.getenv("MODE", "online"),        
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", ""),
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
