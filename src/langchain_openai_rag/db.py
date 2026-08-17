from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, Text, TIMESTAMP, create_engine, insert, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from langchain_core.documents import Document
from langchain_openai_rag.config import Settings

_metadata = MetaData()

ask_logs = Table(
    "ask_logs",
    _metadata,
    Column("id", String(36), primary_key=True),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("retrieved_chunks", JSONB, nullable=False),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    ),
)

_engine: Engine | None = None
_db_initialized = False


def get_engine(settings: Settings) -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, future=True)
    return _engine


def init_database(settings: Settings) -> None:
    global _db_initialized
    if _db_initialized:
        return

    engine = get_engine(settings)
    _metadata.create_all(engine)
    _db_initialized = True


def serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "start_index": doc.metadata.get("start_index", None),
            "content": doc.page_content,
        }
        for doc in documents
    ]


def log_question_answer(
    question: str,
    answer: str,
    retrieved_documents: list[Document],
    settings: Settings,
) -> None:
    if not settings.database_url:
        return

    init_database(settings)
    engine = get_engine(settings)
    values = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "retrieved_chunks": serialize_documents(retrieved_documents),
    }

    with engine.begin() as conn:
        conn.execute(insert(ask_logs).values(**values))


def fetch_ask_logs(settings: Settings, limit: int = 100) -> list[dict[str, Any]]:
    if not settings.database_url:
        return []

    engine = get_engine(settings)
    with engine.begin() as conn:
        result = conn.execute(
            select(ask_logs)
            .order_by(ask_logs.c.created_at.desc())
            .limit(limit)
        )
        return [dict(row._mapping) for row in result]
