from __future__ import annotations

from langchain_core.documents import Document

from langchain_openai_rag import db


def test_serialize_documents_shape():
    docs = [
        Document(page_content="chunk one", metadata={"source": "a.txt", "start_index": 0}),
        Document(page_content="chunk two", metadata={}),
    ]

    rows = db.serialize_documents(docs)

    assert rows[0] == {"source": "a.txt", "start_index": 0, "content": "chunk one"}
    assert rows[1] == {"source": "unknown", "start_index": None, "content": "chunk two"}


def test_log_question_answer_is_noop_without_database_url(make_settings, monkeypatch):
    settings = make_settings(database_url="")

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("get_engine should not be called without DATABASE_URL")

    monkeypatch.setattr(db, "get_engine", _boom)

    # Should simply return without attempting any DB connection.
    assert db.log_question_answer("q", "a", [], settings) is None


def test_fetch_ask_logs_returns_empty_without_database_url(make_settings):
    assert db.fetch_ask_logs(make_settings(database_url="")) == []
