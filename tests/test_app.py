from __future__ import annotations

from langchain_core.documents import Document

from langchain_openai_rag.app import serialize_documents


def test_serialize_documents_includes_source_and_content():
    docs = [
        Document(page_content="alpha", metadata={"source": "a.txt"}),
        Document(page_content="beta", metadata={"source": "b.txt"}),
    ]

    serialized = serialize_documents(docs)

    assert "Content: alpha" in serialized
    assert "Content: beta" in serialized
    assert "a.txt" in serialized


def test_serialize_documents_empty():
    assert serialize_documents([]) == ""
