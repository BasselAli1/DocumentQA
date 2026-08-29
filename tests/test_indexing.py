from __future__ import annotations

from langchain_core.documents import Document

from langchain_openai_rag.indexing import document_id, split_documents


def test_document_id_is_deterministic():
    doc = Document(page_content="some content", metadata={"source": "a.txt", "start_index": 0})

    assert document_id(doc) == document_id(doc)


def test_document_id_changes_with_content():
    a = Document(page_content="content A", metadata={"source": "a.txt", "start_index": 0})
    b = Document(page_content="content B", metadata={"source": "a.txt", "start_index": 0})

    assert document_id(a) != document_id(b)


def test_document_id_includes_source_and_start_index():
    doc = Document(page_content="x", metadata={"source": "report.pdf", "start_index": 42})

    assert document_id(doc).startswith("report.pdf:42:")


def test_split_documents_respects_chunk_size(make_settings):
    settings = make_settings(chunk_size=100, chunk_overlap=10)
    long_text = "word " * 400  # ~2000 chars
    docs = [Document(page_content=long_text, metadata={"source": "big.txt"})]

    splits = split_documents(docs, settings)

    assert len(splits) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in splits)
    assert all("start_index" in chunk.metadata for chunk in splits)


def test_split_documents_short_input_stays_single(make_settings):
    settings = make_settings(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content="short body", metadata={"source": "s.txt"})]

    splits = split_documents(docs, settings)

    assert len(splits) == 1
    assert splits[0].page_content == "short body"
