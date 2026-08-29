from __future__ import annotations

from pathlib import Path

import pytest

from langchain_openai_rag import loaders
from langchain_openai_rag.loaders import load_uploaded_document


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_txt_file(tmp_path: Path):
    path = _write(tmp_path, "notes.txt", "hello world")

    docs = load_uploaded_document(path, original_filename="notes.txt")

    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "notes.txt"


def test_load_markdown_file(tmp_path: Path):
    path = _write(tmp_path, "readme.md", "# Title\n\nBody text.")

    docs = load_uploaded_document(path, original_filename="readme.md")

    assert "Body text." in docs[0].page_content


def test_load_html_file_strips_markup(tmp_path: Path):
    path = _write(
        tmp_path,
        "page.html",
        "<html><body><h1>Heading</h1><p>Paragraph</p><script>ignored()</script></body></html>",
    )

    docs = load_uploaded_document(path, original_filename="page.html")
    text = docs[0].page_content

    assert "Heading" in text
    assert "Paragraph" in text
    assert "<h1>" not in text


def test_unsupported_extension_raises(tmp_path: Path):
    path = _write(tmp_path, "data.xyz", "whatever")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded_document(path, original_filename="data.xyz")


def test_empty_text_file_raises(tmp_path: Path):
    path = _write(tmp_path, "empty.txt", "   \n  ")

    with pytest.raises(ValueError, match="empty"):
        load_uploaded_document(path, original_filename="empty.txt")


def test_original_filename_drives_type_detection(tmp_path: Path):
    # File on disk has a temp-style name; the original filename is what matters.
    path = _write(tmp_path, "tmpabcd", "plain text body")

    docs = load_uploaded_document(path, original_filename="upload.txt")

    assert docs[0].metadata["source"] == "upload.txt"


def test_demo_article_loader_exists_and_is_renamed():
    assert hasattr(loaders, "load_demo_article")
    assert not hasattr(loaders, "load_tutorial_source")
