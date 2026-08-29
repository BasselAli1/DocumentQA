from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai_rag.config import Settings
from langchain_openai_rag.loaders import load_demo_article, load_uploaded_document


def create_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store(settings: Settings) -> Chroma:
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=create_embeddings(settings),
        persist_directory=str(settings.persist_dir),
    )


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def index_documents(documents: list[Document], settings: Settings, *, reset: bool = True) -> int:
    """Split, embed, and store an arbitrary list of already-loaded documents."""
    if reset:
        reset_index(settings.persist_dir)

    splits = split_documents(documents, settings)
    vector_store = get_vector_store(settings)
    vector_store.add_documents(documents=splits, ids=[document_id(doc) for doc in splits])
    return len(splits)


def index_uploaded_file(
    file_path: Path,
    original_filename: str,
    settings: Settings,
    *,
    reset: bool = True,
) -> int:
    """Load a single user-uploaded file (pdf/docx/html/txt/md) and index it."""
    documents = load_uploaded_document(file_path, original_filename=original_filename)
    return index_documents(documents, settings, reset=reset)


def index_source(settings: Settings, *, reset: bool = True) -> int:
    """Kept for convenience/demo purposes: indexes the built-in demo web article."""
    documents = load_demo_article(settings.source_url)
    return index_documents(documents, settings, reset=reset)


def reset_index(persist_dir: Path) -> None:
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


def document_id(document: Document) -> str:
    source = document.metadata.get("source", "")
    start_index = document.metadata.get("start_index", "")
    digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{start_index}:{digest}"
