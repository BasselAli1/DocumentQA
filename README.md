# DocumentQA (LangChain RAG)

[![CI](https://github.com/BasselAli1/DocumentQA/actions/workflows/ci.yml/badge.svg)](https://github.com/BasselAli1/DocumentQA/actions/workflows/ci.yml)

A Retrieval-Augmented Generation (RAG) service for asking questions over your own documents. Upload a PDF, DOCX, HTML, TXT, or Markdown file, and ask questions about it through a web UI, a REST API, or a CLI. Answers stream back token by token and are grounded in the retrieved chunks, with every Q&A logged to Postgres.

Chat and embeddings are both served by [OpenAI](https://platform.openai.com).

## Features

- **Document ingestion**: index `.pdf`, `.docx`, `.html`/`.htm`, `.txt`, `.md` files, or a demo web article, with automatic chunking and embedding.
- **Vector search**: [Chroma](https://www.trychroma.com/) as the vector store, persisted to disk.
- **Two answering strategies**: `agent` mode (the model decides when to call a retrieval tool) and `chain` mode (retrieval always runs before the model responds).
- **Streaming answers**: Server-Sent Events (SSE) endpoint streams tokens as they're generated.
- **Q&A logging**: every question, answer, and the chunks used to answer it are persisted to Postgres.
- **Web UI, REST API, and CLI**: use whichever fits your workflow.

## Architecture

```
static/index.html  →  FastAPI (api.py)  →  app.py (agent/chain + streaming)
                                          →  indexing.py (chunk + embed)  →  Chroma (.rag/chroma)
                                          →  db.py (ask_logs)             →  Postgres
CLI (cli.py) drives the same app.py / indexing.py functions for headless use.
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- A running Postgres instance (used to log Q&A history)
- An [OpenAI](https://platform.openai.com) API key (used for both chat and embeddings)

## Setup

1. **Clone and install dependencies**

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   pip install -e .
   ```

2. **Configure environment variables**

   Create a `.env` file in the project root (see [Configuration](#configuration) below for all options):

   ```bash
   OPENAI_API_KEY=
   OPENAI_MODEL=gpt-5.6-luna
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small

   DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
   ```

## Running

### With Docker Compose (recommended)

Spins up the app and a Postgres database together:

```bash
docker compose up --build
```

The app is available at [http://localhost:8000](http://localhost:8000). Set `OPENAI_API_KEY` in your shell (or an `.env` file Compose picks up) before running this. `docker-compose.yml` passes it through via `${OPENAI_API_KEY}`.

### Locally

Start Postgres yourself (or point `DATABASE_URL` at an existing instance), then run the API server:

```bash
uv run rag-api
```

or directly with uvicorn (with autoreload):

```bash
uv run uvicorn langchain_openai_rag.api:app --reload
```

Open [http://localhost:8000](http://localhost:8000) to use the web UI.

## Usage

### Web UI

Visit `http://localhost:8000`, upload a document, and start asking questions.

### REST API

**Upload and index a document**

```bash
curl -X POST http://localhost:8000/api/documents \
  -F "file=@/path/to/document.pdf" \
  -F "reset=true"
```

`reset=true` (default) clears the existing index before adding the new document. Set it to `false` to add to the existing index instead.

**Ask a question (streams Server-Sent Events)**

```bash
curl -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### CLI

The `rag` command wraps the same indexing and Q&A logic for headless use:

```bash
# Index a local file (clears the existing index first)
uv run rag index /path/to/document.pdf

# Index without clearing the existing index
uv run rag index /path/to/document.pdf --reset=false

# No path given -> indexes the built-in demo article instead
uv run rag index

# Ask a question ("chain" or "agent" mode)
uv run rag ask "What is this document about?" --mode chain

# Inspect the raw chunks retrieved for a query
uv run rag retrieve "search query"
```

## Configuration

All settings are read from environment variables (loaded from `.env` via `python-dotenv`), defined in [`config.py`](src/langchain_openai_rag/config.py):

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_MODEL` | `gpt-5.6-luna` | OpenAI chat model name. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model name. |
| `OPENAI_API_KEY` | _(empty)_ | API key for OpenAI. |
| `DATABASE_URL` | _(empty)_ | Postgres connection string used to log Q&A history. |
| `RAG_COLLECTION_NAME` | `documentqa` | Chroma collection name. |
| `RAG_PERSIST_DIR` | `.rag/chroma` | Directory where the Chroma index is persisted. |
| `RAG_UPLOAD_DIR` | `.rag/uploads` | Directory for uploaded files (if retained). |
| `RAG_SOURCE_URL` | _(built-in demo article)_ | Web article indexed when running `index` without a file. |
| `RAG_CHUNK_SIZE` | `1000` | Character length of each document chunk. |
| `RAG_CHUNK_OVERLAP` | `200` | Character overlap between consecutive chunks. |
| `RAG_RETRIEVAL_K` | `4` | Number of chunks retrieved per query. |

## Project structure

```
src/langchain_openai_rag/
├── api.py            FastAPI app: upload, ask (SSE), serves the web UI
├── app.py             Agent/chain construction, ask/ask_stream/retrieve logic
├── cli.py             `rag index|ask|retrieve` command-line entry point
├── config.py          Settings loaded from environment variables
├── db.py               Postgres logging of questions, answers, and retrieved chunks
├── indexing.py         Chunking, embedding, and Chroma vector store management
└── loaders.py           Document loaders for PDF, DOCX, HTML, TXT/MD, and web pages
static/index.html      Web UI
tests/                 Offline pytest suite
.github/workflows/ci.yml   Lint + test + Docker build pipeline
Dockerfile, docker-compose.yml, docker-entrypoint.sh   Containerized deployment
```

## Testing

```bash
uv sync --extra dev      # installs pytest + ruff
uv run ruff check .
uv run pytest            # 31 tests, no network / OpenAI / Postgres needed
```

The suite runs fully offline, using temp files and hand-built `Settings`:

- `test_config.py`: environment parsing, the documented defaults, and the
  positive / non-negative integer validation for chunk size and overlap.
- `test_loaders.py`: per-type document loading (`.txt`, `.md`, `.html`),
  HTML markup stripping, and the unsupported-type and empty-file errors.
- `test_indexing.py`: deterministic chunk IDs (stable across re-indexing)
  and `chunk_size` / `chunk_overlap` splitting behaviour.
- `test_cli.py`: the `index` / `ask` / `retrieve` argument parser.
- `test_db.py`: retrieved-chunk serialization, and that logging is a no-op
  when `DATABASE_URL` is unset.
- `test_app.py`: context serialization passed to the model.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff and
these tests on Python 3.11 and 3.12, then builds the Docker image, on every
push and pull request.

## License

MIT
