# DocumentQA (LangChain Ollama RAG)

A Retrieval-Augmented Generation (RAG) service for asking questions over your own documents. Upload a PDF, DOCX, HTML, TXT, or Markdown file, and ask questions about it through a web UI, a REST API, or a CLI. Answers stream back token by token and are grounded in the retrieved chunks, with every Q&A logged to Postgres.

It can run fully **offline** with local [Ollama](https://ollama.com) models, or **online** using [OpenRouter](https://openrouter.ai) for chat and NVIDIA NIM embeddings — controlled by a single `MODE` setting.

## Features

- **Document ingestion** — index `.pdf`, `.docx`, `.html`/`.htm`, `.txt`, `.md` files, or a demo web article, with automatic chunking and embedding.
- **Vector search** — [Chroma](https://www.trychroma.com/) as the vector store, persisted to disk.
- **Two answering strategies** — `agent` mode (the model decides when to call a retrieval tool) and `chain` mode (retrieval always runs before the model responds).
- **Streaming answers** — Server-Sent Events (SSE) endpoint streams tokens as they're generated.
- **Q&A logging** — every question, answer, and the chunks used to answer it are persisted to Postgres.
- **Web UI, REST API, and CLI** — use whichever fits your workflow.
- **Local or cloud models** — swap between Ollama (offline) and OpenRouter/NVIDIA (online) via one environment variable.

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
- Either:
  - [Ollama](https://ollama.com) running locally with a chat model and an embedding model pulled, **or**
  - An [OpenRouter](https://openrouter.ai) API key (for chat) and NVIDIA NIM access (for embeddings) for online mode

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
   # Choose "offline" (Ollama) or "online" (OpenRouter + NVIDIA embeddings)
   MODE=offline

   # Offline mode (Ollama)
   OLLAMA_CHAT_MODEL=llama3.2:1b
   OLLAMA_EMBEDDING_MODEL=snowflake-arctic-embed:22m
   OLLAMA_BASE_URL=http://127.0.0.1:11434

   # Online mode (OpenRouter + NVIDIA)
   OPENROUTER_API_KEY=
   OPENROUTER_CHAT_MODEL=
   OPENROUTER_EMBEDDING_MODEL=
   OPENROUTER_BASE_URL=

   # Required in both modes
   DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
   ```

3. **If running offline**, make sure Ollama is running and the models are pulled:

   ```bash
   ollama pull llama3.2:1b
   ollama pull snowflake-arctic-embed:22m
   ```

## Running

### With Docker Compose (recommended)

Spins up the app and a Postgres database together:

```bash
docker compose up --build
```

The app is available at [http://localhost:8000](http://localhost:8000). By default `docker-compose.yml` is configured for offline mode against `host.docker.internal:11434` — update the `environment` block for online mode instead.

### Locally

Start Postgres yourself (or point `DATABASE_URL` at an existing instance), then run the API server:

```bash
uv run rag-api
```

or directly with uvicorn (with autoreload):

```bash
uv run uvicorn langchain_ollama_rag.api:app --reload
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

`reset=true` (default) clears the existing index before adding the new document; set it to `false` to add to the existing index instead.

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

All settings are read from environment variables (loaded from `.env` via `python-dotenv`), defined in [`config.py`](src/langchain_ollama_rag/config.py):

| Variable | Default | Description |
| --- | --- | --- |
| `MODE` | `online` | `online` uses OpenRouter/NVIDIA; anything else uses local Ollama. |
| `OLLAMA_CHAT_MODEL` | `llama3.2:1b` | Ollama chat model name (offline mode). |
| `OLLAMA_EMBEDDING_MODEL` | `snowflake-arctic-embed:22m` | Ollama embedding model name (offline mode). |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL. |
| `OPENROUTER_CHAT_MODEL` | _(empty)_ | Chat model name for OpenRouter (online mode). |
| `OPENROUTER_EMBEDDING_MODEL` | _(empty)_ | Embedding model name for NVIDIA NIM (online mode). |
| `OPENROUTER_API_KEY` | _(empty)_ | API key for OpenRouter / NVIDIA endpoints. |
| `OPENROUTER_BASE_URL` | _(empty)_ | Optional override for the OpenRouter base URL. |
| `DATABASE_URL` | _(empty)_ | Postgres connection string used to log Q&A history. |
| `RAG_COLLECTION_NAME` | `rag_tutorial` | Chroma collection name. |
| `RAG_PERSIST_DIR` | `.rag/chroma` | Directory where the Chroma index is persisted. |
| `RAG_UPLOAD_DIR` | `.rag/uploads` | Directory for uploaded files (if retained). |
| `RAG_SOURCE_URL` | Lilian Weng's agents post | Demo article used when indexing without a file. |
| `RAG_CHUNK_SIZE` | `1000` | Character length of each document chunk. |
| `RAG_CHUNK_OVERLAP` | `200` | Character overlap between consecutive chunks. |
| `RAG_RETRIEVAL_K` | `4` | Number of chunks retrieved per query. |

## Project structure

```
src/langchain_ollama_rag/
├── api.py            FastAPI app: upload, ask (SSE), serves the web UI
├── app.py             Agent/chain construction, ask/ask_stream/retrieve logic
├── cli.py             `rag index|ask|retrieve` command-line entry point
├── config.py          Settings loaded from environment variables
├── db.py               Postgres logging of questions, answers, and retrieved chunks
├── indexing.py         Chunking, embedding, and Chroma vector store management
└── loaders.py           Document loaders for PDF, DOCX, HTML, TXT/MD, and web pages
static/index.html      Web UI
Dockerfile, docker-compose.yml, docker-entrypoint.sh   Containerized deployment
```

## Development

```bash
# Install dev dependencies (pytest, ruff)
uv sync --extra dev

# Lint
uv run ruff check .

# Run tests
uv run pytest
```

## License

MIT
