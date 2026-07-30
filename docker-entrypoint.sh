#!/bin/sh
set -eu

cmd=${1:-}

if [ "$cmd" = "index" ]; then
  exec python -m langchain_ollama_rag.cli "$@"
fi

if [ "$cmd" = "ask" ]; then
  exec python -m langchain_ollama_rag.cli "$@"
fi

if [ "$cmd" = "retrieve" ]; then
  exec python -m langchain_ollama_rag.cli "$@"
fi

exec python -m uvicorn langchain_ollama_rag.api:app --host 0.0.0.0 --port 8000
