from __future__ import annotations

import argparse
from pathlib import Path

from langchain_openai_rag.app import ask, retrieve
from langchain_openai_rag.config import Settings
from langchain_openai_rag.indexing import index_source, index_uploaded_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a document (or the demo web page).")
    index_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a local file (.pdf, .docx, .html, .txt, .md). "
        "If omitted, indexes the built-in demo article instead.",
    )
    index_parser.add_argument(
        "--reset", action="store_true", help="Clear the existing index first."
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=["agent", "chain"], default="chain")

    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Show raw retrieved chunks for a query."
    )
    retrieve_parser.add_argument("query")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "index":
        if args.path:
            file_path = Path(args.path)
            if not file_path.exists():
                raise SystemExit(f"File not found: {file_path}")
            chunks = index_uploaded_file(file_path, file_path.name, settings, reset=args.reset)
            print(f"Indexed {chunks} chunk(s) from '{file_path.name}'.")
        else:
            chunks = index_source(settings, reset=args.reset)
            print(f"Indexed {chunks} chunk(s) from the demo article.")

    elif args.command == "ask":
        answer = ask(args.question, settings, mode=args.mode)
        print(answer)

    elif args.command == "retrieve":
        docs = retrieve(args.query, settings)
        for doc in docs:
            print(f"--- {doc.metadata} ---")
            print(doc.page_content[:500])
            print()


if __name__ == "__main__":
    main()
