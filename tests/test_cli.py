from __future__ import annotations

import pytest

from langchain_openai_rag.cli import build_parser


def test_index_without_path_defaults_to_none():
    args = build_parser().parse_args(["index"])

    assert args.command == "index"
    assert args.path is None
    assert args.reset is False


def test_index_with_path_and_reset_flag():
    args = build_parser().parse_args(["index", "doc.pdf", "--reset"])

    assert args.path == "doc.pdf"
    assert args.reset is True


def test_ask_mode_choices():
    args = build_parser().parse_args(["ask", "what is this?", "--mode", "agent"])

    assert args.question == "what is this?"
    assert args.mode == "agent"

    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "q", "--mode", "bogus"])


def test_ask_defaults_to_chain_mode():
    args = build_parser().parse_args(["ask", "q"])

    assert args.mode == "chain"


def test_retrieve_requires_query():
    args = build_parser().parse_args(["retrieve", "search terms"])
    assert args.query == "search terms"

    with pytest.raises(SystemExit):
        build_parser().parse_args(["retrieve"])


def test_missing_command_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
