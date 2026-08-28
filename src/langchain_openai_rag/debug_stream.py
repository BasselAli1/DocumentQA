"""Standalone script to inspect what each LangGraph streaming mode actually
produces, with clean/summarized output instead of raw object dumps.

Usage:
    1. Place this file at the root of your project (next to pyproject.toml),
       or anywhere on your PYTHONPATH alongside the installed package.
    2. Make sure you've already indexed at least one document (via the UI,
       `rag index <path>`, or the /api/documents endpoint).
    3. Make sure OPENAI_API_KEY is set (in your .env or the shell environment).
    4. Run:
         python debug_stream.py
         python debug_stream.py "What is this document about?"
"""

from __future__ import annotations

import asyncio
import sys

from langchain_openai_rag.app import create_rag_chain
from langchain_openai_rag.config import Settings

SEPARATOR = "-" * 60


def summarize_message(msg) -> str:
    """One readable line per message: type, short content, tool info."""
    kind = type(msg).__name__
    content = (getattr(msg, "content", "") or "")
    if isinstance(content, str):
        content_preview = content[:80] + ("..." if len(content) > 80 else "")
    else:
        content_preview = str(content)[:80]

    extra = ""
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        names = [tc.get("name", "?") for tc in tool_calls]
        extra += f"  tool_calls={names}"

    artifact = getattr(msg, "artifact", None)
    if artifact:
        extra += f"  artifact_docs={len(artifact)}"
        for i, doc in enumerate(artifact):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            page_str = f" p.{page}" if page is not None else ""
            extra += f"\n        [{i}] source={source}{page_str}"

    return f"{kind:<15} content={content_preview!r}{extra}"


async def show_values(agent, question: str) -> None:
    print(f"\n{SEPARATOR}\nstream_mode='values'  (full state snapshot per step)\n{SEPARATOR}")
    step_num = 0
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step_num += 1
        messages = step.get("messages", [])
        context = step.get("context", [])
        print(f"\n[snapshot {step_num}] total messages so far: {len(messages)}")
        if messages:
            print("   last message:", summarize_message(messages[-1]))
        if context:
            print(f"context has {len(context)} doc(s)")

async def show_updates(agent, question: str) -> None:
    print(f"\n{SEPARATOR}\nstream_mode='updates'  (diff per node execution)\n{SEPARATOR}")
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        for node_name, changes in update.items():
            print(f"\n[node: {node_name}]")
            for msg in changes.get("messages", []):
                print("   ", summarize_message(msg))
                if changes.get("context"):
                    print(f"   context has {len(changes['context'])} doc(s)")

async def show_messages(agent, question: str) -> None:
    print(f"\n{SEPARATOR}\nstream_mode='messages'  (token-by-token)\n{SEPARATOR}")
    token_count = 0
    buffer = ""
    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        token = getattr(chunk, "content", "")
        if token:
            token_count += 1
            buffer += token
            print(f"  token {token_count:>3}: {token!r}")

    print(f"\nFull streamed answer ({token_count} tokens):\n  {buffer!r}")


async def show_events(agent, question: str) -> None:
    print(f"\n{SEPARATOR}\nastream_events  (fine-grained event log)\n{SEPARATOR}")
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": question}]},
        version="v2",
    ):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chat_model_stream":
            token = getattr(event["data"]["chunk"], "content", "")
            if token:
                print(f"  [on_chat_model_stream] token={token!r}")
        elif kind in ("on_tool_start", "on_tool_end"):
            print(f"  [{kind}] name={name}")
            if kind == "on_tool_end":
                output = event["data"].get("output")
                artifact = getattr(output, "artifact", None)
                if artifact:
                    print(f"      -> {len(artifact)} document(s) retrieved")
        # Uncomment to see every single event, including the noisy ones:
        # else:
        #     print(f"  [{kind}] name={name}")


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    print(f"Question: {question!r}")

    settings = Settings.from_env()
    agent = create_rag_chain(settings)

    await show_values(agent, question)
    await show_updates(agent, question)
    await show_messages(agent, question)
    await show_events(agent, question)


if __name__ == "__main__":
    asyncio.run(main())
