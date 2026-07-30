# debug_stream.py
import asyncio
from langchain_ollama_rag.app import create_rag_agent, create_rag_chain
from langchain_ollama_rag.config import Settings


async def main():
    settings = Settings.from_env()
    agent = create_rag_chain(settings)

    question = "What is this document about?"

    print("=" * 20, "stream_mode='values'", "=" * 20)
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        print(step)
        print("---")

    print("\n" + "=" * 20, "stream_mode='updates'", "=" * 20)
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        print(update)
        print("---")

    print("\n" + "=" * 20, "stream_mode='messages'", "=" * 20)
    async for chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        print(f"chunk.content={chunk.content!r}  metadata={metadata}")


if __name__ == "__main__":
    asyncio.run(main())