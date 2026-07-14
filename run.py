#!/usr/bin/env python3
"""
RAG Knowledge Agent CLI.

Usage:
    python run.py                          # Interactive mode
    python run.py --query "你的问题"        # Single query
    python run.py --ingest data/documents  # Ingest documents
    python run.py --api                    # Start REST API server
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from agent.core.agent import RAGAgent


async def interactive_mode(agent: RAGAgent):
    """Run the agent in interactive REPL mode."""
    print("=" * 50)
    print("  RAG Knowledge Agent - Interactive Mode")
    print("  Type 'exit' or 'quit' to leave.")
    print("  Type '!ingest <path>' to ingest a document.")
    print("=" * 50)

    session_id = None

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if query.startswith("!ingest"):
            path = query[len("!ingest"):].strip()
            if path:
                print(f"Ingesting: {path}")
                count = await agent.ingest(path)
                print(f"Ingested {count} chunks.")
            continue

        if query.startswith("!count"):
            count = await agent.memory.count_documents() if agent.memory else 0
            print(f"Documents in knowledge base: {count}")
            continue

        if session_id is None:
            session_id = "session_default"

        print("Agent: ", end="", flush=True)
        try:
            state = await agent.ask(query, session_id=session_id)
            if state.error:
                print(f"\n[Error] {state.error}")
            else:
                print(state.answer)
                if state.citations:
                    print("\n--- Sources ---")
                    for i, c in enumerate(state.citations):
                        print(f"  [{i+1}] {c.source}")
        except Exception as e:
            print(f"\n[Error] {e}")


async def single_query(agent: RAGAgent, query: str):
    """Run a single query and print the result."""
    state = await agent.ask(query)
    if state.error:
        print(f"[Error] {state.error}", file=sys.stderr)
        sys.exit(1)
    print(state.answer)
    if state.citations:
        print("\n--- Sources ---")
        for i, c in enumerate(state.citations):
            print(f"  [{i+1}] {c.source}")


async def ingest_docs(agent: RAGAgent, path: str):
    """Ingest documents from a file or directory."""
    p = Path(path)
    if not p.exists():
        print(f"[Error] Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    if p.is_file():
        print(f"Ingesting: {p}")
        count = await agent.ingest(str(p))
        print(f"Done. Created {count} chunks from {p.name}.")
    elif p.is_dir():
        print(f"Ingesting all documents in: {p}")
        results = await agent.ingest_directory(str(p))
        total_chunks = sum(v for v in results.values() if v > 0)
        errors = sum(1 for v in results.values() if v < 0)
        print(f"Done. Created {total_chunks} chunks from {len(results)} files "
              f"({errors} errors).")
        if errors:
            for f, count in results.items():
                if count < 0:
                    print(f"  Failed: {f}")


async def main():
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Agent - Private knowledge base Q&A",
    )
    parser.add_argument(
        "--query", "-q",
        help="Ask a single question (non-interactive mode)",
    )
    parser.add_argument(
        "--ingest",
        help="Ingest a document file or directory",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Stream the response token by token",
    )

    args = parser.parse_args()

    # Initialize the agent
    agent = RAGAgent(config_path=args.config)

    try:
        await agent.initialize()
    except Exception as e:
        print(f"[Error] Failed to initialize agent: {e}", file=sys.stderr)
        print("Make sure Ollama is running. Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

    if args.ingest:
        await ingest_docs(agent, args.ingest)
    elif args.query:
        if args.stream:
            async for token in agent.ask_stream(args.query):
                print(token, end="", flush=True)
            print()
        else:
            await single_query(agent, args.query)
    else:
        await interactive_mode(agent)


if __name__ == "__main__":
    asyncio.run(main())
