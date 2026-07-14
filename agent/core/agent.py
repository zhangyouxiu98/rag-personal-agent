"""RAGAgent: Main orchestrator for the RAG knowledge QA pipeline."""

import asyncio
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import yaml
from loguru import logger

from agent.core.state import (
    AgentState,
    PipelineStage,
    StateMachine,
    Citation,
)
from agent.core.memory import MemoryManager
from agent.core.tools import ToolRegistry
from agent.memory.base import BaseVectorStore, BaseSessionMemory
from agent.memory.vector_memory import ChromaVectorStore, OllamaEmbeddingFunction
from agent.memory.session_memory import SessionMemory
from agent.skills.search_skill import DocumentRetrieverSkill
from agent.skills.text_chunker import TextChunker


class LLMClient:
    """Lightweight wrapper around the OpenAI-compatible Ollama chat API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        import openai
        self.client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a complete response from the LLM."""
        loop = asyncio.get_event_loop()

        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content

        return await loop.run_in_executor(None, _call)

    async def generate_stream(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM."""
        loop = asyncio.get_event_loop()

        def _stream():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        for token in await loop.run_in_executor(None, list, _stream()):
            yield token


class RAGAgent:
    """RAG Knowledge Q&A Agent.

    Orchestrates the full RAG pipeline:
    analyze → retrieve → generate → validate.

    Usage:
        agent = RAGAgent()
        await agent.initialize()
        state = await agent.ask("你的问题")
        print(state.answer)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self._config_path = Path(config_path)
        self._config = self._load_config()
        self._state_machine = StateMachine()

        # Components (initialized in initialize())
        self.llm: Optional[LLMClient] = None
        self.vector_store: Optional[BaseVectorStore] = None
        self.session_memory: Optional[BaseSessionMemory] = None
        self.memory: Optional[MemoryManager] = None
        self.retriever: Optional[DocumentRetrieverSkill] = None
        self.tools: Optional[ToolRegistry] = None
        self._system_prompt: str = ""

    # ---- Lifecycle ----

    def _load_config(self) -> dict:
        """Load YAML configuration."""
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_dotenv():
        """Load .env file if python-dotenv is available."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    async def initialize(self):
        """Initialize all components. Must be called before ask()."""
        logger.info("Initializing RAG Agent...")

        # Determine provider: "ollama" (default) or "deepseek"
        provider = self._config.get("provider", "ollama")

        if provider == "deepseek":
            # ---- DeepSeek API for chat ----
            import os
            self._load_dotenv()

            ds_cfg = self._config.get("deepseek", {})
            ds_api_key = os.environ.get("DEEPSEEK_API_KEY", ds_cfg.get("api_key", ""))
            if not ds_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY not set. Set it via environment variable "
                    "or in config.yaml under deepseek.api_key"
                )
            self.llm = LLMClient(
                base_url=ds_cfg.get("base_url", "https://api.deepseek.com"),
                model=ds_cfg.get("model", "deepseek-chat"),
                api_key=ds_api_key,
                temperature=ds_cfg.get("temperature", 0.1),
                max_tokens=ds_cfg.get("max_tokens", 2048),
                timeout=ds_cfg.get("request_timeout", 60),
                max_retries=ds_cfg.get("max_retries", 3),
            )
            logger.info(f"Using DeepSeek API: {ds_cfg.get('model', 'deepseek-chat')}")
        else:
            # ---- Ollama local for chat ----
            ollama_cfg = self._config.get("ollama", {})
            self.llm = LLMClient(
                base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
                model=ollama_cfg.get("chat_model", "qwen2.5:3b"),
                api_key="ollama",
                temperature=ollama_cfg.get("temperature", 0.1),
                max_tokens=ollama_cfg.get("max_tokens", 2048),
                timeout=ollama_cfg.get("request_timeout", 60),
                max_retries=ollama_cfg.get("max_retries", 3),
            )
            logger.info(f"Using Ollama: {ollama_cfg.get('chat_model', 'qwen2.5:3b')}")

        # Vector store — always uses Ollama for embeddings
        chroma_cfg = self._config["chroma"]
        ollama_cfg = self._config.get("ollama", {})
        embedding_fn = OllamaEmbeddingFunction(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            model=ollama_cfg.get("embedding_model", "nomic-embed-text"),
        )
        self.vector_store = ChromaVectorStore(
            persist_directory=chroma_cfg["persist_directory"],
            collection_name=chroma_cfg["collection_name"],
            embedding_fn=embedding_fn,
        )
        await self.vector_store.initialize()

        # Session memory
        mem_cfg = self._config.get("memory", {})
        self.session_memory = SessionMemory(
            max_history=mem_cfg.get("max_session_history", 20),
            ttl_minutes=mem_cfg.get("session_ttl_minutes", 1440),
        )

        # Memory manager
        self.memory = MemoryManager(
            session_memory=self.session_memory,
            vector_store=self.vector_store,
        )

        # Document retriever skill
        chunk_cfg = self._config.get("chunking", {})
        chunker = TextChunker(
            chunk_size=chunk_cfg.get("chunk_size", 512),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
        )
        self.retriever = DocumentRetrieverSkill(
            vector_store=self.vector_store,
            chunker=chunker,
        )

        # Tool registry
        self.tools = ToolRegistry()

        # System prompt
        self._load_system_prompt()

        logger.info("RAG Agent initialized successfully.")

    def _load_system_prompt(self):
        """Load the system prompt from file."""
        prompt_path = Path("agent/prompts/system.md")
        try:
            if prompt_path.exists():
                self._system_prompt = prompt_path.read_text(encoding="utf-8")
            else:
                self._system_prompt = "You are a helpful knowledge base assistant."
        except Exception:
            self._system_prompt = "You are a helpful knowledge base assistant."

    # ---- Document ingestion ----

    async def ingest(self, file_path: str) -> int:
        """Ingest a single document into the knowledge base.

        Returns the number of chunks created.
        """
        return await self.retriever.ingest(file_path)

    async def ingest_directory(self, dir_path: str) -> Dict[str, int]:
        """Batch ingest all documents in a directory.

        Returns {file_path: chunk_count}.
        """
        return await self.retriever.ingest_directory(dir_path)

    # ---- Main pipeline ----

    async def ask(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> AgentState:
        """Run the full RAG pipeline for a single query.

        Args:
            query: The user's question.
            session_id: Optional session ID for multi-turn conversation.
                        A new one is generated if not provided.

        Returns:
            AgentState containing the answer, citations, and pipeline metadata.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        state = self._state_machine.create_state(session_id, query)

        try:
            # Stage 1: Retrieve context
            self._state_machine.transition(state, PipelineStage.RETRIEVING)
            retrieval_cfg = self._config.get("retrieval", {})
            top_k = retrieval_cfg.get("top_k_initial", 10)
            chunks = await self.memory.retrieve(query, k=top_k)

            # Filter by similarity threshold
            threshold = retrieval_cfg.get("similarity_threshold", 0.3)
            chunks = [c for c in chunks if c.score >= threshold]

            # Apply top_k_final limit
            top_k_final = retrieval_cfg.get("top_k_final", 5)
            chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k_final]

            from agent.core.state import RetrievalResult
            state.retrieved = RetrievalResult(
                chunks=chunks,
                strategy_used="vector_only",
            )

            # Build context
            state.context = self.memory.build_context(chunks)

            # Stage 2: Generate answer
            self._state_machine.transition(state, PipelineStage.GENERATING)

            # Build messages
            messages = self._build_messages(query, state.context, session_id)

            state.answer = await self.llm.generate(messages)

            # Stage 3: Extract citations
            self._state_machine.transition(state, PipelineStage.VALIDATING)
            state.citations = self._extract_citations(state.answer, chunks)

            # Store in session memory
            self.memory.add_turn(session_id, query, state.answer, state.citations)

            self._state_machine.transition(state, PipelineStage.DONE)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._state_machine.transition(state, PipelineStage.ERROR)
            state.error = str(e)

        return state

    async def ask_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Run the RAG pipeline with streaming output.

        Yields answer tokens as they are generated.
        Non-streaming parts (retrieval) complete first.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        state = self._state_machine.create_state(session_id, query)

        # Retrieve context (non-streaming)
        retrieval_cfg = self._config.get("retrieval", {})
        top_k = retrieval_cfg.get("top_k_initial", 10)
        chunks = await self.memory.retrieve(query, k=top_k)
        threshold = retrieval_cfg.get("similarity_threshold", 0.3)
        chunks = [c for c in chunks if c.score >= threshold]
        chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:retrieval_cfg.get("top_k_final", 5)]
        state.context = self.memory.build_context(chunks)

        # Generate with streaming
        messages = self._build_messages(query, state.context, session_id)
        full_answer = ""
        async for token in self.llm.generate_stream(messages):
            full_answer += token
            yield token

        state.answer = full_answer
        state.citations = self._extract_citations(full_answer, chunks)
        self.memory.add_turn(session_id, query, full_answer, state.citations)
        self._state_machine.transition(state, PipelineStage.DONE)

    # ---- Internal helpers ----

    def _build_messages(
        self,
        query: str,
        context: str,
        session_id: str,
    ) -> List[dict]:
        """Build the message list for the LLM call."""
        messages = []

        # System prompt with context
        if self._system_prompt:
            system_msg = self._system_prompt
            if context:
                system_msg += f"\n\n## Retrieved Context\n\n{context}"
            messages.append({"role": "system", "content": system_msg})
        elif context:
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer the user's question:\n\n{context}",
            })

        # Conversation history
        history = self.memory.get_history(session_id)
        messages.extend(history)

        # Current query
        messages.append({"role": "user", "content": query})

        return messages

    def _extract_citations(
        self,
        answer: str,
        chunks: list,
    ) -> List[Citation]:
        """Extract citations from the answer based on referenced sources."""
        citations = []
        seen_sources = set()

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_name = Path(source).name if isinstance(source, str) else str(source)

            if source_name in answer and source_name not in seen_sources:
                seen_sources.add(source_name)
                excerpt = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                citations.append(Citation(
                    chunk_id=chunk.id,
                    source=source_name,
                    content_excerpt=excerpt,
                ))

        return citations

    def get_state(self, session_id: str) -> Optional[AgentState]:
        """Retrieve the last pipeline state for a session."""
        return self._state_machine.get_state(session_id)

    @property
    def config(self) -> dict:
        """Read-only access to the agent configuration."""
        return dict(self._config)
