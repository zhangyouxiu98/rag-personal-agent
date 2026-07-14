"""KnowledgeRetrieverAgent: multi-strategy document retrieval."""

from typing import List, Optional

from agent.core.state import DocumentChunk, AnalyzedQuery, RetrievalResult
from agent.memory.base import BaseVectorStore
from agent.core.tools import BaseTool


class KnowledgeRetrieverAgent:
    """Multi-strategy knowledge retrieval orchestrator.

    Retrieval strategies (in priority order):
    1. Vector search on rewritten query (primary)
    2. Sub-query expansion search
    3. Re-ranking of merged results
    4. Web search fallback (if enabled)

    Usage:
        retriever = KnowledgeRetrieverAgent(vector_store, ranker)
        result = await retriever.retrieve(analyzed_query)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        ranker,  # RelevanceRanker
        web_search_tool: Optional[BaseTool] = None,
        top_k_initial: int = 10,
        top_k_final: int = 5,
        similarity_threshold: float = 0.3,
    ):
        self.vector_store = vector_store
        self.ranker = ranker
        self.web_search = web_search_tool
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.similarity_threshold = similarity_threshold

    async def retrieve(self, analyzed: AnalyzedQuery) -> RetrievalResult:
        """Execute multi-strategy retrieval.

        Args:
            analyzed: The analyzed query with rewritten text and sub-queries.

        Returns:
            RetrievalResult with ranked chunks and strategy used.
        """
        all_chunks: List[DocumentChunk] = []
        strategy = "vector_only"

        # Strategy 1: Primary query search
        primary_chunks = await self.vector_store.similarity_search(
            analyzed.rewritten,
            k=self.top_k_initial,
        )
        all_chunks.extend(primary_chunks)

        # Strategy 2: Sub-query expansion
        for sub_q in analyzed.sub_queries:
            if sub_q != analyzed.rewritten:
                sub_chunks = await self.vector_store.similarity_search(
                    sub_q,
                    k=max(3, self.top_k_initial // 2),
                )
                all_chunks.extend(sub_chunks)
                strategy = "query_expansion"

        # Deduplicate by chunk ID
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            if c.id not in seen:
                seen.add(c.id)
                unique_chunks.append(c)

        # Re-rank merged results
        if self.ranker and unique_chunks:
            unique_chunks = await self.ranker.rerank(
                analyzed.rewritten,
                unique_chunks,
                top_k=self.top_k_final,
            )
            strategy = "reranked"

        # Filter by threshold
        unique_chunks = [
            c for c in unique_chunks
            if c.score >= self.similarity_threshold
        ]

        # Strategy 3: Web search fallback
        if len(unique_chunks) == 0 and self.web_search:
            try:
                web_result = await self.web_search.execute(query=analyzed.original)
                if web_result:
                    # Create a pseudo-chunk from web results
                    web_chunk = DocumentChunk(
                        id="web_search_result",
                        content=web_result,
                        metadata={"source": "web_search", "file_type": "web"},
                        score=0.5,
                    )
                    unique_chunks.append(web_chunk)
                    strategy = "web_search"
            except Exception:
                pass

        return RetrievalResult(
            chunks=unique_chunks,
            strategy_used=strategy,
        )
