"""RelevanceRanker: re-rank retrieved chunks using LLM scoring."""

from typing import List

from agent.core.state import DocumentChunk


class RelevanceRanker:
    """Re-rank retrieved document chunks by relevance to the query.

    Uses LLM-based scoring for accuracy, falling back to cosine
    similarity when the LLM is unavailable.

    Usage:
        ranker = RelevanceRanker(llm_client)
        reranked = await ranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 5,
    ) -> List[DocumentChunk]:
        """Re-rank chunks by relevance to the query.

        Args:
            query: The search query.
            chunks: Retrieved chunks to re-rank.
            top_k: Number of top chunks to return.

        Returns:
            Top-k chunks sorted by relevance score descending.
        """
        if len(chunks) <= top_k:
            return sorted(chunks, key=lambda c: c.score, reverse=True)

        # Score each chunk
        scored = []
        for chunk in chunks:
            try:
                score = await self._score_relevance(query, chunk.content)
            except Exception:
                # Fall back to existing score on error
                score = chunk.score
            chunk.score = score
            scored.append(chunk)

        # Sort by score descending and return top-k
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    async def _score_relevance(self, query: str, chunk_text: str) -> float:
        """Use LLM to score the relevance of a chunk to a query (0.0-1.0)."""
        # Truncate chunk for scoring
        snippet = chunk_text[:800] if len(chunk_text) > 800 else chunk_text

        prompt = f"""Rate the relevance of this document chunk to the query on a scale of 0.0 to 1.0.
0.0 = completely irrelevant, 1.0 = perfectly answers the query.
Reply with ONLY a number (e.g. 0.85). No explanation.

Query: {query}

Document chunk: {snippet}

Relevance score:"""

        response = await self.llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        try:
            # Extract the first float-like number
            import re
            match = re.search(r"(\d+\.?\d*)", response.strip())
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            pass

        return 0.0
