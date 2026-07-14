"""QueryAnalyzerAgent: query analysis, rewriting, and decomposition."""

import json
from typing import List, Optional

from agent.core.state import AnalyzedQuery


class QueryAnalyzerAgent:
    """Analyzes and rewrites user queries for optimal RAG retrieval.

    Responsibilities:
    - Query rewriting with conversation context
    - Intent classification (factual, summarization, comparison, procedural)
    - Query decomposition into sub-queries for complex questions
    - Keyword extraction for focused retrieval
    """

    def __init__(self, llm_client, memory_manager):
        self.llm = llm_client
        self.memory = memory_manager

    async def analyze(
        self,
        query: str,
        session_id: str,
    ) -> AnalyzedQuery:
        """Analyze a query end-to-end.

        Steps:
        1. Load conversation history for context
        2. Rewrite query as standalone question
        3. Classify intent and extract keywords
        4. Decompose into sub-queries if complex

        Returns AnalyzedQuery with full analysis.
        """
        # Step 1: Get conversation history
        history = self.memory.get_history(session_id)

        # Step 2: Rewrite query with conversation context
        rewritten = await self._rewrite_with_context(query, history)

        # Step 3: Analyze intent + extract keywords
        analysis = await self._analyze_intent(rewritten)

        sub_queries = analysis.get("sub_queries", [rewritten])

        return AnalyzedQuery(
            original=query,
            rewritten=rewritten,
            sub_queries=sub_queries,
            intent=analysis.get("intent", "factual"),
            keywords=analysis.get("keywords", []),
        )

    async def _rewrite_with_context(
        self,
        query: str,
        history: List[dict],
    ) -> str:
        """Rewrite the query as a standalone question using conversation history.

        If no relevant history, returns the original query unchanged.
        """
        if not history:
            return query

        # Only use the last few turns for context
        recent = history[-4:]

        prompt = f"""Based on the conversation history, rewrite the latest question
as a standalone, self-contained query for document retrieval.

Conversation history:
{self._format_history(recent)}

Latest question: {query}

Standalone query:"""

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten = response.strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    async def _analyze_intent(self, query: str) -> dict:
        """Classify query intent and extract keywords/sub-queries."""
        prompt = f"""Analyze this question for RAG retrieval. Return ONLY a JSON object (no other text):

{{
  "intent": "factual|summarization|comparison|procedural",
  "sub_queries": ["sub-question 1", "sub-question 2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Question: {query}
JSON:"""

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return self._parse_json_safely(response)
        except Exception:
            return {"intent": "factual", "sub_queries": [query], "keywords": []}

    @staticmethod
    def _format_history(history: List[dict]) -> str:
        """Format conversation history for prompt inclusion."""
        lines = []
        for entry in history:
            role = "User" if entry["role"] == "user" else "Assistant"
            content = entry.get("content", "")
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_safely(response: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = response.strip()

        # Remove markdown code fence if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (``` or ```json) and last line (```)
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        return json.loads(text)
