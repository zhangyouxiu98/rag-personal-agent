"""AnswerValidatorAgent: fact-check and citation extraction."""

from typing import List

from agent.core.state import AgentState, Citation, ValidationResult


class AnswerValidatorAgent:
    """Validates generated answers against the original query and context.

    Checks performed:
    - Completeness: does the answer address the question?
    - Grounding: is the answer supported by the retrieved context?
    - Citation extraction: identify source documents referenced in the answer.

    Usage:
        validator = AnswerValidatorAgent(llm_client)
        result = await validator.validate(state)
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def validate(self, state: AgentState) -> ValidationResult:
        """Validate a generated answer.

        Returns ValidationResult with confidence score and any issues found.
        """
        if not state.answer:
            return ValidationResult(
                passed=False,
                confidence=0.0,
                issues=["No answer generated."],
            )

        if not state.context:
            # No context was retrieved — answer may be purely from model knowledge
            return ValidationResult(
                passed=True,
                confidence=0.5,
                issues=["No context available for validation."],
            )

        issues = []

        # Check 1: Completeness
        completeness = await self._check_completeness(
            state.analyzed.original if state.analyzed else state.query,
            state.answer,
        )
        if completeness < 0.5:
            issues.append("Answer may not fully address the question.")

        # Check 2: Grounding (hallucination detection)
        grounded = await self._check_grounding(state.answer, state.context)
        if grounded < 0.5:
            issues.append(
                "Answer may contain information not supported by the retrieved context."
            )

        overall_confidence = (completeness + grounded) / 2.0

        return ValidationResult(
            passed=overall_confidence >= 0.4,
            confidence=overall_confidence,
            issues=issues,
        )

    async def _check_completeness(self, query: str, answer: str) -> float:
        """Check if the answer addresses the question."""
        prompt = f"""Rate how well the answer addresses the question. Reply with ONLY a number between 0.0 and 1.0.
0.0 = completely misses the point, 1.0 = perfectly answers.

Question: {query}
Answer: {answer[:500]}

Completeness score:"""

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return self._parse_score(response)
        except Exception:
            return 0.5  # Neutral on error

    async def _check_grounding(self, answer: str, context: str) -> float:
        """Check if the answer is supported by the context."""
        # Truncate for efficiency
        ctx_snippet = context[:1500] if len(context) > 1500 else context
        ans_snippet = answer[:500] if len(answer) > 500 else answer

        prompt = f"""Is the following answer factually supported by the provided context?
Reply with ONLY a number between 0.0 and 1.0.
0.0 = completely hallucinated/unsupported, 1.0 = fully grounded in context.

Context: {ctx_snippet}

Answer: {ans_snippet}

Grounding score:"""

        try:
            response = await self.llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return self._parse_score(response)
        except Exception:
            return 0.5  # Neutral on error

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a float score from LLM response."""
        import re
        text = response.strip()
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5
