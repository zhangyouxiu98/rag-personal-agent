"""Tests for core agent components."""

import pytest
import asyncio

from agent.core.state import (
    AgentState,
    PipelineStage,
    StateMachine,
    AnalyzedQuery,
    RetrievalResult,
    ValidationResult,
    DocumentChunk,
    Citation,
)


class TestStateMachine:
    """Tests for the StateMachine class."""

    def test_create_state(self):
        sm = StateMachine()
        state = sm.create_state("sess_1", "What is AI?")
        assert state.session_id == "sess_1"
        assert state.query == "What is AI?"
        assert state.stage == PipelineStage.IDLE
        assert sm.active_sessions == 1

    def test_get_state(self):
        sm = StateMachine()
        sm.create_state("sess_1", "test query")
        state = sm.get_state("sess_1")
        assert state is not None
        assert state.query == "test query"
        assert sm.get_state("nonexistent") is None

    def test_transition_valid(self):
        sm = StateMachine()
        state = sm.create_state("sess_1", "query")
        sm.transition(state, PipelineStage.RETRIEVING)
        assert state.stage == PipelineStage.RETRIEVING

    def test_transition_invalid(self):
        sm = StateMachine()
        state = sm.create_state("sess_1", "query")
        with pytest.raises(ValueError):
            sm.transition(state, PipelineStage.GENERATING)  # Can't skip Analyzing

    def test_delete_state(self):
        sm = StateMachine()
        sm.create_state("sess_1", "query")
        sm.delete_state("sess_1")
        assert sm.get_state("sess_1") is None
        assert sm.active_sessions == 0

    def test_full_pipeline_transitions(self):
        sm = StateMachine()
        state = sm.create_state("sess_1", "query")

        sm.transition(state, PipelineStage.RETRIEVING)
        assert state.stage == PipelineStage.RETRIEVING

        sm.transition(state, PipelineStage.GENERATING)
        assert state.stage == PipelineStage.GENERATING

        sm.transition(state, PipelineStage.VALIDATING)
        assert state.stage == PipelineStage.VALIDATING

        sm.transition(state, PipelineStage.DONE)
        assert state.stage == PipelineStage.DONE


class TestStateModels:
    """Tests for state dataclasses."""

    def test_document_chunk_defaults(self):
        chunk = DocumentChunk(id="1", content="test content")
        assert chunk.id == "1"
        assert chunk.score == 0.0
        assert chunk.metadata == {}

    def test_analyzed_query(self):
        aq = AnalyzedQuery(
            original="原始问题",
            rewritten="改写后问题",
            intent="comparison",
            keywords=["key1", "key2"],
        )
        assert aq.intent == "comparison"
        assert len(aq.keywords) == 2

    def test_retrieval_result(self):
        chunks = [
            DocumentChunk(id="1", content="chunk1", score=0.9),
        ]
        result = RetrievalResult(
            chunks=chunks,
            strategy_used="vector_only",
        )
        assert len(result.chunks) == 1
        assert result.strategy_used == "vector_only"

    def test_validation_result(self):
        vr = ValidationResult(
            passed=True,
            confidence=0.85,
            issues=["Minor concern"],
        )
        assert vr.passed is True
        assert vr.confidence == 0.85
        assert len(vr.issues) == 1

    def test_agent_state(self):
        state = AgentState(
            session_id="sess_1",
            query="What is Python?",
            answer="Python is a programming language.",
        )
        assert state.stage == PipelineStage.IDLE
        assert state.answer == "Python is a programming language."
