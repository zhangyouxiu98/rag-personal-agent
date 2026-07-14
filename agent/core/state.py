"""Data models and state machine for the RAG agent pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class PipelineStage(Enum):
    """Stages of the RAG pipeline."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    VALIDATING = "validating"
    DONE = "done"
    ERROR = "error"


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class Citation:
    """A citation linking a claim to its source document."""
    chunk_id: str
    source: str
    content_excerpt: str


@dataclass
class AnalyzedQuery:
    """Result of query analysis."""
    original: str
    rewritten: str
    sub_queries: List[str] = field(default_factory=list)
    intent: str = "factual"
    keywords: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result of knowledge retrieval."""
    chunks: List[DocumentChunk] = field(default_factory=list)
    strategy_used: str = "vector_only"


@dataclass
class ValidationResult:
    """Result of answer validation."""
    passed: bool = True
    confidence: float = 1.0
    issues: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    """Complete state for one RAG query pipeline execution."""
    session_id: str = ""
    query: str = ""
    analyzed: Optional[AnalyzedQuery] = None
    retrieved: Optional[RetrievalResult] = None
    context: str = ""
    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    stage: PipelineStage = PipelineStage.IDLE
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class StateMachine:
    """Manages state transitions through the RAG pipeline.

    Provides a central registry for tracking active sessions and
    enforcing valid state transitions between pipeline stages.
    """

    VALID_TRANSITIONS = {
        PipelineStage.IDLE:       {PipelineStage.ANALYZING, PipelineStage.RETRIEVING},
        PipelineStage.ANALYZING:  {PipelineStage.RETRIEVING, PipelineStage.ERROR},
        PipelineStage.RETRIEVING: {PipelineStage.GENERATING, PipelineStage.ERROR},
        PipelineStage.GENERATING: {PipelineStage.VALIDATING, PipelineStage.ERROR},
        PipelineStage.VALIDATING: {PipelineStage.DONE, PipelineStage.ERROR},
        PipelineStage.DONE:       set(),
        PipelineStage.ERROR:      set(),
        # Terminal transition — any stage can transition to DONE
        "_default":               set(),
    }

    # Allow DONE as a valid target from any non-terminal stage
    _DONE_FROM_ANY = {PipelineStage.IDLE, PipelineStage.ANALYZING,
                      PipelineStage.RETRIEVING, PipelineStage.GENERATING,
                      PipelineStage.VALIDATING}

    def __init__(self):
        self._states: Dict[str, AgentState] = {}

    def create_state(self, session_id: str, query: str) -> AgentState:
        """Create a new agent state for a query."""
        state = AgentState(
            session_id=session_id,
            query=query,
            stage=PipelineStage.IDLE,
        )
        self._states[session_id] = state
        return state

    def get_state(self, session_id: str) -> Optional[AgentState]:
        """Retrieve an existing agent state by session ID."""
        return self._states.get(session_id)

    def update_state(self, state: AgentState):
        """Store an updated agent state."""
        self._states[state.session_id] = state

    def delete_state(self, session_id: str):
        """Remove a session state."""
        self._states.pop(session_id, None)

    def transition(self, state: AgentState, next_stage: PipelineStage):
        """Transition to the next pipeline stage, validating the move."""
        valid = self.VALID_TRANSITIONS.get(state.stage, set())
        # Allow DONE from any non-terminal stage
        if next_stage == PipelineStage.DONE and state.stage in self._DONE_FROM_ANY:
            state.stage = next_stage
            return state
        if next_stage not in valid:
            raise ValueError(
                f"Invalid transition: {state.stage.value} -> {next_stage.value}. "
                f"Valid next stages: {[s.value for s in valid]}"
            )
        state.stage = next_stage
        return state

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        return len(self._states)
