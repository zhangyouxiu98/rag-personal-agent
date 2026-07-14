"""In-memory session memory for multi-turn conversation history."""

import time
from typing import Dict, List, Optional
from collections import OrderedDict

from agent.memory.base import BaseSessionMemory


class SessionMemory(BaseSessionMemory):
    """In-memory conversation history with sliding window and TTL support.

    Each session maintains an ordered list of conversation turns.
    When max_history is exceeded, the oldest turns are evicted.
    When session_ttl_minutes elapses, the session is automatically
    pruned on the next access.
    """

    def __init__(
        self,
        max_history: int = 20,
        ttl_minutes: int = 1440,  # default 24 hours
    ):
        self.max_history = max_history
        self.ttl_seconds = ttl_minutes * 60
        self._sessions: Dict[str, dict] = OrderedDict()

    def _ensure_session(self, session_id: str):
        """Create a session entry if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history": [],
                "created_at": time.time(),
                "last_access": time.time(),
            }
        self._sessions[session_id]["last_access"] = time.time()

    def _is_expired(self, session_id: str) -> bool:
        """Check if a session has exceeded its TTL."""
        session = self._sessions.get(session_id)
        if session is None:
            return True
        elapsed = time.time() - session["last_access"]
        return elapsed > self.ttl_seconds

    def get_history(self, session_id: str) -> List[dict]:
        """Retrieve conversation history for a session.

        Returns an empty list if the session is expired or does not exist.
        """
        if self._is_expired(session_id):
            self._sessions.pop(session_id, None)
            return []
        self._ensure_session(session_id)
        return list(self._sessions[session_id]["history"])

    def add_entry(self, session_id: str, role: str, content: str):
        """Add a conversation turn and enforce the sliding window limit."""
        self._ensure_session(session_id)
        history = self._sessions[session_id]["history"]
        history.append({"role": role, "content": content})

        # Enforce max_history: keep the most recent turns
        if len(history) > self.max_history:
            excess = len(history) - self.max_history
            self._sessions[session_id]["history"] = history[excess:]

    def get_recent(self, session_id: str, n: int = 5) -> List[dict]:
        """Get the most recent n conversation turns."""
        history = self.get_history(session_id)
        return history[-n:] if n > 0 else []

    def clear(self, session_id: str):
        """Remove a session's history."""
        self._sessions.pop(session_id, None)

    def clear_all(self):
        """Remove all sessions."""
        self._sessions.clear()

    def prune_expired(self) -> int:
        """Remove all expired sessions. Returns number pruned."""
        expired = [
            sid for sid in list(self._sessions.keys())
            if self._is_expired(sid)
        ]
        for sid in expired:
            self._sessions.pop(sid)
        return len(expired)

    @property
    def active_sessions(self) -> int:
        """Number of active sessions."""
        self.prune_expired()
        return len(self._sessions)

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        """Get metadata about a session without the full history."""
        if self._is_expired(session_id):
            self._sessions.pop(session_id, None)
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "message_count": len(session["history"]),
            "created_at": session["created_at"],
            "last_access": session["last_access"],
        }
