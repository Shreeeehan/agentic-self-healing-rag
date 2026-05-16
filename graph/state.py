"""
graph/state.py

This is the SHARED MEMORY object that every agent reads and writes.
Think of it like a whiteboard all agents can see.

When LangGraph runs the pipeline, it passes this state object
from node to node — each agent adds its results to it.
"""

from typing import List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from enum import Enum
import operator

class HealingStatus(Enum):
    NOT_STARTED  = "not_started"
    RETRIEVAL_OK = "retrieval_ok"
    RETRIEVAL_BAD = "retrieval_bad"
    HEALING      = "healing"
    HEALED       = "healed"
    FAILED       = "failed"

class VerificationStatus(Enum):
    NOT_VERIFIED  = "not_verified"
    VERIFIED      = "verified"
    HALLUCINATED  = "hallucinated"

@dataclass
class RAGState:
    # ── Input ──────────────────────────────────────────────────────
    query: str = ""                          # Original user query
    rephrased_query: str = ""               # Repair Agent rewrites this

    # ── Retrieval ──────────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    # Each chunk = {"text": ..., "source": ..., "page": ..., "score": ...}

    # ── Critic ─────────────────────────────────────────────────────
    critic_score: float = 0.0               # 0.0 to 1.0 relevance score
    critic_feedback: str = ""               # Why it passed or failed
    critic_passed: bool = False             # True = good retrieval

    # ── Healing ────────────────────────────────────────────────────
    healing_status: HealingStatus = HealingStatus.NOT_STARTED
    healing_loop_count: int = 0             # How many times we've healed
    max_healing_loops: int = 3              # From .env

    # ── Answer ─────────────────────────────────────────────────────
    final_answer: str = ""                  # What the user sees
    answer_sources: List[Dict[str, Any]] = field(default_factory=list)

    # ── Verification ───────────────────────────────────────────────
    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED
    verification_feedback: str = ""         # Why it passed or failed

    # ── Metadata ───────────────────────────────────────────────────
    error: Optional[str] = None             # Any error that occurred
    run_log: List[str] = field(default_factory=list)
    # run_log tracks every step — great for debugging + UI display

    def log(self, message: str):
        """Add a timestamped log entry. All agents call this."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.run_log.append(entry)
        print(entry)  # also prints to terminal during dev

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dict for API responses."""
        return {
            "query": self.query,
            "rephrased_query": self.rephrased_query,
            "critic_score": self.critic_score,
            "critic_passed": self.critic_passed,
            "critic_feedback": self.critic_feedback,
            "healing_status": self.healing_status.value,
            "healing_loop_count": self.healing_loop_count,
            "final_answer": self.final_answer,
            "verification_status": self.verification_status.value,
            "verification_feedback": self.verification_feedback,
            "sources": self.answer_sources,
            "run_log": self.run_log,
            "error": self.error
        }