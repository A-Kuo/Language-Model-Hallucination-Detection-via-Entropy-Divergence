"""
Benchmark Schema
==================

`AgentTraceRecord` — one logged trace through the answer/retrieve/abstain
router for a single benchmark item. Mirrors the field/auto-id conventions
of data_generator.py::LabeledSample (hash-derived sample_id, plain-value
fields for clean JSON serialization) rather than introducing a new pattern.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentTraceRecord:
    """
    One item's full trace through the agent.

    action_sequence : e.g. ["answer"], ["retrieve", "answer"],
        ["abstain"], or ["retrieve", "abstain"] — see agent/router.py.
    correct : whether `final_answer` matches the item's gold answer(s).
        None when the agent abstained (nothing to grade).
    groundedness : whether `final_answer` overlaps the context actually
        supplied to the model. None unless "retrieve" is in
        action_sequence — the no-context `answer` path has no context to
        ground against, so this is "not applicable", not False.
    task_success : whether the agent's overall behavior was the right call
        for this item (see benchmarks/tasks.py::_task_success for the exact
        rule — distinct from `correct`, which only judges answer content).
    scores : p_hallucination at each routing call, keyed "no_context" and
        (if retrieval happened) "with_context".
    """
    query: str
    is_answerable: bool
    has_context_available: bool
    action_sequence: List[str]
    final_answer: Optional[str]
    correct: Optional[bool]
    groundedness: Optional[bool]
    task_success: bool
    scores: Dict[str, float]
    latency_s: float
    sample_id: str = ""

    def __post_init__(self):
        if not self.sample_id:
            raw = f"{self.query}|{self.final_answer}"
            self.sample_id = hashlib.md5(raw.encode()).hexdigest()[:12]
