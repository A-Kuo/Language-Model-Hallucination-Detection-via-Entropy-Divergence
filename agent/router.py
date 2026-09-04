"""
Agent Router — answer / retrieve / abstain
=============================================

MVP routing policy for the "Calibrated Agent Routing" pivot (see
README.md's Agent Routing section and AGENT.md's design-doc origin). This
is deliberately a THIN wrapper, not a new calibration module: the actual
uncertainty math already lives in calibrated_entropy_detector.py's
CalibratedEntropyDetector.route()/route_one(), which fits two calibrated
thresholds (reliable_quantile/unreliable_quantile) from a labeled
calibration set and maps a blended probability into a 3-way RELIABLE /
UNCERTAIN / UNRELIABLE decision. RoutingPolicy here does nothing more than
relabel that 3-way decision onto the 3 MVP agent actions:

    RELIABLE   -> answer     (trust the current answer as-is)
    UNCERTAIN  -> retrieve   (fetch context, re-answer, re-route)
    UNRELIABLE -> abstain    (decline to answer)

Only these three actions are implemented. `tool_call` and `verify` — two of
the five actions in the original design doc — are intentionally not built
yet; adding them is future work once this MVP has produced real numbers to
build on (see README.md's Agent Routing section for the explicit scope
list).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AgentAction(str, Enum):
    ANSWER = "answer"
    RETRIEVE = "retrieve"
    ABSTAIN = "abstain"


@dataclass
class AgentDecision:
    """One routing call's result: the chosen action, plus the underlying
    calibrated label/probability it was derived from (for logging/debugging
    — see benchmarks/schemas.py::AgentTraceRecord, which stores these as
    plain values, not this dataclass, to keep the trace log JSON-friendly)."""
    action: AgentAction
    label: str  # "RELIABLE" | "UNCERTAIN" | "UNRELIABLE", from the wrapped detector
    p_hallucination: float
    used_retrieval: bool


class RoutingPolicy:
    """
    Wraps TWO fitted CalibratedEntropyDetector instances and maps their
    route_one() output onto the answer/retrieve/abstain actions — one
    detector per routing call, matching the two distinct feature regimes a
    single agent trace can produce (see benchmarks/tasks.py::run_agent_task):

        decide(features)                  first pass, before any retrieval
                                           -> consults no_context_detector
        decide_after_retrieval(features)  second pass, after `retrieve` has
                                           attached context and the model
                                           has re-answered
                                           -> consults with_context_detector

    A single shared detector was tried first and failed concretely: a
    CalibratedEntropyDetector calibrated on with-context features, applied
    to no-context features (a different regime — shorter prompt, different
    attention shape), didn't land in the UNCERTAIN middle band the way it's
    supposed to. In one real run this produced 0% `retrieve` actions (every
    item resolved straight to RELIABLE or UNRELIABLE) and the few items that
    did route to `answer` were wrong 100% of the time — confidently wrong,
    which is worse than the honest all-abstain result a no-context-only
    calibration produced. Each detector should only ever score the kind of
    feature vector it was calibrated on.

    decide_after_retrieval() only recognizes RELIABLE (-> answer) as a
    non-abstain outcome. A second UNCERTAIN result means retrieval did not
    resolve the uncertainty, and this MVP has no further evidence-gathering
    action to try (no tool_call/verify) — so, like UNRELIABLE, it falls
    through to abstain rather than looping.
    """

    def __init__(self, no_context_detector, with_context_detector) -> None:
        self.no_context_detector = no_context_detector
        self.with_context_detector = with_context_detector

    def _decide(self, decision, used_retrieval: bool) -> AgentDecision:
        if decision.label == "RELIABLE":
            action = AgentAction.ANSWER
        elif decision.label == "UNCERTAIN" and not used_retrieval:
            action = AgentAction.RETRIEVE
        else:  # UNRELIABLE, or UNCERTAIN-after-retrieval
            action = AgentAction.ABSTAIN
        return AgentDecision(
            action=action,
            label=decision.label,
            p_hallucination=decision.p_hallucination,
            used_retrieval=used_retrieval,
        )

    def decide(self, features: np.ndarray) -> AgentDecision:
        """First-pass routing call, before any retrieval has happened.
        Consults no_context_detector."""
        decision = self.no_context_detector.route_one(features)
        return self._decide(decision, used_retrieval=False)

    def decide_after_retrieval(self, features: np.ndarray) -> AgentDecision:
        """Second-pass routing call, after `retrieve` has attached context
        and the model has re-answered with it. Consults
        with_context_detector."""
        decision = self.with_context_detector.route_one(features)
        return self._decide(decision, used_retrieval=True)
