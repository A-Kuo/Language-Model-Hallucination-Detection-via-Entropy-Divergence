"""
Tests for agent/router.py::RoutingPolicy — the RELIABLE/UNCERTAIN/UNRELIABLE
-> answer/retrieve/abstain mapping, and the two-detector split (one for
no-context routing, one for post-retrieval routing). Uses fake detector
stubs (matching CalibratedEntropyDetector.route_one()'s signature) so only
the mapping/dispatch is under test — calibration itself is already covered
by tests/test_calibrated_entropy_detector.py.
"""

import numpy as np

from agent.router import AgentAction, RoutingPolicy
from calibrated_entropy_detector import RoutingDecision

_ACTION_FOR_LABEL = {"RELIABLE": "accept", "UNCERTAIN": "escalate", "UNRELIABLE": "reject"}


class _FakeDetector:
    """Returns a fixed RoutingDecision regardless of input features, so
    each test can pin the exact underlying label under test."""

    def __init__(self, label: str):
        self._label = label

    def route_one(self, x):
        return RoutingDecision(
            label=self._label,
            action=_ACTION_FOR_LABEL[self._label],
            p_hallucination=0.5,
            threshold_reliable=0.3,
            threshold_unreliable=0.7,
        )


def _policy(no_context_label: str, with_context_label: str) -> RoutingPolicy:
    return RoutingPolicy(_FakeDetector(no_context_label), _FakeDetector(with_context_label))


def test_reliable_routes_to_answer():
    policy = _policy("RELIABLE", "RELIABLE")
    decision = policy.decide(np.zeros(24))
    assert decision.action == AgentAction.ANSWER
    assert decision.used_retrieval is False


def test_uncertain_routes_to_retrieve():
    policy = _policy("UNCERTAIN", "RELIABLE")
    decision = policy.decide(np.zeros(24))
    assert decision.action == AgentAction.RETRIEVE


def test_unreliable_routes_to_abstain():
    policy = _policy("UNRELIABLE", "RELIABLE")
    decision = policy.decide(np.zeros(24))
    assert decision.action == AgentAction.ABSTAIN


def test_post_retrieval_reliable_routes_to_answer():
    policy = _policy("UNCERTAIN", "RELIABLE")
    decision = policy.decide_after_retrieval(np.zeros(24))
    assert decision.action == AgentAction.ANSWER
    assert decision.used_retrieval is True


def test_post_retrieval_uncertain_falls_through_to_abstain():
    """A second UNCERTAIN result (still not enough signal even with
    context) has no further evidence-gathering action to try in this MVP
    (no tool_call/verify), so it falls through to abstain rather than
    looping back to `retrieve` again."""
    policy = _policy("RELIABLE", "UNCERTAIN")
    decision = policy.decide_after_retrieval(np.zeros(24))
    assert decision.action == AgentAction.ABSTAIN


def test_post_retrieval_unreliable_routes_to_abstain():
    policy = _policy("RELIABLE", "UNRELIABLE")
    decision = policy.decide_after_retrieval(np.zeros(24))
    assert decision.action == AgentAction.ABSTAIN


# --- Regression coverage for the bug this two-detector split fixes:
# a shared detector, calibrated on one feature regime, silently misrouting
# the other regime's features. -----------------------------------------

def test_decide_only_consults_no_context_detector():
    """decide() must ignore with_context_detector entirely — regardless of
    what it would say, its label must not leak into the first-pass result."""
    policy = _policy(no_context_label="RELIABLE", with_context_label="UNRELIABLE")
    decision = policy.decide(np.zeros(24))
    assert decision.label == "RELIABLE"
    assert decision.action == AgentAction.ANSWER


def test_decide_after_retrieval_only_consults_with_context_detector():
    """decide_after_retrieval() must ignore no_context_detector entirely —
    regardless of what it would say, its label must not leak into the
    second-pass result."""
    policy = _policy(no_context_label="UNRELIABLE", with_context_label="RELIABLE")
    decision = policy.decide_after_retrieval(np.zeros(24))
    assert decision.label == "RELIABLE"
    assert decision.action == AgentAction.ANSWER
