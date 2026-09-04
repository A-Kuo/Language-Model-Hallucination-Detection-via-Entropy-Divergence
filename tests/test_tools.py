"""Tests for agent/tools.py::retrieve_context — the MVP's one tool."""

from dataclasses import dataclass

from agent.tools import retrieve_context


@dataclass
class _FakeItem:
    context: str


def test_retrieve_context_returns_item_context():
    item = _FakeItem(context="Paris is the capital of France.")
    assert retrieve_context(item) == "Paris is the capital of France."
