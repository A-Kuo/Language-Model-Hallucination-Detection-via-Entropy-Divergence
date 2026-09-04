"""
Agent Tools
=============

The MVP's single tool: `retrieve_context`.

Deliberately NOT a search — it returns a benchmark item's own gold context
passage (e.g. benchmarks.loaders.SquadItem.context). This keeps `retrieve`
an honest test of the *routing* decision ("does the agent correctly decide
it needs more context before answering?") without pretending to have solved
retrieval-as-search, which is out of scope for this pass (see README.md's
Agent Routing section, "Explicitly deferred").

A calculator/SQL tool was considered for this pass and deferred — see the
same README section — so this is the only tool in agent/tools.py for now.
"""

from __future__ import annotations

from typing import Any


def retrieve_context(item: Any) -> str:
    """
    Return the context already attached to `item` (any object exposing a
    `.context: str` attribute — see benchmarks.loaders.SquadItem).
    """
    return item.context
