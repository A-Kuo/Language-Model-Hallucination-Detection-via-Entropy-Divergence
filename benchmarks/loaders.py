"""
Benchmark Loaders
===================

`load_squad_v2` — loads a balanced sample of SQuAD 2.0 questions
(answerable + unanswerable) for the answer/retrieve/abstain benchmark.

SQuAD 2.0 was chosen (see README.md's Agent Routing section) because it
gives, natively and for free, exactly the two signals this MVP needs:
  - a genuine "should the agent abstain?" ground truth (the unanswerable
    subset), unlike HaluEval's matched correct/hallucinated *answer* pairs,
    which have no notion of "no correct answer exists".
  - a gold context passage per question, which doubles as the content
    agent.tools.retrieve_context returns — no retrieval index needs to be
    built for this pass.

Requires: pip install datasets (same optional dependency HaluEval loading
already requires — see data_generator.py::DataGenerator.from_halueval).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class SquadItem:
    """One SQuAD 2.0 question with its gold context and answerability."""
    question: str
    context: str
    is_answerable: bool
    answers: List[str]  # empty for unanswerable items
    sample_id: str


def load_squad_v2(num_samples: int = 200, seed: int = 42) -> List[SquadItem]:
    """
    Load a balanced sample of SQuAD 2.0 validation-split questions: as close
    to a 50/50 answerable/unanswerable split as the requested `num_samples`
    and the dataset's own composition allow.

    Parameters
    ----------
    num_samples : int
        Total items to return (split as evenly as available rows allow
        between answerable and unanswerable).
    seed : int
        Random seed for shuffling/sampling.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("HuggingFace `datasets` required: pip install datasets")

    rng = random.Random(seed)

    print("  Downloading SQuAD 2.0 validation split from HuggingFace Hub...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rows = list(ds)
    rng.shuffle(rows)
    print(f"  Loaded {len(rows)} rows from SQuAD 2.0.")

    answerable = [r for r in rows if len(r["answers"]["text"]) > 0]
    unanswerable = [r for r in rows if len(r["answers"]["text"]) == 0]

    half = num_samples // 2
    n_ans = min(half, len(answerable))
    n_unans = min(num_samples - n_ans, len(unanswerable))
    # If one pool ran short, backfill from the other rather than silently
    # returning fewer than num_samples items when the total is available.
    n_ans = min(n_ans + max(0, (num_samples - n_ans - n_unans)), len(answerable))

    picked = answerable[:n_ans] + unanswerable[:n_unans]
    rng.shuffle(picked)

    items = [
        SquadItem(
            question=row["question"],
            context=row["context"],
            is_answerable=len(row["answers"]["text"]) > 0,
            answers=list(row["answers"]["text"]),
            sample_id=str(row["id"]),
        )
        for row in picked
    ]
    n_answerable_final = sum(item.is_answerable for item in items)
    print(
        f"  SQuAD 2.0 sample: {len(items)} items "
        f"({n_answerable_final} answerable, {len(items) - n_answerable_final} unanswerable)"
    )
    return items
