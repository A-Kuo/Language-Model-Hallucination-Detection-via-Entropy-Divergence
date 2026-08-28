"""
Tests for data_generator.py::DataGenerator.from_halueval().

The one invariant that matters here: every sample it emits must be part of a
matched question pair (one "correct" + one "hallucinated" LabeledSample
sharing the same question text). An earlier version of this loader used a
HaluEval mirror with no matched pairs at all — see README.md §5.4 — which
silently turned "hallucination detection" into "tell these two disjoint
question populations apart". These tests run against a small mocked dataset
(no network) so the invariant is protected in CI without depending on
HuggingFace Hub availability.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("datasets")

from data_generator import DataGenerator


def _fake_halueval_rows():
    return [
        {
            "knowledge": f"fact {i}",
            "question": f"question {i}?",
            "right_answer": f"right answer {i}",
            "hallucinated_answer": f"hallucinated answer {i}",
        }
        for i in range(10)
    ]


def test_from_halueval_produces_matched_pairs():
    with patch("datasets.load_dataset", return_value=_fake_halueval_rows()):
        samples = DataGenerator.from_halueval(num_samples=12, seed=0)

    by_question = {}
    for s in samples:
        by_question.setdefault(s.question, []).append(s.label)

    assert len(by_question) > 0
    for question, labels in by_question.items():
        assert set(labels) == {"correct", "hallucinated"}, (
            f"question {question!r} is not a matched pair: {labels}"
        )


def test_from_halueval_answers_match_their_label():
    with patch("datasets.load_dataset", return_value=_fake_halueval_rows()):
        samples = DataGenerator.from_halueval(num_samples=8, seed=1)

    for s in samples:
        if s.label == "correct":
            assert s.model_answer == s.ground_truth
            assert "hallucinated" not in s.model_answer
        else:
            assert s.label == "hallucinated"
            assert s.model_answer != s.ground_truth


def test_from_halueval_respects_num_samples():
    with patch("datasets.load_dataset", return_value=_fake_halueval_rows()):
        samples = DataGenerator.from_halueval(num_samples=6, seed=2)
    assert len(samples) == 6  # 3 matched pairs


def test_from_halueval_clips_to_available_rows():
    with patch("datasets.load_dataset", return_value=_fake_halueval_rows()):
        samples = DataGenerator.from_halueval(num_samples=1000, seed=3)
    assert len(samples) == 20  # only 10 rows available -> 10 pairs
