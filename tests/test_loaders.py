"""
Tests for benchmarks/loaders.py::load_squad_v2. Mocks datasets.load_dataset
so this runs offline (no HuggingFace Hub access needed), matching the
pattern in tests/test_data_generator.py.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("datasets")

from benchmarks.loaders import load_squad_v2


def _fake_squad_rows():
    rows = []
    for i in range(6):
        rows.append({
            "id": f"answerable-{i}",
            "question": f"question {i}?",
            "context": f"context passage {i}.",
            "answers": {"text": [f"answer {i}"], "answer_start": [0]},
        })
    for i in range(6):
        rows.append({
            "id": f"unanswerable-{i}",
            "question": f"unanswerable question {i}?",
            "context": f"context passage {i}.",
            "answers": {"text": [], "answer_start": []},
        })
    return rows


def test_load_squad_v2_balances_answerable_and_unanswerable():
    with patch("datasets.load_dataset", return_value=_fake_squad_rows()):
        items = load_squad_v2(num_samples=8, seed=0)
    assert len(items) == 8
    assert sum(i.is_answerable for i in items) == 4
    assert sum(not i.is_answerable for i in items) == 4


def test_load_squad_v2_unanswerable_items_have_no_answers():
    with patch("datasets.load_dataset", return_value=_fake_squad_rows()):
        items = load_squad_v2(num_samples=12, seed=1)
    for item in items:
        if item.is_answerable:
            assert len(item.answers) > 0
        else:
            assert item.answers == []


def test_load_squad_v2_clips_to_available_rows():
    with patch("datasets.load_dataset", return_value=_fake_squad_rows()):
        items = load_squad_v2(num_samples=1000, seed=2)
    assert len(items) == 12  # only 6 + 6 rows available
