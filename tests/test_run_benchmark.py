"""
Tests for the grading/summarization logic in benchmarks/tasks.py and
experiments/run_benchmark.py. Uses only synthetic SquadItem/AgentTraceRecord
objects — no model download or network access required, matching
tests/test_pipeline_integration.py's synthetic-data pattern.
"""

from benchmarks.loaders import SquadItem
from benchmarks.schemas import AgentTraceRecord
from benchmarks.tasks import _grade_correctness, _grade_groundedness, _task_success
from experiments.run_benchmark import _summarize


def _item(is_answerable, answers=None, context="Paris is the capital of France."):
    return SquadItem(
        question="What is the capital of France?",
        context=context,
        is_answerable=is_answerable,
        answers=answers or [],
        sample_id="test-item",
    )


# --- _grade_correctness -----------------------------------------------

def test_grade_correctness_true_on_gold_contained_in_verbose_answer():
    item = _item(True, answers=["Paris"])
    assert _grade_correctness("The capital of France is Paris.", item) is True


def test_grade_correctness_false_on_wrong_answer():
    item = _item(True, answers=["Paris"])
    assert _grade_correctness("The capital of France is Berlin.", item) is False


def test_grade_correctness_false_for_unanswerable_item_regardless_of_text():
    item = _item(False, answers=[])
    assert _grade_correctness("Paris", item) is False


# --- _grade_groundedness -------------------------------------------------

def test_grade_groundedness_none_without_retrieval():
    item = _item(True, answers=["Paris"], context="Paris is the capital of France.")
    assert _grade_groundedness("Paris", item, action_sequence=["answer"]) is None


def test_grade_groundedness_true_when_answer_overlaps_context():
    item = _item(True, answers=["Paris"], context="Paris is the capital of France.")
    assert _grade_groundedness("Paris is the capital.", item, action_sequence=["retrieve", "answer"]) is True


def test_grade_groundedness_false_when_answer_unrelated_to_context():
    item = _item(True, answers=["Paris"], context="Paris is the capital of France.")
    assert _grade_groundedness("bananas are yellow", item, action_sequence=["retrieve", "answer"]) is False


# --- _task_success ---------------------------------------------------------

def test_task_success_true_for_correct_abstention_on_unanswerable():
    item = _item(False)
    assert _task_success(item, final_answer=None, correct=None) is True


def test_task_success_false_for_answering_unanswerable_item():
    item = _item(False)
    assert _task_success(item, final_answer="anything", correct=False) is False


def test_task_success_true_for_correct_answer():
    item = _item(True, answers=["Paris"])
    assert _task_success(item, final_answer="Paris", correct=True) is True


def test_task_success_false_for_incorrect_answer():
    item = _item(True, answers=["Paris"])
    assert _task_success(item, final_answer="Berlin", correct=False) is False


def test_task_success_false_for_abstaining_on_answerable_item():
    item = _item(True, answers=["Paris"])
    assert _task_success(item, final_answer=None, correct=None) is False


# --- _summarize --------------------------------------------------------

def _record(action_sequence, is_answerable, correct, task_success, final_answer="ans"):
    return AgentTraceRecord(
        query="q",
        is_answerable=is_answerable,
        has_context_available=True,
        action_sequence=action_sequence,
        final_answer=final_answer if action_sequence[-1] != "abstain" else None,
        correct=correct,
        groundedness=None,
        task_success=task_success,
        scores={"no_context": 0.1},
        latency_s=0.2,
    )


def test_summarize_counts_actions_and_rates():
    records = [
        _record(["answer"], is_answerable=True, correct=True, task_success=True),
        _record(["retrieve", "answer"], is_answerable=True, correct=True, task_success=True),
        _record(["retrieve", "abstain"], is_answerable=False, correct=None, task_success=True),
        _record(["abstain"], is_answerable=True, correct=None, task_success=False),
    ]
    summary = _summarize(records)

    assert summary["num_samples"] == 4
    assert summary["action_counts"]["answer"] == 2
    assert summary["action_counts"]["abstain"] == 2
    assert summary["task_success_rate"] == 0.75
    assert summary["retrieval_rate"] == 0.5
    assert summary["abstain_rate"] == 0.5
    # 1 of 2 abstentions was on a genuinely unanswerable item
    assert summary["abstain_precision"] == 0.5
    # the only unanswerable item was correctly abstained on
    assert summary["abstain_recall"] == 1.0


def test_summarize_handles_empty_records():
    assert _summarize([]) == {"num_samples": 0}
