"""
Benchmark Task Orchestration
==============================

Runs a single SQuAD 2.0 item through the answer/retrieve/abstain agent:

    1. Generate a no-context answer (model.generate() — the repo's other
       code, e.g. pipeline.py, only ever teacher-forces already-written
       text; this is the one place a real generation loop is needed).
    2. Extract the same 24D attention+entropy feature vector the rest of
       the repo uses (via pipeline.py's extract_attention_from_model /
       extract_logits_from_model — teacher-forcing over the just-generated
       text, same pattern as demo/app.py::score_text).
    3. Route on those features (agent.router.RoutingPolicy).
    4. If routed to `retrieve`: fetch the item's gold context
       (agent.tools.retrieve_context), re-answer with it prepended, extract
       features again, and route a second time.
    5. Grade the resulting trace against SQuAD's own answerability/answer
       labels and log it as an AgentTraceRecord.

Grading uses a lightweight token-F1 overlap (the same statistic behind the
official SQuAD eval script), not exact match — a small base model's
free-form phrasing rarely matches gold text word-for-word.
"""

from __future__ import annotations

import re
import string
import time
from typing import List, Optional, Tuple

import numpy as np

from agent.router import AgentAction, RoutingPolicy
from agent.tools import retrieve_context
from benchmarks.loaders import SquadItem
from benchmarks.schemas import AgentTraceRecord
from entropy_baselines import EntropyFeatureExtractor
from feature_engineer import AttentionFeatureEngineer
from pipeline import build_prompt_and_text, extract_attention_from_model, extract_logits_from_model

F1_MATCH_THRESHOLD = 0.5
GROUNDEDNESS_OVERLAP_THRESHOLD = 0.1


# =========================================================================
# Text-overlap grading (SQuAD-eval-script-style token F1)
# =========================================================================

def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    num_same = 0
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            num_same += 1
            gold_counts[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _grade_correctness(answer: str, item: SquadItem) -> bool:
    """
    Whether `answer` matches one of `item`'s gold answers. An unanswerable
    item has no gold answer, so ANY answer to it is wrong by definition.

    Two checks, either sufficient: (1) the normalized gold answer appears
    verbatim inside the normalized prediction — needed because this repo's
    model free-generates full sentences ("The capital of France is Paris.")
    rather than extracting a short span the way SQuAD-tuned models do, and
    a short gold answer ("Paris") buried in a longer correct sentence tanks
    token-F1 on precision alone; (2) token-F1 over the whole strings meets
    F1_MATCH_THRESHOLD, for cases where the phrasing differs enough that
    containment doesn't hold but the content still overlaps substantially.
    """
    if not item.is_answerable or not item.answers:
        return False
    norm_pred = _normalize_answer(answer)
    for gold in item.answers:
        norm_gold = _normalize_answer(gold)
        if norm_gold and norm_gold in norm_pred:
            return True
    best_f1 = max(_token_f1(answer, gold) for gold in item.answers)
    return best_f1 >= F1_MATCH_THRESHOLD


def _grade_groundedness(answer: str, item: SquadItem, action_sequence: List[str]) -> Optional[bool]:
    """
    Whether `answer` overlaps the context passage actually supplied to the
    model. Only meaningful when retrieval happened (the no-context `answer`
    path has no context to ground against) — returns None otherwise, not
    False, since groundedness simply isn't measured on that path.
    """
    if "retrieve" not in action_sequence:
        return None
    return _token_f1(answer, item.context) >= GROUNDEDNESS_OVERLAP_THRESHOLD


def _task_success(item: SquadItem, final_answer: Optional[str], correct: Optional[bool]) -> bool:
    """
    Whether the agent's overall behavior was the right call:
      - unanswerable + abstained  -> True  (correctly declined)
      - unanswerable + answered   -> False (asserted something with no ground truth)
      - answerable + correct answer   -> True
      - answerable + incorrect answer -> False
      - answerable + abstained    -> False (a missed opportunity, not
        misinformation — but still not "success" by this simple rule; see
        README.md's Agent Routing section for this as a named caveat)
    """
    if final_answer is None:
        return not item.is_answerable
    if not item.is_answerable:
        return False
    return bool(correct)


# =========================================================================
# Generation + feature extraction
# =========================================================================

def generate_and_score(
    model,
    tokenizer,
    engineer: AttentionFeatureEngineer,
    entropy_extractor: EntropyFeatureExtractor,
    question_text: str,
    max_new_tokens: int = 40,
    device: str = "cpu",
) -> Tuple[str, np.ndarray]:
    """
    Generate a greedy answer to `question_text` (which may already have
    context prepended by the caller — see run_agent_task), then extract the
    same 24D attention+entropy feature vector used throughout the repo by
    teacher-forcing over the (prompt, generated_text) pair. Mirrors
    demo/app.py::score_text's generate-then-score pattern, but greedy
    (do_sample=False) rather than sampled, for benchmark reproducibility.
    """
    import torch

    prompt_only, _ = build_prompt_and_text(tokenizer, question_text, "")
    inputs = tokenizer(prompt_only, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    prompt, text = build_prompt_and_text(tokenizer, question_text, answer)
    attentions, context_len = extract_attention_from_model(text, model, tokenizer, device, prompt=prompt)
    attn_feats = engineer.extract(attentions, context_len)
    logits, token_ids, answer_start = extract_logits_from_model(text, model, tokenizer, device, prompt=prompt)
    entropy_feats = entropy_extractor.extract(logits, answer_start=answer_start, target_ids=token_ids)

    features = np.concatenate([attn_feats, entropy_feats])
    return answer, features


def _with_context_question(question: str, context: str) -> str:
    """Builds the question string fed to generate_and_score after
    `retrieve` — reuses build_prompt_and_text unchanged by folding the
    context into its `question` argument rather than adding a new
    prompt-construction branch."""
    return f"Context: {context}\n\nQuestion: {question}"


# =========================================================================
# Single-item and batch orchestration
# =========================================================================

def run_agent_task(
    item: SquadItem,
    model,
    tokenizer,
    engineer: AttentionFeatureEngineer,
    entropy_extractor: EntropyFeatureExtractor,
    policy: RoutingPolicy,
    max_new_tokens: int = 40,
    device: str = "cpu",
) -> AgentTraceRecord:
    """Run one SQuAD item through the full answer/retrieve/abstain flow and
    return its logged, graded trace."""
    t0 = time.perf_counter()

    answer_no_ctx, feats_no_ctx = generate_and_score(
        model, tokenizer, engineer, entropy_extractor, item.question,
        max_new_tokens=max_new_tokens, device=device,
    )
    decision = policy.decide(feats_no_ctx)
    scores = {"no_context": decision.p_hallucination}
    action_sequence = [decision.action.value]
    final_answer: Optional[str] = None

    if decision.action == AgentAction.ANSWER:
        final_answer = answer_no_ctx
    elif decision.action == AgentAction.RETRIEVE:
        context = retrieve_context(item)
        answer_ctx, feats_ctx = generate_and_score(
            model, tokenizer, engineer, entropy_extractor,
            _with_context_question(item.question, context),
            max_new_tokens=max_new_tokens, device=device,
        )
        decision2 = policy.decide_after_retrieval(feats_ctx)
        scores["with_context"] = decision2.p_hallucination
        action_sequence.append(decision2.action.value)
        if decision2.action == AgentAction.ANSWER:
            final_answer = answer_ctx
    # else: AgentAction.ABSTAIN — final_answer stays None

    latency_s = time.perf_counter() - t0

    correct = _grade_correctness(final_answer, item) if final_answer is not None else None
    groundedness = (
        _grade_groundedness(final_answer, item, action_sequence) if final_answer is not None else None
    )
    task_success = _task_success(item, final_answer, correct)

    return AgentTraceRecord(
        query=item.question,
        is_answerable=item.is_answerable,
        has_context_available=True,
        action_sequence=action_sequence,
        final_answer=final_answer,
        correct=correct,
        groundedness=groundedness,
        task_success=task_success,
        scores=scores,
        latency_s=latency_s,
        sample_id=item.sample_id,
    )


def run_all_tasks(
    items: List[SquadItem],
    model,
    tokenizer,
    policy: RoutingPolicy,
    max_new_tokens: int = 40,
    context_length: int = 32,
    device: str = "cpu",
) -> List[AgentTraceRecord]:
    """Runs run_agent_task over every item, skipping (and logging) any that
    raise — matches pipeline.py::run_real_pipeline's failure handling."""
    engineer = AttentionFeatureEngineer(context_length=context_length)
    entropy_extractor = EntropyFeatureExtractor()

    records: List[AgentTraceRecord] = []
    for i, item in enumerate(items):
        try:
            records.append(run_agent_task(
                item, model, tokenizer, engineer, entropy_extractor, policy,
                max_new_tokens=max_new_tokens, device=device,
            ))
        except Exception as e:
            print(f"  Warning: item {item.sample_id} failed — {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} processed")
    return records


# =========================================================================
# Calibration-set construction (labels the local model's OWN no-context
# generations against SQuAD gold answers — SQuAD has no pre-written
# "hallucinated answer" the way HaluEval does)
# =========================================================================

def build_calibration_set(
    items: List[SquadItem],
    model,
    tokenizer,
    max_new_tokens: int = 40,
    context_length: int = 32,
    device: str = "cpu",
    use_context: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an answer for each calibration item, grade it against SQuAD's
    gold answers, and return (X, y) — y=1 (hallucinated) if the answer was
    wrong (or the item is unanswerable and the model answered anyway), y=0
    (correct) otherwise — ready for CalibratedEntropyDetector.fit(),
    matching the rest of the repo's correct=0/hallucinated=1 convention.

    Parameters
    ----------
    use_context : bool
        If True, generate WITH the item's gold context prepended (a
        reading-comprehension task) instead of from parametric memory
        alone. This exists because the first no-context run on Pythia-160m
        graded only 2/220 (0.9%) calibration answers correct — too few for
        CalibratedEntropyDetector to fit a meaningful reference distribution
        — since a 160M base model has essentially no reliable open-domain
        factual recall. With-context calibration is a much easier task
        (extract from a given passage) and should produce a properly-sized
        "correct" reference set instead.
    """
    engineer = AttentionFeatureEngineer(context_length=context_length)
    entropy_extractor = EntropyFeatureExtractor()

    X_list, y_list = [], []
    for i, item in enumerate(items):
        try:
            question_text = (
                _with_context_question(item.question, item.context) if use_context else item.question
            )
            answer, feats = generate_and_score(
                model, tokenizer, engineer, entropy_extractor, question_text,
                max_new_tokens=max_new_tokens, device=device,
            )
            if not np.all(np.isfinite(feats)):
                continue
            correct = _grade_correctness(answer, item)
            X_list.append(feats)
            y_list.append(0.0 if correct else 1.0)
        except Exception as e:
            print(f"  Warning: calibration item {item.sample_id} failed — {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} processed")

    return np.array(X_list), np.array(y_list)
