"""
Detector-Stacking Ensemble
=============================

Answers a narrow question: does stacking this repo's existing detectors
(CalibratedEntropyDetector, LogReg, MLP, BiLSTM, BlackBoxEntropyDetector) —
or explicitly modeling their disagreement — beat the best individual
detector on held-out matched-pair HaluEval data?

Evaluated with a LEAKAGE-SAFE NESTED cross-validation: for each outer fold,
base detectors are trained only on inner-fold splits of the outer-training
data to produce out-of-fold meta-features (no base detector ever sees a row
it later predicts on for meta-training), the meta-model trains on those,
base detectors are then retrained on the full outer-training set, and only
that retrained set predicts the untouched outer-test fold. See
nested_cv_stacking()'s docstring for the exact step order.

Three variants (see README.md's Detector Stacking Ensemble section for
results): "uniform_mean" (plain averaging), "logistic_stacker"
(data-driven weights over per-detector probability+rank features only),
"stacker_disagreement" (adds mean/std/range/pairwise-disagreement features
on top — tests whether detector *conflict* itself is predictive).

Stop rule, decided before running anything:
    Win: outer-fold AUROC AND PR-AUC both beat the best base detector, with
        paired_bootstrap_delta_ci excluding zero, calibration no worse.
    Partial win: AUROC flat, but calibration/precision-at-recall/AURC improves.
    No win: no robust improvement -- keep the best base detector, report
        that finding directly (redundant detectors is real evidence).

Usage:
    python ensemble.py --num_samples 400 --model EleutherAI/pythia-160m \\
        --results results/halueval_pythia160m_ensemble_n400.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from abstention import area_under_risk_coverage, risk_coverage_curve
from blackbox_detector import BlackBoxEntropyDetector, extract_blackbox_features, simulate_topk_from_full_logits
from calibrated_entropy_detector import CalibratedEntropyDetector
from detector import HallucinationDetector, LogisticRegression, compute_auroc
from entropy_baselines import EntropyFeatureExtractor
from feature_engineer import AttentionFeatureEngineer
from pipeline import build_prompt_and_text, extract_attention_from_model, extract_logits_from_model

BASE_DETECTOR_NAMES = ["calibrated_entropy", "logistic", "mlp", "bilstm", "blackbox"]
VARIANTS = ("uniform_mean", "logistic_stacker", "stacker_disagreement")


# =========================================================================
# Feature extraction (reuses pipeline.py's primitives; same per-sample loop
# and shared-shuffle pattern as pipeline.py::run_real_pipeline, so X/
# X_blackbox/X_seq_all/y stay index-aligned the same way)
# =========================================================================

def extract_features_for_halueval(
    model_name: str = "EleutherAI/pythia-160m",
    num_samples: int = 400,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, X_blackbox, X_seq_all, y), shuffled together."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data_generator import DataGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = DataGenerator.from_halueval(num_samples=num_samples, seed=seed)
    clean = [s for s in samples if s.label != "ambiguous"]
    print(f"  {len(clean)} non-ambiguous samples")

    print(f"  Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    engineer = AttentionFeatureEngineer(context_length=32)
    entropy_extractor = EntropyFeatureExtractor()

    X_attn_list, X_entropy_list, X_blackbox_list, X_seq_list, y_list = [], [], [], [], []
    failed = 0
    for i, sample in enumerate(clean):
        try:
            prompt, text = build_prompt_and_text(tokenizer, sample.question, sample.model_answer)
            attentions, context_len = extract_attention_from_model(text, model, tokenizer, device, prompt=prompt)
            attn_feats = engineer.extract(attentions, context_len)
            seq_feats = engineer.extract_layer_sequence(attentions)

            logits, token_ids, answer_start = extract_logits_from_model(text, model, tokenizer, device, prompt=prompt)
            entropy_feats = entropy_extractor.extract(logits, answer_start=answer_start, target_ids=token_ids)
            topk_seq = simulate_topk_from_full_logits(logits[answer_start:], token_ids[answer_start:], top_k=5)
            blackbox_feats = extract_blackbox_features(topk_seq)

            if not (
                np.all(np.isfinite(attn_feats)) and np.all(np.isfinite(seq_feats))
                and np.all(np.isfinite(entropy_feats)) and np.all(np.isfinite(blackbox_feats))
            ):
                raise ValueError("non-finite feature value")

            X_attn_list.append(attn_feats)
            X_entropy_list.append(entropy_feats)
            X_blackbox_list.append(blackbox_feats)
            X_seq_list.append(seq_feats)
            y_list.append(1.0 if sample.label == "hallucinated" else 0.0)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Warning: sample {i} failed — {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clean)} processed (failed: {failed})")

    X = np.hstack([np.array(X_attn_list), np.array(X_entropy_list)])
    X_blackbox = np.array(X_blackbox_list)
    X_seq_all = np.array(X_seq_list)
    y = np.array(y_list)
    print(f"  Feature matrix: {X.shape}  blackbox: {X_blackbox.shape}  seq: {X_seq_all.shape}  failed: {failed}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return X[idx], X_blackbox[idx], X_seq_all[idx], y[idx]


# =========================================================================
# Base detectors
# =========================================================================

def fit_base_detectors_and_predict(
    X_train: np.ndarray, X_blackbox_train: np.ndarray, X_seq_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, X_blackbox_test: np.ndarray, X_seq_test: np.ndarray,
    bilstm_epochs: int = 60,
) -> Dict[str, np.ndarray]:
    """Fits all 5 base detectors on the *_train arrays, returns their
    predict_proba output on *_test, keyed by name. The single place both
    the inner-fold OOF loop and the outer-fold refit-and-predict step call
    into, so the two never drift out of sync."""
    probs: Dict[str, np.ndarray] = {}

    det_ce = CalibratedEntropyDetector()
    det_ce.fit(X_train, y_train)
    probs["calibrated_entropy"] = det_ce.predict_proba(X_test)

    det_lr = HallucinationDetector(classifier_type="logistic")
    det_lr.fit(X_train, y_train)
    probs["logistic"] = det_lr.predict_proba(X_test)

    det_mlp = HallucinationDetector(classifier_type="mlp")
    det_mlp.fit(X_train, y_train)
    probs["mlp"] = det_mlp.predict_proba(X_test)

    det_bilstm = HallucinationDetector(classifier_type="bilstm", epochs=bilstm_epochs)
    det_bilstm.fit_sequence(X_seq_train, y_train)
    probs["bilstm"] = det_bilstm.predict_proba_sequence(X_seq_test)

    det_bb = BlackBoxEntropyDetector()
    det_bb.fit(X_blackbox_train, y_train)
    probs["blackbox"] = det_bb.predict_proba(X_blackbox_test)

    return probs


# =========================================================================
# Meta-features
# =========================================================================

def build_meta_features(
    base_probs: Dict[str, np.ndarray], include_aggregate: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    base_probs: name -> (N,) array, all the same length and row order.

    Columns, in order: 5 raw probabilities, 5 per-detector percentile ranks
    (captures nonlinear miscalibration the raw score alone doesn't), then
    -- if include_aggregate -- 4 cross-detector columns: mean, std, range
    (max-min), and disagreement (mean pairwise absolute difference across
    all 5 detectors -- a distinct diversity signal from std, testing
    whether detector *conflict* itself is predictive of hallucination).
    """
    names = BASE_DETECTOR_NAMES
    raw = np.column_stack([base_probs[n] for n in names])  # (N, 5)
    n = raw.shape[0]

    ranks = np.empty_like(raw)
    for j in range(raw.shape[1]):
        order = np.argsort(np.argsort(raw[:, j]))
        ranks[:, j] = order / max(n - 1, 1)

    blocks = [raw, ranks]
    feature_names = [f"{name}_prob" for name in names] + [f"{name}_rank" for name in names]

    if include_aggregate:
        mean = raw.mean(axis=1, keepdims=True)
        std = raw.std(axis=1, keepdims=True)
        value_range = (raw.max(axis=1) - raw.min(axis=1)).reshape(-1, 1)

        k = raw.shape[1]
        pairwise_sum = np.zeros(n)
        pair_count = 0
        for a in range(k):
            for b in range(a + 1, k):
                pairwise_sum += np.abs(raw[:, a] - raw[:, b])
                pair_count += 1
        disagreement = (pairwise_sum / pair_count).reshape(-1, 1)

        blocks += [mean, std, value_range, disagreement]
        feature_names += ["mean_prob", "std_prob", "range_prob", "disagreement"]

    return np.hstack(blocks), feature_names


# =========================================================================
# Stratified fold splitting (duplicated in reduced form from
# pipeline.py::stratified_kfold_cv -- matches abstention.py's own
# established precedent of duplicating this split logic rather than
# sharing it, since it's a few lines and used at two nesting levels here)
# =========================================================================

def _stratified_folds(y: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """Returns k arrays of LOCAL indices into y (0..len(y)-1), stratified
    by class."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    return [np.concatenate([pos_folds[i], neg_folds[i]]) for i in range(k)]


# =========================================================================
# Nested CV stacking
# =========================================================================

def _run_nested_base_detector_fits(
    X: np.ndarray, X_blackbox: np.ndarray, X_seq: np.ndarray, y: np.ndarray,
    outer_k: int, inner_k: int, seed: int, bilstm_epochs: int,
) -> List[Dict[str, Any]]:
    """
    Does the expensive part ONCE, independent of which ensemble variant
    will consume it: for each outer fold, fits base detectors across inner
    folds to get out-of-fold meta-features over the outer-training set, then
    refits base detectors on the full outer-training set and predicts
    outer-test. Returns one dict per outer fold with everything a variant's
    meta-model step needs. Splitting this out from the meta-model step (see
    _stack_variant) means run_ensemble_experiment can evaluate all 3
    variants from ONE nested pass over the base detectors instead of three
    -- the variants only ever differ in the meta-model, never in how the
    base detectors are trained.
    """
    outer_folds = _stratified_folds(y, outer_k, seed)
    fold_data = []

    for outer_fold in range(outer_k):
        outer_test_idx = outer_folds[outer_fold]
        outer_train_idx = np.concatenate([outer_folds[j] for j in range(outer_k) if j != outer_fold])
        y_outer_train = y[outer_train_idx]

        # --- Inner loop: pooled out-of-fold meta-features over outer-train ---
        inner_folds_local = _stratified_folds(y_outer_train, inner_k, seed=seed + outer_fold + 1)
        oof_base_probs = {name: np.zeros(len(outer_train_idx)) for name in BASE_DETECTOR_NAMES}

        for inner_fold in range(inner_k):
            inner_val_local = inner_folds_local[inner_fold]
            inner_train_local = np.concatenate([inner_folds_local[j] for j in range(inner_k) if j != inner_fold])

            inner_train_global = outer_train_idx[inner_train_local]
            inner_val_global = outer_train_idx[inner_val_local]

            inner_probs = fit_base_detectors_and_predict(
                X[inner_train_global], X_blackbox[inner_train_global], X_seq[inner_train_global], y[inner_train_global],
                X[inner_val_global], X_blackbox[inner_val_global], X_seq[inner_val_global],
                bilstm_epochs=bilstm_epochs,
            )
            for name in BASE_DETECTOR_NAMES:
                oof_base_probs[name][inner_val_local] = inner_probs[name]

        # --- Retrain base detectors on the FULL outer-train set, predict outer-test ---
        outer_test_base_probs = fit_base_detectors_and_predict(
            X[outer_train_idx], X_blackbox[outer_train_idx], X_seq[outer_train_idx], y_outer_train,
            X[outer_test_idx], X_blackbox[outer_test_idx], X_seq[outer_test_idx],
            bilstm_epochs=bilstm_epochs,
        )

        fold_data.append({
            "outer_test_idx": outer_test_idx,
            "outer_train_idx": outer_train_idx,
            "y_outer_train": y_outer_train,
            "oof_base_probs": oof_base_probs,
            "outer_test_base_probs": outer_test_base_probs,
        })

    return fold_data


def _stack_variant(fold_data: List[Dict[str, Any]], y: np.ndarray, variant: str) -> Dict[str, Any]:
    """Consumes _run_nested_base_detector_fits's cached per-fold base-
    detector predictions to produce one ensemble variant's pooled,
    out-of-sample probabilities -- the (cheap) part that actually differs
    between variants."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}. Choose from {VARIANTS}")

    n = len(y)
    pooled_ensemble_probs = np.zeros(n)
    pooled_base_probs = {name: np.zeros(n) for name in BASE_DETECTOR_NAMES}
    include_aggregate = variant == "stacker_disagreement"
    meta_feature_names: List[str] = []

    for fd in fold_data:
        meta_X_train, meta_feature_names = build_meta_features(fd["oof_base_probs"], include_aggregate=include_aggregate)
        meta_mean = meta_X_train.mean(axis=0)
        meta_std = meta_X_train.std(axis=0) + 1e-8

        meta_model: Optional[LogisticRegression] = None
        if variant != "uniform_mean":
            meta_model = LogisticRegression()
            meta_model.fit((meta_X_train - meta_mean) / meta_std, fd["y_outer_train"])

        meta_X_test, _ = build_meta_features(fd["outer_test_base_probs"], include_aggregate=include_aggregate)
        if variant == "uniform_mean":
            ensemble_probs_fold = meta_X_test[:, :len(BASE_DETECTOR_NAMES)].mean(axis=1)
        else:
            ensemble_probs_fold = meta_model.predict_proba((meta_X_test - meta_mean) / meta_std)

        outer_test_idx = fd["outer_test_idx"]
        pooled_ensemble_probs[outer_test_idx] = ensemble_probs_fold
        for name in BASE_DETECTOR_NAMES:
            pooled_base_probs[name][outer_test_idx] = fd["outer_test_base_probs"][name]

    return {
        "ensemble_probs": pooled_ensemble_probs,
        "base_probs": pooled_base_probs,
        "y": y,
        "variant": variant,
        "meta_feature_names": meta_feature_names,
    }


def nested_cv_stacking(
    X: np.ndarray,
    X_blackbox: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    outer_k: int = 5,
    inner_k: int = 5,
    seed: int = 42,
    variant: str = "stacker_disagreement",
    bilstm_epochs: int = 60,
) -> Dict[str, Any]:
    """
    Leakage-safe nested CV for ONE ensemble variant. For each outer fold:
      1. Inner stratified folds partition the outer-training rows.
      2. Base detectors train on inner-train, predict on inner-val, per
         inner fold -> pooled out-of-fold meta-features covering the whole
         outer-train set exactly once (no base detector ever predicts on
         rows it trained on).
      3. Meta-model trains on those OOF meta-features (skipped for
         "uniform_mean", which just averages).
      4. Base detectors are RETRAINED on the full outer-training set (not
         the smaller inner-train pieces).
      5. Those refit base detectors predict outer-test; the same
         meta-feature construction + the already-trained meta-model scores
         it.
    Pooled outer-test predictions across all outer folds are the final,
    honest, out-of-sample probabilities -- for both the ensemble AND each
    base detector individually (from the SAME outer folds, so "best
    individual detector" is measured on identical splits, not a separate,
    differently-seeded run).

    Returns dict with "ensemble_probs", "base_probs" (name -> (N,) array),
    "y", "variant", "meta_feature_names".

    Evaluating multiple variants on the same data? Call
    _run_nested_base_detector_fits once and _stack_variant per variant
    instead (see run_ensemble_experiment) -- this function reruns the
    (expensive) base-detector fitting from scratch every call, which is
    correct but wasteful if you need more than one variant's numbers.
    """
    fold_data = _run_nested_base_detector_fits(X, X_blackbox, X_seq, y, outer_k, inner_k, seed, bilstm_epochs)
    return _stack_variant(fold_data, y, variant)


# =========================================================================
# Metrics -- independent of agent/calibration.py (that module is
# gitignored/local-only by design; this is committed, main-pipeline code)
# =========================================================================

@dataclass
class PRPoint:
    threshold: float
    precision: float
    recall: float


def precision_recall_curve_for_hallucination(
    probs: np.ndarray, y: np.ndarray, n_thresholds: int = 100
) -> List[PRPoint]:
    """Precision-recall curve for predicting HALLUCINATED (y==1) via
    {p >= threshold} -- matches this repo's existing AUROC/F1 convention
    throughout detector.py (higher p = more likely hallucinated), the
    opposite direction from the agent-routing MVP's PR curve (which
    predicts "correct" via low p -- a different, gitignored module for a
    different, severely-imbalanced task)."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    n_positive = int((y == 1).sum())

    thresholds = np.linspace(probs.min(), probs.max(), n_thresholds)
    points = []
    for tau in thresholds:
        predicted = probs >= tau
        n_predicted = int(predicted.sum())
        precision = float((y[predicted] == 1).mean()) if n_predicted else float("nan")
        recall = float((y[predicted] == 1).sum() / n_positive) if n_positive else float("nan")
        points.append(PRPoint(threshold=float(tau), precision=precision, recall=recall))
    return points


def average_precision(points: List[PRPoint]) -> float:
    """PR-AUC: trapezoidal integral of precision over recall, mirroring
    abstention.py::area_under_risk_coverage's style (same trapz fallback,
    same "integrate the tradeoff into one number" framing)."""
    valid = [p for p in points if not np.isnan(p.precision) and not np.isnan(p.recall)]
    if len(valid) < 2:
        return float("nan")
    valid = sorted(valid, key=lambda p: p.recall)
    recalls = np.array([p.recall for p in valid])
    precisions = np.array([p.precision for p in valid])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(precisions, recalls))


def precision_at_recall(points: List[PRPoint], target_recall: float) -> float:
    """Precision at the lowest-threshold point whose recall is still >=
    target_recall (the standard "precision at a safety-relevant recall
    floor" operating-point read)."""
    valid = [p for p in points if not np.isnan(p.precision) and p.recall >= target_recall]
    if not valid:
        return float("nan")
    # Among points meeting the recall floor, the highest-threshold one
    # (tightest, since PR points are already threshold-ascending, recall
    # descending) gives the best achievable precision at that floor.
    return max(valid, key=lambda p: p.threshold).precision


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probability and the true
    hallucination label -- a proper scoring rule for probability quality,
    not just discrimination (unlike AUROC)."""
    return float(np.mean((np.asarray(probs) - np.asarray(y)) ** 2))


def log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bins predictions by confidence, weights each bin's
    |accuracy - mean confidence| by its share of the data."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(in_bin):
            continue
        bin_confidence = probs[in_bin].mean()
        bin_accuracy = y[in_bin].mean()  # fraction actually hallucinated in this confidence bin
        ece += (in_bin.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def paired_bootstrap_delta_ci(
    probs_a: np.ndarray, probs_b: np.ndarray, y: np.ndarray,
    n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap CI for AUROC(a) - AUROC(b), resampling the SAME indices for
    both arrays each iteration (paired), not independently -- gives a
    tighter, statistically correct CI for the delta itself, unlike
    differencing two separately-bootstrapped CIs
    (pipeline.py::bootstrap_auroc_ci resamples one array at a time and
    doesn't pair, so it can't answer "is the delta itself significant").
    Returns (mean_delta, (lo, hi)).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        auroc_a = compute_auroc(probs_a[idx], y_b)
        auroc_b = compute_auroc(probs_b[idx], y_b)
        deltas.append(auroc_a - auroc_b)
    deltas = np.array(deltas)
    lo = float(np.percentile(deltas, 100 * (1 - ci) / 2))
    hi = float(np.percentile(deltas, 100 * (1 - (1 - ci) / 2)))
    return float(deltas.mean()), (lo, hi)


# =========================================================================
# Reporting
# =========================================================================

def _detector_report(name: str, probs: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    points = precision_recall_curve_for_hallucination(probs, y)
    return {
        "name": name,
        "auroc": compute_auroc(probs, y),
        "pr_auc": average_precision(points),
        "brier": brier_score(probs, y),
        "log_loss": log_loss(probs, y),
        "ece": expected_calibration_error(probs, y),
        "precision_at_recall_0.95": precision_at_recall(points, 0.95),
    }


def run_ensemble_experiment(
    model_name: str = "EleutherAI/pythia-160m",
    num_samples: int = 400,
    outer_k: int = 5,
    inner_k: int = 5,
    seed: int = 42,
    bilstm_epochs: int = 60,
    results_path: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"Extracting features for {model_name} (n={num_samples})...")
    X, X_blackbox, X_seq, y = extract_features_for_halueval(model_name, num_samples, seed)

    # The expensive part (fitting 5 base detectors across outer_k*(inner_k+1)
    # folds) runs exactly ONCE here and is reused by all 3 variants below --
    # they only ever differ in the meta-model step, never in how the base
    # detectors are trained. See _run_nested_base_detector_fits/_stack_variant.
    print(f"\nFitting base detectors across nested CV (outer_k={outer_k}, inner_k={inner_k})...")
    t0 = time.perf_counter()
    fold_data = _run_nested_base_detector_fits(X, X_blackbox, X_seq, y, outer_k, inner_k, seed, bilstm_epochs)
    base_fit_elapsed = time.perf_counter() - t0
    print(f"  Base-detector fitting took {base_fit_elapsed:.1f}s")

    variant_reports: Dict[str, Any] = {}
    base_report: Optional[Dict[str, Any]] = None
    best_base_probs: Optional[np.ndarray] = None
    best_base_name: str = ""
    ensemble_probs_by_variant: Dict[str, np.ndarray] = {}

    for variant in VARIANTS:
        print(f"\nStacking variant={variant}...")
        t0 = time.perf_counter()
        result = _stack_variant(fold_data, y, variant)
        elapsed = time.perf_counter() - t0
        # Latency reported here is the meta-model's own overhead; base
        # detector fitting is a fixed, amortized cost shared across variants,
        # reported separately as base_fit_elapsed in the summary.
        latency_per_sample = elapsed / len(y)

        ensemble_report = _detector_report(f"ensemble_{variant}", result["ensemble_probs"], y)
        ensemble_report["latency_s_per_sample"] = latency_per_sample
        variant_reports[variant] = ensemble_report
        ensemble_probs_by_variant[variant] = result["ensemble_probs"]

        if base_report is None:
            # Base-detector reports are identical across variants (same
            # outer folds, same base-detector fits) -- compute once, so
            # "best individual detector" is measured on the exact same
            # splits as every ensemble variant.
            base_report = {name: _detector_report(name, result["base_probs"][name], y) for name in BASE_DETECTOR_NAMES}
            best_base_name = max(base_report, key=lambda n: base_report[n]["auroc"])
            best_base_probs = result["base_probs"][best_base_name]

    # Delta over the best individual base detector, with a PAIRED bootstrap
    # CI (same resampled indices for both arrays each iteration) -- this is
    # the number the stop rule's "win" criterion actually checks.
    for variant in VARIANTS:
        delta_mean, delta_ci = paired_bootstrap_delta_ci(
            ensemble_probs_by_variant[variant], best_base_probs, y, seed=seed,
        )
        variant_reports[variant]["auroc_delta_vs_best_base"] = delta_mean
        variant_reports[variant]["auroc_delta_vs_best_base_ci"] = list(delta_ci)

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "num_samples": num_samples,
        "num_samples_used": int(len(y)),
        "outer_k": outer_k,
        "inner_k": inner_k,
        "base_fit_elapsed_s": base_fit_elapsed,
        "base_detectors": base_report,
        "best_base_detector": best_base_name,
        "ensemble_variants": variant_reports,
    }

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults summary saved to {results_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Detector-stacking ensemble on matched-pair HaluEval")
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--outer_k", type=int, default=5)
    parser.add_argument("--inner_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bilstm_epochs", type=int, default=60,
                         help="Lower this (e.g. 20) if nested-CV runtime is prohibitive -- BiLSTM is "
                              "the slowest base detector to refit across outer_k*(inner_k+1) fits.")
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  DETECTOR-STACKING ENSEMBLE — matched-pair HaluEval")
    print("=" * 60)

    summary = run_ensemble_experiment(
        model_name=args.model,
        num_samples=args.num_samples,
        outer_k=args.outer_k,
        inner_k=args.inner_k,
        seed=args.seed,
        bilstm_epochs=args.bilstm_epochs,
        results_path=args.results,
    )

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Best base detector: {summary['best_base_detector']} "
          f"(AUROC {summary['base_detectors'][summary['best_base_detector']]['auroc']:.4f})")
    for variant, report in summary["ensemble_variants"].items():
        print(f"  {variant:<22} AUROC {report['auroc']:.4f}  PR-AUC {report['pr_auc']:.4f}  "
              f"ECE {report['ece']:.4f}  Brier {report['brier']:.4f}")


if __name__ == "__main__":
    main()
