"""
End-to-End Pipeline — Hallucination Detection
=================================================

Orchestrates the full workflow:
    1. Load labeled samples (from DataGenerator or file)
    2. Extract attention tensors from a local model
    3. Compute multi-family features
    4. Train/evaluate the lightweight detector

Usage:
    # Synthetic demo (no model or API needed)
    python pipeline.py --synthetic --num_samples 1000

    # HaluEval benchmark (no API — pip install datasets)
    python pipeline.py --halueval --num_samples 500 --model EleutherAI/pythia-160m

    # Full pipeline with self-generated data (requires ANTHROPIC_API_KEY)
    python pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m

    # Save trained detector
    python pipeline.py --halueval --num_samples 500 --save detector.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from feature_engineer import AttentionFeatureEngineer, FeatureConfig
from detector import HallucinationDetector, BiLSTMDetector, DetectorMetrics
from entropy_baselines import EntropyFeatureExtractor
from calibrated_entropy_detector import CalibratedEntropyDetector
from blackbox_detector import BlackBoxEntropyDetector, simulate_topk_from_full_logits, extract_blackbox_features


# =========================================================================
# Attention extraction (model-agnostic)
# =========================================================================

def extract_attention_from_model(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    prompt: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Run a forward pass and extract attention tensors.

    Parameters
    ----------
    text : str
        Full text (prompt + answer) to run through the model.
    prompt : str, optional
        The prompt-only prefix (e.g. "Question: ...\\nAnswer:"). If given, it
        is tokenized separately so the returned `context_length` is the true
        prompt length. If omitted, `context_length` falls back to the full
        `text`'s token count (previous behavior) — which is NOT the prompt
        length and will make lookback-ratio features measure "attention to
        the entire sequence" rather than "attention to context", so pass
        `prompt` whenever the caller has it (see extract_logits_from_model,
        which already does the same split for its own `answer_start`).

    Returns
    -------
    attentions : np.ndarray, shape (L, H, T, T)
    context_length : int
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(device)

    if prompt is not None:
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        context_length = prompt_inputs["input_ids"].shape[1]
    else:
        context_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    # Stack attention layers → (L, H, T, T)
    attn_list = []
    for layer_attn in outputs.attentions:
        attn_list.append(layer_attn[0].detach().cpu().numpy())

    return np.stack(attn_list), context_length


def extract_logits_from_model(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    prompt: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run a teacher-forcing forward pass and extract next-token logits.

    Parameters
    ----------
    text : str
        Full text (prompt + answer) to run through the model.
    prompt : str, optional
        The prompt-only prefix (e.g. "Question: ...\\nAnswer:"). If given,
        it is tokenized separately to compute `answer_start` precisely.
        If omitted, `answer_start` is 0 (caller treats the whole sequence
        as the span of interest).

    Returns
    -------
    logits : np.ndarray, shape (T, V)
        logits[t] is the model's predictive distribution for the token at
        position t+1 (teacher forcing) — i.e. already shifted/aligned with
        `token_ids`.
    token_ids : np.ndarray, shape (T,)
        The actually-realized token id that logits[t] should have predicted.
    answer_start : int
        Index into `logits`/`token_ids` where the answer span begins (0 if
        `prompt` not given).
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    raw_logits = outputs.logits[0].detach().cpu().numpy()      # (T_full, V)
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()  # (T_full,)

    logits = raw_logits[:-1]      # logits[t] predicts input_ids[t+1]
    token_ids = input_ids[1:]

    answer_start = 0
    if prompt is not None:
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        answer_start = max(prompt_len - 1, 0)

    return logits, token_ids, answer_start


def build_prompt_and_text(tokenizer, question: str, answer: str) -> Tuple[str, str]:
    """
    Build the (prompt, full_text) pair fed to extract_attention_from_model /
    extract_logits_from_model, branching on whether `tokenizer` has a chat
    template.

    Base models (e.g. Pythia) have no chat_template and were pretrained on
    raw completion text, so a plain "Question: ...\\nAnswer: ..." string is
    on-distribution for them. Instruct-tuned models (e.g. Qwen2.5-Instruct)
    are fine-tuned specifically to expect their own chat-formatted prompt
    (special tokens marking turns/roles); feeding them the base-model-style
    string instead would put every attention/entropy feature this repo
    extracts off-distribution — the model would be "confused" by the input
    format itself, which is a different effect than genuine hallucination
    uncertainty and would contaminate any comparison between the two model
    types. Detecting the chat template and branching here (once, at the
    prompt-construction call site) keeps extract_attention_from_model and
    extract_logits_from_model themselves model-agnostic — they already just
    take whatever `prompt`/`text` strings they're given.

    Returns
    -------
    prompt : str
        The prompt-only prefix (pass to `prompt=` in extract_*_from_model).
    text : str
        prompt + answer (pass as `text` in extract_*_from_model).
    """
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = prompt + answer
    else:
        prompt = f"Question: {question}\nAnswer:"
        text = f"{prompt} {answer}"
    return prompt, text


# =========================================================================
# Synthetic data generation (no model needed)
# =========================================================================

def generate_synthetic_dataset(
    num_samples: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic multi-family features + labels.

    Simulates the feature distributions we'd expect from a real model:
    - Hallucinated outputs: higher entropy, lower lookback, higher freq energy
    - Correct outputs: lower entropy, higher lookback, lower freq energy
    """
    rng = np.random.default_rng(seed)
    D = 18  # full feature vector dimension (5 families)
    y = (rng.random(num_samples) > 0.6).astype(float)  # 40% hallucinated

    X = np.zeros((num_samples, D))

    for i in range(num_samples):
        if y[i] == 1:  # hallucinated
            # Entropy features: higher (diffuse attention)
            X[i, 0:3] = rng.normal([3.8, 5.0, 1.0], [0.5, 0.8, 0.3])
            # Lookback features: lower (not grounding in context)
            X[i, 3:7] = rng.normal([0.3, 0.1, 0.15, 0.8], [0.1, 0.05, 0.05, 0.1])
            # Frequency features: higher energy (unstable attention)
            X[i, 7:11] = rng.normal([0.45, 0.65, 3.0, 3.5], [0.1, 0.1, 0.5, 0.3])
            # Spectral features: lower Fiedler (fragmented graph)
            X[i, 11:15] = rng.normal([0.15, 0.08, 0.05, 8.0], [0.05, 0.03, 0.02, 1.0])
            # Cross-layer KL: higher (layers disagree)
            X[i, 15:18] = rng.normal([3.5, 1.2, 0.8], [0.8, 0.4, 0.2])
        else:  # correct
            X[i, 0:3] = rng.normal([2.2, 3.5, 0.5], [0.4, 0.6, 0.2])
            X[i, 3:7] = rng.normal([0.7, 0.4, 0.10, 0.6], [0.1, 0.1, 0.04, 0.1])
            X[i, 7:11] = rng.normal([0.25, 0.40, 2.0, 3.0], [0.08, 0.1, 0.4, 0.3])
            X[i, 11:15] = rng.normal([0.35, 0.05, 0.12, 5.0], [0.08, 0.02, 0.03, 0.8])
            X[i, 15:18] = rng.normal([1.5, 0.5, 0.3], [0.5, 0.2, 0.1])

    return X, y


# =========================================================================
# Evaluation and reporting
# =========================================================================

def bootstrap_auroc_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for AUROC.

    Resamples (probs, labels) with replacement n_boot times and computes
    the AUROC distribution. Returns the (lower, upper) percentile bounds.
    """
    rng = np.random.default_rng(seed)
    aurocs = []
    N = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        p_b, y_b = probs[idx], labels[idx]
        if y_b.sum() == 0 or y_b.sum() == N:
            continue
        # Mann-Whitney AUROC
        pos = p_b[y_b == 1]
        neg = p_b[y_b == 0]
        auroc = float(np.mean(pos[:, None] > neg[None, :]))
        aurocs.append(auroc)
    lo = np.percentile(aurocs, 100 * (1 - ci) / 2)
    hi = np.percentile(aurocs, 100 * (1 - (1 - ci) / 2))
    return float(lo), float(hi)


def stratified_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    classifier_type: str = "logistic",
    seed: int = 42,
    detector_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[float, Tuple[float, float]]:
    """
    Stratified k-fold cross-validation with bootstrap AUROC CI.

    Maintains class balance across folds (important for imbalanced data).
    Returns mean AUROC and 95% CI from bootstrap on held-out predictions.

    Parameters
    ----------
    classifier_type : str
        Used to build a HallucinationDetector when `detector_factory` is
        not given (existing behavior, unchanged).
    detector_factory : Callable[[], Any], optional
        If given, called once per fold to build a fresh detector exposing
        fit(X, y)/predict_proba(X) — lets CalibratedEntropyDetector,
        BlackBoxEntropyDetector, etc. run through this same CV/bootstrap
        harness without special-casing.
    """
    rng = np.random.default_rng(seed)

    # Stratified split: interleave pos/neg samples across folds
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    all_probs  = np.zeros(len(y))
    all_labels = y.copy()

    for fold in range(k):
        val_idx   = np.concatenate([pos_folds[fold], neg_folds[fold]])
        train_idx = np.concatenate([
            np.concatenate([pos_folds[j] for j in range(k) if j != fold]),
            np.concatenate([neg_folds[j] for j in range(k) if j != fold]),
        ])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val       = X[val_idx]

        det = detector_factory() if detector_factory is not None \
            else HallucinationDetector(classifier_type=classifier_type)
        det.fit(X_tr, y_tr)
        all_probs[val_idx] = det.predict_proba(X_val)

    # AUROC on all out-of-fold predictions
    from scipy.stats import mannwhitneyu
    pos_p = all_probs[y == 1]
    neg_p = all_probs[y == 0]
    stat, _ = mannwhitneyu(pos_p, neg_p, alternative="greater")
    mean_auroc = float(stat / (len(pos_p) * len(neg_p)))

    lo, hi = bootstrap_auroc_ci(all_probs, all_labels, seed=seed)
    return mean_auroc, (lo, hi)


def print_metrics(metrics: DetectorMetrics, title: str = "Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}")
    print(f"  AUROC:     {metrics.auroc:.4f}")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  FPR:       {metrics.false_positive_rate:.4f}")
    print(f"  Samples:   {metrics.num_samples}")
    print(f"{'═' * 50}")


DEFAULT_ABLATION_FAMILIES: Dict[str, slice] = {
    "entropy": slice(0, 3),
    "lookback": slice(3, 7),
    "frequency": slice(7, 11),
    "spectral": slice(11, 15),
    "cross_layer_kl": slice(15, 18),
}


def ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    families: Optional[Dict[str, slice]] = None,
) -> Dict[str, Any]:
    """
    Feature family ablation: train with each family removed to measure
    its contribution.

    Parameters
    ----------
    families : Dict[str, slice], optional
        Maps family name -> column slice within X. Defaults to the 5
        attention families (DEFAULT_ABLATION_FAMILIES); pass an extended
        dict (e.g. adding "entropy_token": slice(18, 24)) once X has been
        concatenated with additional feature blocks.

    Returns
    -------
    Dict[str, Any]
        {"full_auroc": float, "families": {name: {"auroc": float, "delta": float}}}
        — the same numbers that get printed, for callers (e.g. run_real_pipeline's
        results_path) that want to persist them instead of just reading the log.
    """
    families = families if families is not None else DEFAULT_ABLATION_FAMILIES

    N = len(y)
    split = int(0.7 * N)

    # Baseline: all features
    det_full = HallucinationDetector(classifier_type="logistic")
    det_full.fit(X[:split], y[:split])
    full_auroc = det_full.evaluate(X[split:], y[split:]).auroc

    print(f"\n{'═' * 50}")
    print(f"  FEATURE FAMILY ABLATION")
    print(f"{'═' * 50}")
    print(f"  Full model ({X.shape[1]} features): AUROC = {full_auroc:.4f}")
    print(f"  {'─' * 46}")

    results: Dict[str, Any] = {"full_auroc": full_auroc, "families": {}}

    for family, s in families.items():
        # Remove this family
        mask = np.ones(X.shape[1], dtype=bool)
        mask[s] = False
        X_ablated = X[:, mask]

        det = HallucinationDetector(classifier_type="logistic")
        det.fit(X_ablated[:split], y[:split])
        ablated_auroc = det.evaluate(X_ablated[split:], y[split:]).auroc

        delta = full_auroc - ablated_auroc
        results["families"][family] = {"auroc": ablated_auroc, "delta": delta}
        direction = "↓" if delta > 0.001 else "→"
        print(f"  Without {family:<16}: AUROC = {ablated_auroc:.4f}  ({direction} {delta:+.4f})")

    print(f"{'═' * 50}")
    return results


# =========================================================================
# Main pipeline
# =========================================================================

def run_synthetic_demo(num_samples: int = 500, seed: int = 42):
    """Run the full pipeline on synthetic data (no model/API needed)."""

    print(f"Generating synthetic dataset: {num_samples} samples...")
    X, y = generate_synthetic_dataset(num_samples, seed)

    n_hall = int(y.sum())
    print(f"  Hallucinated: {n_hall} ({n_hall/len(y)*100:.0f}%)")
    print(f"  Correct: {len(y) - n_hall} ({(len(y) - n_hall)/len(y)*100:.0f}%)")
    print(f"  Features per sample: {X.shape[1]}")

    # Train/test split
    split = int(0.7 * num_samples)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    engineer = AttentionFeatureEngineer(context_length=10)
    feature_names = engineer.feature_names

    # Train logistic regression
    print(f"\nTraining logistic regression on {split} samples...")
    det_lr = HallucinationDetector(
        classifier_type="logistic",
        feature_names=feature_names,
    )
    det_lr.fit(X_train, y_train)
    metrics_lr = det_lr.evaluate(X_test, y_test)
    print_metrics(metrics_lr, "Logistic Regression")

    # Train MLP
    print(f"\nTraining MLP on {split} samples...")
    det_mlp = HallucinationDetector(
        classifier_type="mlp",
        hidden_dim=32,
        feature_names=feature_names,
    )
    det_mlp.fit(X_train, y_train)
    metrics_mlp = det_mlp.evaluate(X_test, y_test)
    print_metrics(metrics_mlp, "Two-Layer MLP")

    # Feature importance
    print(f"\n{'═' * 50}")
    print(f"  FEATURE IMPORTANCE (Logistic Regression)")
    print(f"{'═' * 50}")
    importance = det_lr.feature_importance()
    for name, weight in list(importance.items())[:10]:
        bar = "█" * int(weight * 20)
        print(f"  {name:<22} {weight:.4f}  {bar}")

    # Ablation study
    ablation_study(X, y, feature_names)

    # Pass/fail
    print(f"\n{'═' * 50}")
    print(f"  PASS / FAIL")
    print(f"{'═' * 50}")
    best = max(metrics_lr, metrics_mlp, key=lambda m: m.auroc)
    checks = [
        ("AUROC > 0.85", best.auroc > 0.85),
        ("F1 > 0.70", best.f1 > 0.70),
        ("FPR < 10%", best.false_positive_rate < 0.10),
    ]
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}: {passed}")

    return best


def run_real_pipeline(
    samples,
    model_name: str = "EleutherAI/pythia-160m",
    seed: int = 42,
    save_path: Optional[str] = None,
    results_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full pipeline on pre-labeled LabeledSample objects.

    Used by both --halueval and --data modes. Loads the model once,
    extracts features for every non-ambiguous sample, trains both
    classifiers, runs ablation, and optionally saves the detector.

    Parameters
    ----------
    results_path : str, optional
        If given, write a JSON summary of every metric this function prints
        (CV AUROC/CI per detector, held-out metrics, ablation deltas, feature
        importance, dataset stats) to this path — lets a caller (e.g. a
        notebook run on a GPU host) capture structured results instead of
        only a console log.

    Returns
    -------
    Dict[str, Any]
        The same summary dict written to `results_path` (returned regardless
        of whether `results_path` is given).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Filter ambiguous labels
    clean = [s for s in samples if s.label != "ambiguous"]
    print(f"  {len(clean)} non-ambiguous samples (dropped {len(samples) - len(clean)} ambiguous)")

    # Load model once
    print(f"\nLoading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    engineer = AttentionFeatureEngineer(context_length=32)
    entropy_extractor = EntropyFeatureExtractor()

    # Extract features: attention families (flat + per-layer sequence for
    # BiLSTM, from the SAME attentions tensor — one forward pass covers
    # both, instead of extracting attention twice), token-entropy baselines,
    # and black-box top-K features (simulated from the same logits computed
    # for entropy features — no extra forward pass needed either).
    print(f"\nExtracting features for {len(clean)} samples...")
    X_attn_list, X_entropy_list, X_blackbox_list, X_seq_list, y_list = [], [], [], [], []
    failed = 0

    for i, sample in enumerate(clean):
        try:
            prompt, text = build_prompt_and_text(tokenizer, sample.question, sample.model_answer)

            attentions, context_len = extract_attention_from_model(text, model, tokenizer, device, prompt=prompt)
            attn_feats = engineer.extract(attentions, context_len)
            seq_feats = engineer.extract_layer_sequence(attentions)

            logits, token_ids, answer_start = extract_logits_from_model(
                text, model, tokenizer, device, prompt=prompt
            )
            entropy_feats = entropy_extractor.extract(logits, answer_start=answer_start, target_ids=token_ids)
            topk_seq = simulate_topk_from_full_logits(
                logits[answer_start:], token_ids[answer_start:], top_k=5
            )
            blackbox_feats = extract_blackbox_features(topk_seq)

            if not (
                np.all(np.isfinite(attn_feats)) and np.all(np.isfinite(seq_feats))
                and np.all(np.isfinite(entropy_feats)) and np.all(np.isfinite(blackbox_feats))
            ):
                raise ValueError("non-finite feature value (NaN/Inf) — dropping sample")

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
            print(f"  {i + 1}/{len(clean)} processed  (failed: {failed})")

    X_attn = np.array(X_attn_list)
    X_entropy = np.array(X_entropy_list)
    X_blackbox = np.array(X_blackbox_list)
    X_seq_all = np.array(X_seq_list)
    X = np.hstack([X_attn, X_entropy])
    y = np.array(y_list)
    print(f"\nFeature matrix: {X.shape} (attention {X_attn.shape[1]} + entropy {X_entropy.shape[1]})   failed: {failed}")
    print(f"Labels: {int(y.sum())} hallucinated / {int((y == 0).sum())} correct")

    # Train / evaluate — stratified k-fold. X_seq_all is shuffled with the
    # SAME permutation as X/y/X_blackbox (built in the same per-sample loop
    # above, so indices already correspond 1:1 — no separate re-extraction
    # or separate shuffle needed for the BiLSTM sequence path).
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y, X_blackbox, X_seq_all = X[idx], y[idx], X_blackbox[idx], X_seq_all[idx]
    split = int(0.7 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test   = X[split:], y[split:]

    feature_names = engineer.feature_names + entropy_extractor.feature_names
    ablation_families = dict(DEFAULT_ABLATION_FAMILIES)
    ablation_families["entropy_token"] = slice(X_attn.shape[1], X_attn.shape[1] + X_entropy.shape[1])

    print(f"\n{'═' * 50}")
    print(f"  STRATIFIED 5-FOLD CROSS-VALIDATION")
    print(f"{'═' * 50}")

    lr_cv_auroc, lr_ci = stratified_kfold_cv(X, y, k=5, classifier_type="logistic", seed=seed)
    print(f"  LogReg          — AUROC: {lr_cv_auroc:.4f}  95% CI: [{lr_ci[0]:.4f}, {lr_ci[1]:.4f}]")

    mlp_cv_auroc, mlp_ci = stratified_kfold_cv(X, y, k=5, classifier_type="mlp", seed=seed)
    print(f"  MLP             — AUROC: {mlp_cv_auroc:.4f}  95% CI: [{mlp_ci[0]:.4f}, {mlp_ci[1]:.4f}]")

    calib_cv_auroc, calib_ci = stratified_kfold_cv(
        X, y, k=5, seed=seed, detector_factory=lambda: CalibratedEntropyDetector()
    )
    print(f"  CalibratedEntropy — AUROC: {calib_cv_auroc:.4f}  95% CI: [{calib_ci[0]:.4f}, {calib_ci[1]:.4f}]")

    bb_cv_auroc, bb_ci = stratified_kfold_cv(
        X_blackbox, y, k=5, seed=seed, detector_factory=lambda: BlackBoxEntropyDetector()
    )
    print(f"  BlackBoxTopK    — AUROC: {bb_cv_auroc:.4f}  95% CI: [{bb_ci[0]:.4f}, {bb_ci[1]:.4f}]")

    cv_summary = {
        "logistic":           {"auroc": lr_cv_auroc,    "ci": list(lr_ci)},
        "mlp":                {"auroc": mlp_cv_auroc,   "ci": list(mlp_ci)},
        "calibrated_entropy": {"auroc": calib_cv_auroc, "ci": list(calib_ci)},
        "blackbox_topk":      {"auroc": bb_cv_auroc,    "ci": list(bb_ci)},
    }

    # Final held-out evaluation
    print(f"\nTraining on {split} / testing on {len(y) - split}...")

    det_lr = HallucinationDetector(classifier_type="logistic", feature_names=feature_names)
    det_lr.fit(X_train, y_train)
    m_lr = det_lr.evaluate(X_test, y_test)
    print_metrics(m_lr, "Logistic Regression (held-out)")

    det_mlp = HallucinationDetector(classifier_type="mlp", hidden_dim=64, feature_names=feature_names)
    det_mlp.fit(X_train, y_train)
    m_mlp = det_mlp.evaluate(X_test, y_test)
    print_metrics(m_mlp, "MLP (held-out)")

    held_out_summary = {"logistic": asdict(m_lr), "mlp": asdict(m_mlp)}

    # BiLSTM on per-layer sequences (reuses X_seq_all extracted above — no
    # second forward pass, and no separate shuffle to keep in sync with y).
    bilstm_summary = None
    try:
        import torch
        X_seq_train, y_seq_train = X_seq_all[:split], y[:split]
        X_seq_test, y_seq_test = X_seq_all[split:], y[split:]

        bilstm_det = HallucinationDetector(classifier_type="bilstm", hidden_dim=32, epochs=60)
        bilstm_det.fit_sequence(X_seq_train, y_seq_train)
        m_bilstm = bilstm_det.evaluate_sequence(X_seq_test, y_seq_test)
        print_metrics(m_bilstm, "BiLSTM (per-layer sequence, held-out)")

        # Bootstrap CI for BiLSTM
        probs_bilstm = bilstm_det.predict_proba_sequence(X_seq_test)
        bi_lo, bi_hi = bootstrap_auroc_ci(probs_bilstm, y_seq_test)
        print(f"  BiLSTM AUROC 95% CI: [{bi_lo:.4f}, {bi_hi:.4f}]")
        bilstm_summary = {**asdict(m_bilstm), "ci": [bi_lo, bi_hi]}

    except ImportError:
        print("\n  BiLSTM skipped — PyTorch not available (pip install torch)")
        bilstm_det = None
        m_bilstm = None

    # Feature importance
    importance = det_lr.feature_importance()
    print(f"\n{'═' * 50}")
    print(f"  FEATURE IMPORTANCE (Logistic Regression)")
    print(f"{'═' * 50}")
    for name, weight in list(importance.items())[:10]:
        bar = "█" * int(weight * 20)
        print(f"  {name:<28} {weight:.4f}  {bar}")

    ablation_results = ablation_study(X, y, feature_names, families=ablation_families)

    if save_path:
        if m_bilstm and bilstm_det:
            bilstm_det._bilstm.save(save_path)
            print(f"\nBiLSTM detector saved to {save_path}")
        else:
            best = det_mlp if m_mlp.auroc >= m_lr.auroc else det_lr
            best.save(save_path)
            print(f"\nDetector saved to {save_path}")

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "device": device,
        "num_samples_total": len(samples),
        "num_samples_used": len(y),
        "num_failed": failed,
        "num_hallucinated": int(y.sum()),
        "num_correct": int((y == 0).sum()),
        "cv_results": cv_summary,
        "held_out": held_out_summary,
        "bilstm": bilstm_summary,
        "ablation": ablation_results,
        "feature_importance_top10": dict(list(importance.items())[:10]),
    }
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults summary saved to {results_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Pipeline")
    parser.add_argument("--synthetic",  action="store_true", help="Run on synthetic data (no model/API)")
    parser.add_argument("--halueval",   action="store_true", help="Use HaluEval benchmark (no API, needs: pip install datasets)")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--model",      type=str, default="EleutherAI/pythia-160m", help="HuggingFace model")
    parser.add_argument("--data",       type=str, help="Path to labeled JSONL file")
    parser.add_argument("--save",       type=str, help="Save trained detector to this path")
    parser.add_argument("--load",       type=str, help="Load pre-trained detector")
    parser.add_argument("--results",    type=str, help="Write a JSON results summary to this path (--halueval/--data only)")
    parser.add_argument("--abstention", action="store_true",
                         help="Also run the abstention/selective-prediction experiment (see abstention.py)")

    args = parser.parse_args()

    print(f"{'═' * 60}")
    print(f"  HALLUCINATION DETECTION — MULTI-FAMILY FEATURES")
    print(f"  Entropy · Lookback · Frequency · Spectral · Cross-Layer KL")
    print(f"{'═' * 60}\n")

    if args.synthetic:
        best = run_synthetic_demo(args.num_samples, args.seed)
        if args.save:
            X, y = generate_synthetic_dataset(args.num_samples, args.seed)
            det = HallucinationDetector(classifier_type="logistic")
            det.fit(X, y)
            det.save(args.save)
            print(f"\nDetector saved to {args.save}")

        if args.abstention:
            from abstention import run_abstention_experiment, save_risk_coverage_table
            X, y = generate_synthetic_dataset(args.num_samples, args.seed)
            result = run_abstention_experiment(
                X, y, detector_factory=lambda: HallucinationDetector(classifier_type="logistic"),
                seed=args.seed,
            )
            print(f"\n{'═' * 50}")
            print(f"  ABSTENTION / SELECTIVE PREDICTION")
            print(f"{'═' * 50}")
            print(f"  AURC:                          {result['aurc']:.4f}")
            print(f"  Baseline accuracy (100% cov.): {result['baseline_accuracy']:.4f}")
            print(f"  Accuracy @ {result['headline_coverage']:.0%} coverage:      {result['headline_accuracy']:.4f}")
            print(f"  Gain from abstention:          {result['headline_gain']:+.4f}")
            save_risk_coverage_table(result["points"], "results/abstention_risk_coverage.csv")
            print(f"  Risk-coverage table saved to results/abstention_risk_coverage.csv")

    elif args.halueval:
        from data_generator import DataGenerator
        print(f"Mode: HaluEval benchmark  (num_samples={args.num_samples}, no API required)")
        samples = DataGenerator.from_halueval(
            num_samples=args.num_samples,
            seed=args.seed,
        )
        run_real_pipeline(samples, model_name=args.model, seed=args.seed, save_path=args.save, results_path=args.results)
        if args.abstention:
            print("\nFor an abstention/selective-prediction experiment on HaluEval data, run:")
            print(f"  python abstention.py --halueval --num_samples {args.num_samples} --model {args.model}")

    elif args.data:
        from data_generator import DataGenerator
        print(f"Mode: loading data from {args.data}")
        samples = DataGenerator.load(args.data)
        print(f"  Loaded {len(samples)} samples")
        run_real_pipeline(samples, model_name=args.model, seed=args.seed, save_path=args.save, results_path=args.results)

    else:
        print("Choose a data source:")
        print()
        print("  --synthetic          Fast demo, no model or API needed")
        print("    python pipeline.py --synthetic --num_samples 1000")
        print()
        print("  --halueval           Real benchmark data, no API needed (pip install datasets)")
        print("    python pipeline.py --halueval --num_samples 500 --model EleutherAI/pythia-160m")
        print()
        print("  --data <file.jsonl>  Your own labeled dataset")
        print("    python pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m")


if __name__ == "__main__":
    main()
