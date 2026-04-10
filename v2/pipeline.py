"""
End-to-End Pipeline — Hallucination Detection v2
===================================================

Orchestrates the full workflow:
    1. Load labeled samples (from DataGenerator or file)
    2. Extract attention tensors from a local model
    3. Compute multi-family features
    4. Train/evaluate the lightweight detector

Usage:
    # Synthetic demo (no model or API needed)
    python v2/pipeline.py --synthetic --num_samples 1000

    # HaluEval benchmark (no API — pip install datasets)
    python v2/pipeline.py --halueval --num_samples 500 --model EleutherAI/pythia-160m

    # Full pipeline with self-generated data (requires ANTHROPIC_API_KEY)
    python v2/pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m

    # Save trained detector
    python v2/pipeline.py --halueval --num_samples 500 --save detector.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.feature_engineer import AttentionFeatureEngineer, FeatureConfig
from v2.detector import HallucinationDetector, BiLSTMDetector, DetectorMetrics


# =========================================================================
# Attention extraction (model-agnostic)
# =========================================================================

def extract_attention_from_model(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
) -> Tuple[np.ndarray, int]:
    """
    Run a forward pass and extract attention tensors.

    Returns
    -------
    attentions : np.ndarray, shape (L, H, T, T)
    context_length : int (based on input prompt length)
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(device)
    context_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    # Stack attention layers → (L, H, T, T)
    attn_list = []
    for layer_attn in outputs.attentions:
        attn_list.append(layer_attn[0].detach().cpu().numpy())

    return np.stack(attn_list), context_length


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
) -> Tuple[float, Tuple[float, float]]:
    """
    Stratified k-fold cross-validation with bootstrap AUROC CI.

    Maintains class balance across folds (important for imbalanced data).
    Returns mean AUROC and 95% CI from bootstrap on held-out predictions.
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

        det = HallucinationDetector(classifier_type=classifier_type)
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


def ablation_study(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """
    Feature family ablation: train with each family removed to measure
    its contribution.
    """
    families = {
        "entropy": slice(0, 3),
        "lookback": slice(3, 7),
        "frequency": slice(7, 11),
        "spectral": slice(11, 15),
        "cross_layer_kl": slice(15, 18),
    }

    N = len(y)
    split = int(0.7 * N)

    # Baseline: all features
    det_full = HallucinationDetector(classifier_type="logistic")
    det_full.fit(X[:split], y[:split])
    full_auroc = det_full.evaluate(X[split:], y[split:]).auroc

    print(f"\n{'═' * 50}")
    print(f"  FEATURE FAMILY ABLATION")
    print(f"{'═' * 50}")
    print(f"  Full model (18 features): AUROC = {full_auroc:.4f}")
    print(f"  {'─' * 46}")

    for family, s in families.items():
        # Remove this family
        mask = np.ones(X.shape[1], dtype=bool)
        mask[s] = False
        X_ablated = X[:, mask]

        det = HallucinationDetector(classifier_type="logistic")
        det.fit(X_ablated[:split], y[:split])
        ablated_auroc = det.evaluate(X_ablated[split:], y[split:]).auroc

        delta = full_auroc - ablated_auroc
        direction = "↓" if delta > 0.001 else "→"
        print(f"  Without {family:<16}: AUROC = {ablated_auroc:.4f}  ({direction} {delta:+.4f})")

    print(f"{'═' * 50}")


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
) -> None:
    """
    Run the full pipeline on pre-labeled LabeledSample objects.

    Used by both --halueval and --data modes. Loads the model once,
    extracts features for every non-ambiguous sample, trains both
    classifiers, runs ablation, and optionally saves the detector.
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

    # Extract features
    print(f"\nExtracting features for {len(clean)} samples...")
    X_list, y_list = [], []
    failed = 0

    for i, sample in enumerate(clean):
        try:
            text = f"Question: {sample.question}\nAnswer: {sample.model_answer}"
            attentions, context_len = extract_attention_from_model(text, model, tokenizer, device)
            feats = engineer.extract(attentions, context_len)
            X_list.append(feats)
            y_list.append(1.0 if sample.label == "hallucinated" else 0.0)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Warning: sample {i} failed — {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clean)} processed  (failed: {failed})")

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\nFeature matrix: {X.shape}   failed: {failed}")
    print(f"Labels: {int(y.sum())} hallucinated / {int((y == 0).sum())} correct")

    # Train / evaluate — stratified k-fold
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test   = X[split:], y[split:]

    feature_names = engineer.feature_names

    print(f"\n{'═' * 50}")
    print(f"  STRATIFIED 5-FOLD CROSS-VALIDATION")
    print(f"{'═' * 50}")

    lr_cv_auroc, lr_ci = stratified_kfold_cv(X, y, k=5, classifier_type="logistic", seed=seed)
    print(f"  LogReg — AUROC: {lr_cv_auroc:.4f}  95% CI: [{lr_ci[0]:.4f}, {lr_ci[1]:.4f}]")

    mlp_cv_auroc, mlp_ci = stratified_kfold_cv(X, y, k=5, classifier_type="mlp", seed=seed)
    print(f"  MLP    — AUROC: {mlp_cv_auroc:.4f}  95% CI: [{mlp_ci[0]:.4f}, {mlp_ci[1]:.4f}]")

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

    # BiLSTM on per-layer sequences
    try:
        import torch
        print(f"\nExtracting per-layer sequences for BiLSTM...")
        seq_list = []
        for sample in clean:
            try:
                text = f"Question: {sample.question}\nAnswer: {sample.model_answer}"
                attentions, ctx_len = extract_attention_from_model(text, model, tokenizer, device)
                seq = engineer.extract_layer_sequence(attentions)
                seq_list.append(seq)
            except Exception:
                seq_list.append(None)

        valid_mask = [s is not None for s in seq_list]
        X_seq = np.array([s for s in seq_list if s is not None])
        y_seq = y[[i for i, v in enumerate(valid_mask) if v]]

        idx_s = rng.permutation(len(y_seq))
        X_seq, y_seq = X_seq[idx_s], y_seq[idx_s]
        sp = int(0.7 * len(y_seq))

        bilstm_det = HallucinationDetector(classifier_type="bilstm", hidden_dim=32, epochs=60)
        bilstm_det.fit_sequence(X_seq[:sp], y_seq[:sp])
        m_bilstm = bilstm_det.evaluate_sequence(X_seq[sp:], y_seq[sp:])
        print_metrics(m_bilstm, "BiLSTM (per-layer sequence, held-out)")

        # Bootstrap CI for BiLSTM
        probs_bilstm = bilstm_det.predict_proba_sequence(X_seq[sp:])
        bi_lo, bi_hi = bootstrap_auroc_ci(probs_bilstm, y_seq[sp:])
        print(f"  BiLSTM AUROC 95% CI: [{bi_lo:.4f}, {bi_hi:.4f}]")

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

    ablation_study(X, y, feature_names)

    if save_path:
        if m_bilstm and bilstm_det:
            bilstm_det._bilstm.save(save_path)
            print(f"\nBiLSTM detector saved to {save_path}")
        else:
            best = det_mlp if m_mlp.auroc >= m_lr.auroc else det_lr
            best.save(save_path)
            print(f"\nDetector saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Pipeline v2")
    parser.add_argument("--synthetic",  action="store_true", help="Run on synthetic data (no model/API)")
    parser.add_argument("--halueval",   action="store_true", help="Use HaluEval benchmark (no API, needs: pip install datasets)")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--model",      type=str, default="EleutherAI/pythia-160m", help="HuggingFace model")
    parser.add_argument("--data",       type=str, help="Path to labeled JSONL file")
    parser.add_argument("--save",       type=str, help="Save trained detector to this path")
    parser.add_argument("--load",       type=str, help="Load pre-trained detector")

    args = parser.parse_args()

    print(f"{'═' * 60}")
    print(f"  HALLUCINATION DETECTION v2 — MULTI-FAMILY FEATURES")
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

    elif args.halueval:
        from v2.data_generator import DataGenerator
        print(f"Mode: HaluEval benchmark  (num_samples={args.num_samples}, no API required)")
        samples = DataGenerator.from_halueval(
            num_samples=args.num_samples,
            seed=args.seed,
        )
        run_real_pipeline(samples, model_name=args.model, seed=args.seed, save_path=args.save)

    elif args.data:
        from v2.data_generator import DataGenerator
        print(f"Mode: loading data from {args.data}")
        samples = DataGenerator.load(args.data)
        print(f"  Loaded {len(samples)} samples")
        run_real_pipeline(samples, model_name=args.model, seed=args.seed, save_path=args.save)

    else:
        print("Choose a data source:")
        print()
        print("  --synthetic          Fast demo, no model or API needed")
        print("    python v2/pipeline.py --synthetic --num_samples 1000")
        print()
        print("  --halueval           Real benchmark data, no API needed (pip install datasets)")
        print("    python v2/pipeline.py --halueval --num_samples 500 --model EleutherAI/pythia-160m")
        print()
        print("  --data <file.jsonl>  Your own labeled dataset")
        print("    python v2/pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m")


if __name__ == "__main__":
    main()
