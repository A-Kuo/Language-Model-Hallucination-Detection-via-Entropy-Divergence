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
    python v2/pipeline.py --synthetic --num_samples 500

    # Full pipeline (requires model + API key)
    python v2/pipeline.py --model EleutherAI/pythia-160m --data data/train.jsonl

    # Evaluate pre-trained detector
    python v2/pipeline.py --load detector.pkl --data data/test.jsonl
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
from v2.detector import HallucinationDetector, DetectorMetrics


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


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Pipeline v2")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic data")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m", help="HuggingFace model")
    parser.add_argument("--data", type=str, help="Path to labeled JSONL data")
    parser.add_argument("--save", type=str, help="Save detector to this path")
    parser.add_argument("--load", type=str, help="Load pre-trained detector")

    args = parser.parse_args()

    print(f"{'═' * 60}")
    print(f"  HALLUCINATION DETECTION v2 — MULTI-FAMILY FEATURES")
    print(f"  Entropy · Lookback · Frequency · Spectral · Cross-Layer KL")
    print(f"{'═' * 60}\n")

    if args.synthetic:
        best = run_synthetic_demo(args.num_samples, args.seed)
        if args.save:
            # Re-train on full data and save
            X, y = generate_synthetic_dataset(args.num_samples, args.seed)
            det = HallucinationDetector(classifier_type="logistic")
            det.fit(X, y)
            det.save(args.save)
            print(f"\nDetector saved to {args.save}")
    else:
        print("Full pipeline requires --data (labeled JSONL) and --model.")
        print("Generate data first:")
        print("  python -c \"from v2.data_generator import DataGenerator; ...")
        print("  gen = DataGenerator(); dataset = gen.generate(100); gen.save(dataset, 'data/train.jsonl')\"")
        print("\nOr use synthetic mode:")
        print("  python v2/pipeline.py --synthetic --num_samples 1000")


if __name__ == "__main__":
    main()
