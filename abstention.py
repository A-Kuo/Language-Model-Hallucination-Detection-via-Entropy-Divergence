"""
Abstention / Selective-Prediction Experiment
===============================================

Demonstrates the practical payoff of calibrated uncertainty: if the system
abstains (refuses to answer) on the examples a detector flags as
highest-risk, accuracy on the examples it does answer should improve as
coverage (fraction answered) shrinks. This is the standard risk-coverage /
selective-prediction evaluation, computed here on out-of-fold predictions
from the same stratified k-fold splitting pipeline.py already uses (to
avoid the train/test leakage that a plain train/predict-on-train curve
would have).

Usage:
    python abstention.py --synthetic --num_samples 1000
    python abstention.py --halueval --num_samples 500 --detector logistic
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np

from detector import HallucinationDetector, compute_auroc
from pipeline import generate_synthetic_dataset


@dataclass
class RiskCoveragePoint:
    threshold: float
    coverage: float       # fraction of examples answered (not abstained)
    accuracy: float        # accuracy among answered examples only ("nan" if none answered)
    n_answered: int
    n_abstained: int


def _evaluate_threshold(probs: np.ndarray, y_true: np.ndarray, tau: float) -> RiskCoveragePoint:
    """
    Answer examples with probs <= tau (low predicted hallucination risk);
    abstain on the rest. "Correct to answer" means the example was actually
    not hallucinated (y_true == 0). If nothing is answered, accuracy is
    reported as NaN by convention (nothing to be right or wrong about) —
    callers must not silently treat NaN as 0 or 1.
    """
    n = len(y_true)
    answered_mask = probs <= tau
    n_answered = int(answered_mask.sum())
    n_abstained = n - n_answered
    coverage = n_answered / n if n > 0 else float("nan")

    if n_answered == 0:
        accuracy = float("nan")
    else:
        accuracy = float((y_true[answered_mask] == 0).mean())

    return RiskCoveragePoint(
        threshold=float(tau),
        coverage=coverage,
        accuracy=accuracy,
        n_answered=n_answered,
        n_abstained=n_abstained,
    )


def risk_coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_thresholds: int = 50,
) -> List[RiskCoveragePoint]:
    """
    Sweep an abstention threshold tau over [min(probs), max(probs)]. At each
    tau, answer examples with probs <= tau (low predicted hallucination
    risk); abstain on the rest.

    At tau == max(probs), every example is answered (coverage == 1.0,
    exactly the plain no-abstention accuracy). At tau == min(probs),
    coverage is the fraction of examples tied at the global minimum score
    (usually a small but nonzero fraction) — true zero coverage requires a
    threshold strictly below every score; see _evaluate_threshold, which
    this function's sweep is built on and which handles that case directly.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    thresholds = np.linspace(probs.min(), probs.max(), n_thresholds)
    return [_evaluate_threshold(probs, y_true, tau) for tau in thresholds]


def area_under_risk_coverage(points: List[RiskCoveragePoint]) -> float:
    """
    AURC: trapezoidal integral of accuracy over coverage, restricted to
    points with defined (non-NaN) accuracy, sorted by coverage ascending —
    mirrors the trapezoidal-integration style already used for AUROC in
    detector.py's compute_auroc.
    """
    valid = [p for p in points if not np.isnan(p.accuracy)]
    if len(valid) < 2:
        return float("nan")
    valid = sorted(valid, key=lambda p: p.coverage)
    coverages = np.array([p.coverage for p in valid])
    accuracies = np.array([p.accuracy for p in valid])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(accuracies, coverages))


def _stratified_out_of_fold_probs(
    X: np.ndarray, y: np.ndarray, detector_factory: Callable[[], Any], k: int = 5, seed: int = 42
) -> np.ndarray:
    """Pooled out-of-fold probabilities via stratified k-fold, matching
    pipeline.py::stratified_kfold_cv's splitting logic (duplicated here in
    reduced form since that function returns only the AUROC summary, not the
    raw pooled probabilities the risk-coverage curve needs)."""
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    all_probs = np.zeros(len(y))

    for fold in range(k):
        val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        train_idx = np.concatenate([
            np.concatenate([pos_folds[j] for j in range(k) if j != fold]),
            np.concatenate([neg_folds[j] for j in range(k) if j != fold]),
        ])

        det = detector_factory()
        det.fit(X[train_idx], y[train_idx])
        all_probs[val_idx] = det.predict_proba(X[val_idx])

    return all_probs


def run_abstention_experiment(
    X: np.ndarray,
    y: np.ndarray,
    detector_factory: Callable[[], Any],
    k: int = 5,
    seed: int = 42,
    n_thresholds: int = 50,
    coverage_headline: float = 0.8,
) -> dict:
    """
    Run the full abstention experiment: pooled out-of-fold probabilities via
    stratified k-fold, then the risk-coverage curve and its summary metrics.

    Returns
    -------
    dict with keys: "points" (List[RiskCoveragePoint]), "aurc" (float),
    "baseline_accuracy" (accuracy at 100% coverage), "headline_accuracy"
    (accuracy at the closest achieved coverage to `coverage_headline`),
    "headline_coverage" (that actual coverage value), "headline_gain"
    (headline_accuracy - baseline_accuracy).
    """
    probs = _stratified_out_of_fold_probs(X, y, detector_factory, k=k, seed=seed)
    points = risk_coverage_curve(probs, y, n_thresholds=n_thresholds)
    aurc = area_under_risk_coverage(points)

    full_coverage_point = max(points, key=lambda p: p.coverage)
    baseline_accuracy = full_coverage_point.accuracy

    valid = [p for p in points if not np.isnan(p.accuracy)]
    headline_point = min(valid, key=lambda p: abs(p.coverage - coverage_headline))

    return {
        "points": points,
        "aurc": aurc,
        "baseline_accuracy": baseline_accuracy,
        "headline_accuracy": headline_point.accuracy,
        "headline_coverage": headline_point.coverage,
        "headline_gain": headline_point.accuracy - baseline_accuracy,
    }


def save_risk_coverage_table(points: List[RiskCoveragePoint], path: str) -> None:
    """Write the risk-coverage curve to a CSV, consistent with how
    COLABS.md/paper.tex document results as tables rather than plots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "coverage", "accuracy", "n_answered", "n_abstained"])
        for p in points:
            writer.writerow([p.threshold, p.coverage, p.accuracy, p.n_answered, p.n_abstained])


def _detector_factory_for(name: str) -> Callable[[], Any]:
    if name in ("logistic", "mlp"):
        return lambda: HallucinationDetector(classifier_type=name)
    if name == "calibrated":
        from calibrated_entropy_detector import CalibratedEntropyDetector
        return lambda: CalibratedEntropyDetector()
    raise ValueError(f"Unknown detector: {name!r}. Choose: logistic, mlp, calibrated")


def main():
    parser = argparse.ArgumentParser(description="Abstention / selective-prediction experiment")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic feature data")
    parser.add_argument("--halueval", action="store_true", help="Use HaluEval benchmark data")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--detector", type=str, default="logistic",
                         choices=["logistic", "mlp", "calibrated"])
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--out", type=str, default=None, help="CSV output path for the risk-coverage table")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.halueval:
        from data_generator import DataGenerator
        from pipeline import extract_attention_from_model
        from feature_engineer import AttentionFeatureEngineer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        samples = DataGenerator.from_halueval(num_samples=args.num_samples)
        clean = [s for s in samples if s.label != "ambiguous"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, output_attentions=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        engineer = AttentionFeatureEngineer(context_length=32)

        X_list, y_list = [], []
        for sample in clean:
            try:
                prompt = f"Question: {sample.question}\nAnswer:"
                text = f"{prompt} {sample.model_answer}"
                attentions, ctx_len = extract_attention_from_model(text, model, tokenizer, device, prompt=prompt)
                feats = engineer.extract(attentions, ctx_len)
                if not np.all(np.isfinite(feats)):
                    continue
                X_list.append(feats)
                y_list.append(1.0 if sample.label == "hallucinated" else 0.0)
            except Exception:
                continue
        X, y = np.array(X_list), np.array(y_list)
    else:
        X, y = generate_synthetic_dataset(num_samples=args.num_samples, seed=args.seed)

    print(f"Running abstention experiment: detector={args.detector}, N={len(y)}")
    result = run_abstention_experiment(X, y, _detector_factory_for(args.detector), seed=args.seed)

    print(f"\n{'=' * 55}")
    print("  ABSTENTION / SELECTIVE PREDICTION")
    print(f"{'=' * 55}")
    print(f"  AURC:                          {result['aurc']:.4f}")
    print(f"  Baseline accuracy (100% cov.): {result['baseline_accuracy']:.4f}")
    print(f"  Accuracy @ {result['headline_coverage']:.0%} coverage:      {result['headline_accuracy']:.4f}")
    print(f"  Gain from abstention:          {result['headline_gain']:+.4f}")
    print(f"{'=' * 55}")

    out_path = args.out or "results/abstention_risk_coverage.csv"
    save_risk_coverage_table(result["points"], out_path)
    print(f"\nRisk-coverage table saved to {out_path}")


if __name__ == "__main__":
    main()
