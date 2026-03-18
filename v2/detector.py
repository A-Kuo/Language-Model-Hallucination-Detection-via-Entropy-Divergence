"""
Hallucination Detector — Lightweight Classifier
=================================================

Trains a binary classifier on multi-family attention features.

Following Lookback Lens (Chuang et al., EMNLP 2024):
    "We find that a linear classifier based on these lookback ratio
    features is as effective as a richer detector that utilizes the
    entire hidden states of an LLM."

We implement:
    1. Logistic Regression (default, most interpretable)
    2. Two-layer MLP (optional, slightly more expressive)
    3. Feature importance / ablation analysis

Usage:
    detector = HallucinationDetector(classifier_type="logistic")
    detector.fit(X_train, y_train)
    predictions = detector.predict(X_test)
    metrics = detector.evaluate(X_test, y_test)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid


# =========================================================================
# Logistic Regression (pure numpy — no sklearn dependency)
# =========================================================================

class LogisticRegression:
    """
    L2-regularised logistic regression via gradient descent.

    This is the same classifier type used by Lookback Lens.
    Pure numpy implementation — no sklearn needed.

    Loss: L(w,b) = -1/N Σ [y·log(σ(z)) + (1-y)·log(1-σ(z))] + λ||w||²

    where z = Xw + b, σ is sigmoid.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        l2_lambda: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.lr = learning_rate
        self.l2 = l2_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Train on feature matrix X (N, D) and binary labels y (N,)."""
        N, D = X.shape
        self.weights = np.zeros(D)
        self.bias = 0.0
        self.loss_history = []

        for step in range(self.max_iter):
            z = X @ self.weights + self.bias
            p = sigmoid(z)

            # Gradients
            error = p - y  # (N,)
            grad_w = (X.T @ error) / N + 2 * self.l2 * self.weights
            grad_b = error.mean()

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            # Loss
            eps = 1e-12
            loss = -np.mean(
                y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            ) + self.l2 * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            if step > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(hallucination) for each sample."""
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


# =========================================================================
# Two-layer MLP
# =========================================================================

class SimpleMLP:
    """
    Two-layer MLP: input → hidden (ReLU) → output (sigmoid).

    Slightly more expressive than logistic regression, captures
    nonlinear interactions between feature families.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        max_iter: int = 2000,
        l2_lambda: float = 0.001,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.max_iter = max_iter
        self.l2 = l2_lambda
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleMLP:
        N, D = X.shape
        rng = np.random.default_rng(42)

        # Xavier initialization
        self.W1 = rng.normal(0, np.sqrt(2 / D), (D, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2 / self.hidden_dim), (self.hidden_dim,))
        self.b2 = 0.0

        for step in range(self.max_iter):
            # Forward
            h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
            z = h @ self.W2 + self.b2
            p = sigmoid(z)

            # Loss
            eps = 1e-12
            loss = -np.mean(
                y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            ) + self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            self.loss_history.append(loss)

            # Backward
            error = p - y  # (N,)
            grad_W2 = (h.T @ error) / N + 2 * self.l2 * self.W2
            grad_b2 = error.mean()

            dh = np.outer(error, self.W2)  # (N, hidden)
            dh[h <= 0] = 0  # ReLU gradient
            grad_W1 = (X.T @ dh) / N + 2 * self.l2 * self.W1
            grad_b1 = dh.mean(axis=0)

            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# =========================================================================
# Unified Detector
# =========================================================================

@dataclass
class DetectorMetrics:
    """Evaluation metrics for the hallucination detector."""
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    num_samples: int
    threshold: float = 0.5


class HallucinationDetector:
    """
    Hallucination detector: trains on multi-family attention features.

    Supports logistic regression (like Lookback Lens) or MLP.
    """

    def __init__(
        self,
        classifier_type: Literal["logistic", "mlp"] = "logistic",
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.classifier_type = classifier_type
        self.feature_names = feature_names

        if classifier_type == "logistic":
            self.model = LogisticRegression(**kwargs)
        elif classifier_type == "mlp":
            self.model = SimpleMLP(**kwargs)
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> HallucinationDetector:
        """
        Train the detector.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix (output of AttentionFeatureEngineer.extract_vector).
        y : np.ndarray, shape (N,)
            Binary labels: 1 = hallucinated, 0 = correct.
        """
        # Standardise features
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        self.model.fit(X_norm, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(hallucination) for each sample."""
        assert self._fitted, "Must call fit() first"
        X_norm = (X - self._mean) / self._std
        return self.model.predict_proba(X_norm)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> DetectorMetrics:
        """Compute all evaluation metrics."""
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        fpr = fp / max(fp + tn, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

        # AUROC
        auroc = self._compute_auroc(probs, y)

        return DetectorMetrics(
            auroc=auroc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            false_positive_rate=fpr,
            num_samples=len(y),
            threshold=threshold,
        )

    def feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance (absolute weight) for logistic regression.

        Maps feature name → |weight|, sorted descending.
        """
        if self.classifier_type != "logistic":
            return {}

        weights = np.abs(self.model.weights)
        names = self.feature_names or [f"f{i}" for i in range(len(weights))]

        importance = dict(zip(names, weights))
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    @staticmethod
    def _compute_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUROC via trapezoidal integration."""
        sorted_idx = np.argsort(-probs)
        sorted_labels = labels[sorted_idx]

        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_list = [0.0]
        fpr_list = [0.0]
        tp = fp = 0

        for lab in sorted_labels:
            if lab == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(tpr_list, fpr_list))

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save detector to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "classifier_type": self.classifier_type,
                "model": self.model,
                "feature_names": self.feature_names,
                "mean": self._mean,
                "std": self._std,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: str) -> HallucinationDetector:
        """Load detector from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        det = cls(
            classifier_type=state["classifier_type"],
            feature_names=state["feature_names"],
        )
        det.model = state["model"]
        det._mean = state["mean"]
        det._std = state["std"]
        det._fitted = state["fitted"]
        return det


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DETECTOR — STANDALONE VALIDATION")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Generate synthetic data: hallucinations have higher feature values
    N = 400
    D = 18  # matches our feature engineer output
    y = rng.integers(0, 2, N).astype(float)

    # Features: hallucinated samples shifted right
    X = rng.standard_normal((N, D))
    X[y == 1] += 1.0  # hallucinated = higher

    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    # --- Test 1: Logistic regression ---
    print("\n--- Test 1: Logistic Regression ---")
    det_lr = HallucinationDetector(
        classifier_type="logistic",
        feature_names=[f"feat_{i}" for i in range(D)],
    )
    det_lr.fit(X_train, y_train)
    metrics_lr = det_lr.evaluate(X_test, y_test)
    print(f"  AUROC: {metrics_lr.auroc:.4f}")
    print(f"  F1:    {metrics_lr.f1:.4f}")
    print(f"  FPR:   {metrics_lr.false_positive_rate:.4f}")
    assert metrics_lr.auroc > 0.7, "AUROC too low on separable data"
    print("  Logistic regression works ✅")

    # --- Test 2: MLP ---
    print("\n--- Test 2: MLP ---")
    det_mlp = HallucinationDetector(classifier_type="mlp", hidden_dim=16)
    det_mlp.fit(X_train, y_train)
    metrics_mlp = det_mlp.evaluate(X_test, y_test)
    print(f"  AUROC: {metrics_mlp.auroc:.4f}")
    print(f"  F1:    {metrics_mlp.f1:.4f}")
    assert metrics_mlp.auroc > 0.6
    print("  MLP works ✅")

    # --- Test 3: Feature importance ---
    print("\n--- Test 3: Feature importance ---")
    importance = det_lr.feature_importance()
    top3 = list(importance.items())[:3]
    print(f"  Top features: {top3}")
    assert len(importance) == D
    print("  Feature importance works ✅")

    # --- Test 4: Save/load ---
    print("\n--- Test 4: Persistence ---")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmppath = f.name

    det_lr.save(tmppath)
    det_loaded = HallucinationDetector.load(tmppath)
    probs_orig = det_lr.predict_proba(X_test)
    probs_loaded = det_loaded.predict_proba(X_test)
    assert np.allclose(probs_orig, probs_loaded)
    print("  Save/load roundtrip ✅")

    Path(tmppath).unlink()

    print(f"\n{'=' * 60}")
    print(f"Detector — ALL 4 CHECKS PASS ✅")
    print(f"{'=' * 60}")
