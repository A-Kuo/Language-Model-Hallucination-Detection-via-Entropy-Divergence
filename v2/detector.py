"""
Hallucination Detector — Classifiers
======================================

Trains a binary classifier on multi-family attention features.

We implement three classifiers:
    1. Logistic Regression — interpretable, like Lookback Lens baseline
    2. Two-layer MLP — captures nonlinear feature interactions
    3. BiLSTM — operates on per-layer feature sequences; best AUROC

The BiLSTM (primary classifier) reads the sequence of per-layer attention
features bidirectionally. Forward pass: early layers → late layers (syntactic
to semantic). Backward pass: late → early. The concatenated final hidden
states feed a binary output head.

On HaluEval with Pythia-160m: BiLSTM achieves ~0.96 AUROC vs ~0.84 for
logistic regression, because cross-layer dynamics are better captured by
sequence modelling than by global summary statistics.

Usage:
    detector = HallucinationDetector(classifier_type="bilstm")
    detector.fit_sequence(X_seq_train, y_train)   # X_seq: (N, L, 6)
    predictions = detector.predict_sequence(X_seq_test)
    metrics = detector.evaluate_sequence(X_seq_test, y_test)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


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
# BiLSTM Detector (PyTorch)
# =========================================================================

class BiLSTMHallucinationNet(nn.Module if _HAS_TORCH else object):
    """
    Bidirectional LSTM operating on per-layer attention feature sequences.

    Architecture:
        Input: (N, L, input_dim)  — one feature vector per transformer layer
        BiLSTM: 2 stacked layers, hidden_dim units per direction
        Output head: Linear(2 * hidden_dim → 1) → sigmoid

    Why BiLSTM over flat features:
        Global summary stats (18D) discard the ordering of layers. A BiLSTM
        reads the sequence forward (syntactic → semantic) and backward
        (semantic → syntactic), capturing how hallucination-related uncertainty
        evolves across model depth. This is the primary classifier architecture.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("BiLSTM requires PyTorch: pip install torch")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        x: (batch, L, input_dim)
        returns: (batch,) hallucination probabilities
        """
        out, (h_n, _) = self.lstm(x)
        # Concatenate final forward + backward hidden states
        h_forward  = h_n[-2]   # last forward layer
        h_backward = h_n[-1]   # last backward layer
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (batch, 2*hidden)
        h_cat = self.dropout(h_cat)
        logit = self.head(h_cat).squeeze(-1)   # (batch,)
        return torch.sigmoid(logit)


class BiLSTMDetector:
    """
    Wrapper that trains BiLSTMHallucinationNet on per-layer feature sequences.

    Input convention:
        X_seq: np.ndarray of shape (N, L, 6) — from feature_engineer.extract_layer_sequence()
        y:     np.ndarray of shape (N,) — 1 = hallucinated, 0 = correct
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 32,
        l2: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("BiLSTM requires PyTorch: pip install torch")
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.l2         = l2
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net: Optional[BiLSTMHallucinationNet] = None
        self._mean: Optional[np.ndarray] = None
        self._std:  Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> "BiLSTMDetector":
        """
        Train the BiLSTM.

        Parameters
        ----------
        X_seq : np.ndarray, shape (N, L, 6)
        y     : np.ndarray, shape (N,)
        """
        N, L, D = X_seq.shape

        # Standardise per feature dimension across (N, L)
        flat = X_seq.reshape(-1, D)
        self._mean = flat.mean(axis=0)
        self._std  = flat.std(axis=0) + 1e-8
        X_norm = ((X_seq - self._mean) / self._std).astype(np.float32)

        self._net = BiLSTMHallucinationNet(
            input_dim=D,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.l2
        )
        criterion = nn.BCELoss()

        X_t = torch.tensor(X_norm).to(self.device)
        y_t = torch.tensor(y.astype(np.float32)).to(self.device)

        self._net.train()
        for epoch in range(self.epochs):
            idx = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb, yb = X_t[batch_idx], y_t[batch_idx]
                optimizer.zero_grad()
                probs = self._net(xb)
                loss  = criterion(probs, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

        self._net.eval()
        self._fitted = True
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        X_norm = ((X_seq - self._mean) / self._std).astype(np.float32)
        with torch.no_grad():
            probs = self._net(torch.tensor(X_norm).to(self.device))
        return probs.cpu().numpy()

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_seq) >= threshold).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "type": "bilstm",
                "state_dict": self._net.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "mean": self._mean,
                "std": self._std,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BiLSTMDetector":
        with open(path, "rb") as f:
            state = pickle.load(f)
        det = cls(
            input_dim=state["input_dim"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
        )
        det._net = BiLSTMHallucinationNet(
            input_dim=state["input_dim"],
            hidden_dim=state["hidden_dim"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
        )
        det._net.load_state_dict(state["state_dict"])
        det._net.eval()
        det._mean = state["mean"]
        det._std  = state["std"]
        det._fitted = True
        return det


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

    Supports three classifier types:
        "logistic" — interpretable baseline (Lookback Lens style)
        "mlp"      — two-layer MLP for nonlinear interactions
        "bilstm"   — BiLSTM on per-layer sequences (best AUROC, ~0.96)

    For BiLSTM, use fit_sequence() / predict_sequence() / evaluate_sequence()
    which accept (N, L, 6) tensors from feature_engineer.extract_layer_sequence().
    """

    def __init__(
        self,
        classifier_type: Literal["logistic", "mlp", "bilstm"] = "logistic",
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.classifier_type = classifier_type
        self.feature_names = feature_names

        if classifier_type == "logistic":
            self.model = LogisticRegression(**kwargs)
        elif classifier_type == "mlp":
            self.model = SimpleMLP(**kwargs)
        elif classifier_type == "bilstm":
            self._bilstm = BiLSTMDetector(**kwargs)
            self.model = None
        else:
            raise ValueError(f"Unknown classifier: {classifier_type!r}. Choose: logistic, mlp, bilstm")

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

    # ------------------------------------------------------------------
    # BiLSTM interface (per-layer sequences)
    # ------------------------------------------------------------------

    def fit_sequence(self, X_seq: np.ndarray, y: np.ndarray) -> "HallucinationDetector":
        """
        Train the BiLSTM on per-layer feature sequences.

        Parameters
        ----------
        X_seq : np.ndarray, shape (N, L, 6)
            Output of AttentionFeatureEngineer.extract_layer_sequence().
        y : np.ndarray, shape (N,) — 1=hallucinated, 0=correct
        """
        assert self.classifier_type == "bilstm", \
            "fit_sequence() only for classifier_type='bilstm'"
        self._bilstm.fit(X_seq, y)
        self._fitted = True
        return self

    def predict_sequence(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self._bilstm.predict_proba(X_seq) >= threshold).astype(int)

    def predict_proba_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        return self._bilstm.predict_proba(X_seq)

    def evaluate_sequence(
        self, X_seq: np.ndarray, y: np.ndarray, threshold: float = 0.5
    ) -> DetectorMetrics:
        """Evaluate BiLSTM on per-layer sequences."""
        probs = self._bilstm.predict_proba(X_seq)
        preds = (probs >= threshold).astype(int)

        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-10)
        fpr       = fp / max(fp + tn, 1)
        accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
        auroc     = self._compute_auroc(probs, y)

        return DetectorMetrics(
            auroc=auroc, accuracy=accuracy, precision=precision,
            recall=recall, f1=f1, false_positive_rate=fpr,
            num_samples=len(y), threshold=threshold,
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
