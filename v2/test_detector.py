"""
Tests for v2 Hallucination Detector
======================================

Strategy:
    Pure numpy + scipy — no torch, no model loading, no external data.
    Mirrors the structure of v1/test_attention_analyzer.py.

    - Section 1:  LogisticRegression
    - Section 2:  SimpleMLP
    - Section 3:  HallucinationDetector (unified API)
    - Section 4:  AUROC edge cases

Run:
    pytest v2/test_detector.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import (
    DetectorMetrics,
    HallucinationDetector,
    LogisticRegression,
    SimpleMLP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separable(n=300, d=10, shift=3.0, seed=42):
    """
    Linearly separable dataset: class-1 features shifted by `shift`.
    Returns (X_train, y_train, X_test, y_test).
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2).astype(float)
    X = rng.standard_normal((n, d))
    X[y == 1] += shift

    # Shuffle before splitting so both classes appear in train and test
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    split = int(0.75 * n)
    return X[:split], y[:split], X[split:], y[split:]


def _make_random(n=200, d=8, seed=7):
    """Random labels (chance-level dataset)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, n).astype(float)
    split = n // 2
    return X[:split], y[:split], X[split:], y[split:]


# =========================================================================
# Section 1: LogisticRegression
# =========================================================================

class TestLogisticRegression:
    """Tests for the pure-numpy logistic regression."""

    def test_fit_returns_self(self):
        """fit() should return the instance (fluent API)."""
        X_tr, y_tr, _, _ = _make_separable()
        lr = LogisticRegression(max_iter=10)
        result = lr.fit(X_tr, y_tr)
        assert result is lr

    def test_predict_proba_range(self):
        """predict_proba outputs ∈ [0, 1]."""
        X_tr, y_tr, X_te, _ = _make_separable()
        lr = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        probs = lr.predict_proba(X_te)
        assert probs.shape == (len(X_te),)
        assert np.all(probs >= 0.0 - 1e-9)
        assert np.all(probs <= 1.0 + 1e-9)

    def test_predict_binary(self):
        """predict() returns only 0s and 1s."""
        X_tr, y_tr, X_te, _ = _make_separable()
        lr = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        preds = lr.predict(X_te)
        assert set(preds.tolist()).issubset({0, 1})

    def test_weight_shape_after_fit(self):
        """weights.shape == (D,) after fitting."""
        d = 15
        X_tr, y_tr, _, _ = _make_separable(d=d)
        lr = LogisticRegression(max_iter=10).fit(X_tr, y_tr)
        assert lr.weights.shape == (d,)

    def test_loss_decreases_overall(self):
        """Training loss at the end should be lower than at the start."""
        X_tr, y_tr, _, _ = _make_separable(shift=3.0)
        lr = LogisticRegression(max_iter=500, tol=0.0).fit(X_tr, y_tr)
        assert lr.loss_history[-1] < lr.loss_history[0], \
            "Loss did not decrease during training"

    def test_separable_data_high_auroc(self):
        """Linearly separable data (large margin) → AUROC > 0.85."""
        X_tr, y_tr, X_te, y_te = _make_separable(shift=3.0)
        lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        probs = lr.predict_proba(X_te)

        # Compute AUROC manually (same method as HallucinationDetector)
        from detector import HallucinationDetector
        auroc = HallucinationDetector._compute_auroc(probs, y_te)
        assert auroc > 0.85, f"AUROC={auroc:.4f} too low for separable data"

    def test_all_zeros_labels_no_crash(self):
        """Degenerate case: all labels = 0 should not crash."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5))
        y = np.zeros(50)
        lr = LogisticRegression(max_iter=10)
        lr.fit(X, y)  # must not raise
        probs = lr.predict_proba(X)
        assert np.all(np.isfinite(probs))


# =========================================================================
# Section 2: SimpleMLP
# =========================================================================

class TestSimpleMLP:
    """Tests for the two-layer MLP."""

    def test_fit_returns_self(self):
        """fit() should return the instance."""
        X_tr, y_tr, _, _ = _make_separable()
        mlp = SimpleMLP(hidden_dim=8, max_iter=5)
        result = mlp.fit(X_tr, y_tr)
        assert result is mlp

    def test_predict_proba_range(self):
        """predict_proba outputs ∈ [0, 1]."""
        X_tr, y_tr, X_te, _ = _make_separable()
        mlp = SimpleMLP(hidden_dim=8, max_iter=100).fit(X_tr, y_tr)
        probs = mlp.predict_proba(X_te)
        assert probs.shape == (len(X_te),)
        assert np.all(probs >= 0.0 - 1e-9)
        assert np.all(probs <= 1.0 + 1e-9)

    def test_predict_binary(self):
        """predict() returns only 0s and 1s."""
        X_tr, y_tr, X_te, _ = _make_separable()
        mlp = SimpleMLP(hidden_dim=8, max_iter=100).fit(X_tr, y_tr)
        preds = mlp.predict(X_te)
        assert set(preds.tolist()).issubset({0, 1})

    def test_weight_shapes(self):
        """W1 is (D, hidden_dim); W2 is (hidden_dim,) after fitting."""
        d, h = 10, 16
        X_tr, y_tr, _, _ = _make_separable(d=d)
        mlp = SimpleMLP(hidden_dim=h, max_iter=5).fit(X_tr, y_tr)
        assert mlp.W1.shape == (d, h), f"W1.shape={mlp.W1.shape}"
        assert mlp.W2.shape == (h,), f"W2.shape={mlp.W2.shape}"

    def test_separable_data_decent_auroc(self):
        """MLP should achieve AUROC > 0.6 on linearly separable data."""
        X_tr, y_tr, X_te, y_te = _make_separable(shift=3.0)
        mlp = SimpleMLP(hidden_dim=16, max_iter=500).fit(X_tr, y_tr)
        probs = mlp.predict_proba(X_te)

        from detector import HallucinationDetector
        auroc = HallucinationDetector._compute_auroc(probs, y_te)
        assert auroc > 0.6, f"MLP AUROC={auroc:.4f} too low"


# =========================================================================
# Section 3: HallucinationDetector
# =========================================================================

class TestHallucinationDetector:
    """Tests for the unified HallucinationDetector API."""

    def test_logistic_type_creates_logreg(self):
        """classifier_type='logistic' creates a LogisticRegression model."""
        det = HallucinationDetector(classifier_type="logistic")
        assert isinstance(det.model, LogisticRegression)

    def test_mlp_type_creates_mlp(self):
        """classifier_type='mlp' creates a SimpleMLP model."""
        det = HallucinationDetector(classifier_type="mlp")
        assert isinstance(det.model, SimpleMLP)

    def test_unknown_type_raises_value_error(self):
        """Unknown classifier_type raises ValueError."""
        with pytest.raises(ValueError):
            HallucinationDetector(classifier_type="svm")

    def test_predict_before_fit_raises(self):
        """Calling predict_proba before fit raises AssertionError."""
        det = HallucinationDetector()
        X = np.random.default_rng(0).standard_normal((10, 5))
        with pytest.raises(AssertionError):
            det.predict_proba(X)

    def test_fit_sets_fitted_flag(self):
        """_fitted is True after calling fit()."""
        X_tr, y_tr, _, _ = _make_separable()
        det = HallucinationDetector()
        det.fit(X_tr, y_tr)
        assert det._fitted

    def test_fit_standardises_features(self):
        """fit() stores _mean and _std with correct shape."""
        d = 12
        X_tr, y_tr, _, _ = _make_separable(d=d)
        det = HallucinationDetector().fit(X_tr, y_tr)
        assert det._mean.shape == (d,)
        assert det._std.shape == (d,)

    def test_predict_proba_range(self):
        """predict_proba outputs ∈ [0, 1]."""
        X_tr, y_tr, X_te, _ = _make_separable()
        det = HallucinationDetector().fit(X_tr, y_tr)
        probs = det.predict_proba(X_te)
        assert np.all(probs >= 0.0 - 1e-9)
        assert np.all(probs <= 1.0 + 1e-9)

    def test_evaluate_metrics_valid_ranges(self):
        """All metrics returned by evaluate() lie in valid ranges."""
        X_tr, y_tr, X_te, y_te = _make_separable()
        det = HallucinationDetector().fit(X_tr, y_tr)
        m = det.evaluate(X_te, y_te)
        assert isinstance(m, DetectorMetrics)
        for name, val in [
            ("auroc", m.auroc),
            ("accuracy", m.accuracy),
            ("precision", m.precision),
            ("recall", m.recall),
            ("f1", m.f1),
        ]:
            assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9, \
                f"{name}={val} out of [0,1]"
        assert m.false_positive_rate >= 0.0
        assert m.num_samples == len(y_te)

    def test_auroc_perfect_classifier(self):
        """Large-margin separable data → AUROC > 0.95 for logistic."""
        X_tr, y_tr, X_te, y_te = _make_separable(shift=5.0, n=400)
        det = HallucinationDetector().fit(X_tr, y_tr)
        m = det.evaluate(X_te, y_te)
        assert m.auroc > 0.95, f"AUROC={m.auroc:.4f}"

    def test_feature_importance_logistic(self):
        """feature_importance() returns a dict of length D, sorted by |weight|."""
        d = 10
        X_tr, y_tr, _, _ = _make_separable(d=d)
        names = [f"feat_{i}" for i in range(d)]
        det = HallucinationDetector(
            classifier_type="logistic", feature_names=names
        ).fit(X_tr, y_tr)
        imp = det.feature_importance()
        assert len(imp) == d
        vals = list(imp.values())
        # Sorted descending
        assert vals == sorted(vals, reverse=True)

    def test_feature_importance_mlp_empty(self):
        """MLP has no linear weights; feature_importance returns {}."""
        X_tr, y_tr, _, _ = _make_separable()
        det = HallucinationDetector(classifier_type="mlp").fit(X_tr, y_tr)
        assert det.feature_importance() == {}

    def test_save_load_roundtrip(self):
        """Save + load → identical predict_proba outputs."""
        X_tr, y_tr, X_te, _ = _make_separable()
        det = HallucinationDetector().fit(X_tr, y_tr)
        probs_orig = det.predict_proba(X_te)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmppath = f.name

        try:
            det.save(tmppath)
            det2 = HallucinationDetector.load(tmppath)
            probs_loaded = det2.predict_proba(X_te)
            assert np.allclose(probs_orig, probs_loaded), \
                "Loaded detector gives different predictions"
        finally:
            Path(tmppath).unlink(missing_ok=True)


# =========================================================================
# Section 4: AUROC edge cases
# =========================================================================

class TestComputeAUROC:
    """Tests for the static _compute_auroc method."""

    def test_perfect_classifier(self):
        """Perfect ranking (all positives before negatives) → AUROC = 1.0."""
        probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0], dtype=float)
        auroc = HallucinationDetector._compute_auroc(probs, labels)
        assert abs(auroc - 1.0) < 1e-9, f"AUROC={auroc}"

    def test_worst_classifier(self):
        """Inverted ranking (all negatives before positives) → AUROC = 0.0."""
        probs = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 0, 0], dtype=float)
        auroc = HallucinationDetector._compute_auroc(probs, labels)
        assert abs(auroc - 0.0) < 1e-9, f"AUROC={auroc}"

    def test_all_positive_returns_half(self):
        """All labels = 1 → degenerate case returns 0.5."""
        probs = np.array([0.9, 0.8, 0.7])
        labels = np.ones(3)
        auroc = HallucinationDetector._compute_auroc(probs, labels)
        assert abs(auroc - 0.5) < 1e-9, f"AUROC={auroc}"

    def test_all_negative_returns_half(self):
        """All labels = 0 → degenerate case returns 0.5."""
        probs = np.array([0.9, 0.8, 0.7])
        labels = np.zeros(3)
        auroc = HallucinationDetector._compute_auroc(probs, labels)
        assert abs(auroc - 0.5) < 1e-9, f"AUROC={auroc}"

    def test_random_classifier_near_half(self):
        """Random classifier (uncorrelated probs) → AUROC ≈ 0.5 ± 0.15."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, 1000)
        labels = rng.integers(0, 2, 1000).astype(float)
        auroc = HallucinationDetector._compute_auroc(probs, labels)
        assert 0.35 < auroc < 0.65, f"Random AUROC={auroc:.4f} too far from 0.5"


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
