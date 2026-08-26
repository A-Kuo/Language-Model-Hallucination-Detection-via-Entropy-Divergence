"""
Tests for v2/detector.py

Covers the classifier_type misuse guards (fit/predict_proba/evaluate raising
a clear ValueError instead of an AttributeError when classifier_type="bilstm",
and fit_sequence raising ValueError for non-bilstm types), plus regression
tests promoted from the module's __main__ self-test block so they run under
pytest/CI instead of only on manual invocation.
"""

import numpy as np
import pytest

from v2.detector import HallucinationDetector, compute_classification_metrics, compute_auroc

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _make_synthetic_flat(n=400, d=18, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    X = rng.standard_normal((n, d))
    X[y == 1] += 1.0
    return X, y


def _make_synthetic_sequence(n=200, l=6, d=6, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    X = rng.standard_normal((n, l, d))
    X[y == 1] += 1.0
    return X, y


# ---------------------------------------------------------------------------
# Guard regressions (Phase 0 critical fix)
# ---------------------------------------------------------------------------

def test_bilstm_fit_raises_clear_valueerror():
    if not _HAS_TORCH:
        pytest.skip("torch not installed")
    det = HallucinationDetector(classifier_type="bilstm")
    X, y = _make_synthetic_flat()
    with pytest.raises(ValueError, match="fit_sequence"):
        det.fit(X, y)


def test_bilstm_predict_proba_raises_clear_valueerror():
    if not _HAS_TORCH:
        pytest.skip("torch not installed")
    det = HallucinationDetector(classifier_type="bilstm")
    X, _ = _make_synthetic_flat()
    with pytest.raises(ValueError, match="predict_proba_sequence"):
        det.predict_proba(X)


def test_bilstm_evaluate_raises_clear_valueerror():
    if not _HAS_TORCH:
        pytest.skip("torch not installed")
    det = HallucinationDetector(classifier_type="bilstm")
    X, y = _make_synthetic_flat()
    with pytest.raises(ValueError, match="evaluate_sequence"):
        det.evaluate(X, y)


def test_logistic_fit_sequence_raises_valueerror():
    det = HallucinationDetector(classifier_type="logistic")
    X_seq, y = _make_synthetic_sequence()
    with pytest.raises(ValueError, match="fit_sequence"):
        det.fit_sequence(X_seq, y)


def test_mlp_fit_sequence_raises_valueerror():
    det = HallucinationDetector(classifier_type="mlp")
    X_seq, y = _make_synthetic_sequence()
    with pytest.raises(ValueError):
        det.fit_sequence(X_seq, y)


# ---------------------------------------------------------------------------
# Promoted from __main__ self-test block
# ---------------------------------------------------------------------------

def test_logistic_regression_separates_synthetic_data():
    X, y = _make_synthetic_flat()
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    det = HallucinationDetector(
        classifier_type="logistic",
        feature_names=[f"feat_{i}" for i in range(X.shape[1])],
    )
    det.fit(X_train, y_train)
    metrics = det.evaluate(X_test, y_test)
    assert metrics.auroc > 0.7


def test_mlp_separates_synthetic_data():
    X, y = _make_synthetic_flat()
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    det = HallucinationDetector(classifier_type="mlp", hidden_dim=16)
    det.fit(X_train, y_train)
    metrics = det.evaluate(X_test, y_test)
    assert metrics.auroc > 0.6


def test_feature_importance_only_for_logistic():
    X, y = _make_synthetic_flat()
    d = X.shape[1]

    det_lr = HallucinationDetector(
        classifier_type="logistic", feature_names=[f"feat_{i}" for i in range(d)]
    )
    det_lr.fit(X, y)
    importance = det_lr.feature_importance()
    assert len(importance) == d

    det_mlp = HallucinationDetector(classifier_type="mlp")
    det_mlp.fit(X, y)
    assert det_mlp.feature_importance() == {}


def test_save_load_roundtrip(tmp_path):
    X, y = _make_synthetic_flat()
    det = HallucinationDetector(classifier_type="logistic")
    det.fit(X, y)

    path = tmp_path / "detector.pkl"
    det.save(str(path))
    loaded = HallucinationDetector.load(str(path))

    assert np.allclose(det.predict_proba(X), loaded.predict_proba(X))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_bilstm_fit_sequence_predict_evaluate_smoke():
    X_seq, y = _make_synthetic_sequence()
    X_train, y_train = X_seq[:150], y[:150]
    X_test, y_test = X_seq[150:], y[150:]

    det = HallucinationDetector(classifier_type="bilstm", hidden_dim=8, epochs=5)
    det.fit_sequence(X_train, y_train)

    probs = det.predict_proba_sequence(X_test)
    assert probs.shape == (len(y_test),)
    assert np.all((probs >= 0) & (probs <= 1))

    metrics = det.evaluate_sequence(X_test, y_test)
    assert 0.0 <= metrics.auroc <= 1.0


# ---------------------------------------------------------------------------
# Shared metrics helper
# ---------------------------------------------------------------------------

def test_compute_classification_metrics_matches_manual_confusion_matrix():
    y = np.array([1, 1, 0, 0, 1, 0])
    probs = np.array([0.9, 0.4, 0.2, 0.8, 0.6, 0.1])
    metrics = compute_classification_metrics(probs, y, threshold=0.5)

    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    assert metrics.precision == pytest.approx(tp / max(tp + fp, 1))
    assert metrics.recall == pytest.approx(tp / max(tp + fn, 1))
    assert metrics.num_samples == len(y)


def test_compute_auroc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auroc(probs, y) == pytest.approx(1.0)
