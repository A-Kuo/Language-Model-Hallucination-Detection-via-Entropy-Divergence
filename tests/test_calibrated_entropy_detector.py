"""Tests for calibrated_entropy_detector.py"""

import numpy as np
import pytest

from calibrated_entropy_detector import (
    CalibratedEntropyDetector,
    isotonic_regression,
    mahalanobis_distance,
    fit_reference_distribution,
    percentile_rank,
)
from detector import DetectorMetrics


def _make_synthetic(n=400, d=18, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    X = rng.standard_normal((n, d))
    X[y == 1] += 1.0
    return X, y


def test_fit_predict_shapes():
    X, y = _make_synthetic(n=100, d=6)
    det = CalibratedEntropyDetector()
    det.fit(X, y)
    probs = det.predict_proba(X)
    assert probs.shape == (len(y),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_high_d_low_n_covariance_shrinkage_no_crash():
    rng = np.random.default_rng(7)
    D, N = 24, 10
    y = rng.integers(0, 2, N).astype(float)
    X = rng.standard_normal((N, D))
    det = CalibratedEntropyDetector(cov_shrinkage=0.3)
    det.fit(X, y)  # must not raise LinAlgError
    probs = det.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))
    assert not np.any(np.isnan(probs))


def test_separates_synthetic_shifted_classes():
    X, y = _make_synthetic(n=400, d=18)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    det = CalibratedEntropyDetector()
    det.fit(X_train, y_train)
    metrics = det.evaluate(X_test, y_test)
    assert metrics.auroc > 0.7


def test_isotonic_monotonic_on_probe_grid():
    x = np.linspace(-3, 3, 200)
    y = (x > 0).astype(float)
    predict = isotonic_regression(x, y)
    grid = np.linspace(-5, 5, 500)
    fitted = predict(grid)
    assert np.all(np.diff(fitted) >= -1e-9)


def test_isotonic_extrapolation_clips_to_boundary():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.1, 0.5, 0.9])
    predict = isotonic_regression(x, y)
    assert predict(np.array([-10.0]))[0] == pytest.approx(0.1)
    assert predict(np.array([10.0]))[0] == pytest.approx(0.9)


def test_evaluate_matches_detectormetrics_contract():
    X, y = _make_synthetic(n=100, d=6)
    det = CalibratedEntropyDetector()
    det.fit(X, y)
    metrics = det.evaluate(X, y)
    assert isinstance(metrics, DetectorMetrics)
    for field in ("auroc", "accuracy", "precision", "recall", "f1", "false_positive_rate", "num_samples"):
        assert hasattr(metrics, field)


def test_mahalanobis_zero_at_reference_mean():
    mu = np.array([1.0, 2.0])
    sigma_inv = np.eye(2)
    dist = mahalanobis_distance(np.array([[1.0, 2.0]]), mu, sigma_inv)
    assert dist[0] == pytest.approx(0.0)


def test_percentile_rank_bounds():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = np.array([0.0, 3.0, 10.0])
    ranks = percentile_rank(scores, ref)
    assert ranks[0] == 0.0
    assert ranks[2] == 1.0
    assert 0.0 <= ranks[1] <= 1.0


def test_fit_reference_distribution_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 6))
    mu, sigma_inv = fit_reference_distribution(X, shrinkage=0.1)
    assert mu.shape == (6,)
    assert sigma_inv.shape == (6, 6)


def test_fit_raises_with_too_few_correct_examples():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 6))
    y = np.ones(10)  # no y==0 examples at all
    det = CalibratedEntropyDetector()
    with pytest.raises(ValueError, match="correct-answer"):
        det.fit(X, y)


def test_save_load_roundtrip(tmp_path):
    X, y = _make_synthetic(n=100, d=6)
    det = CalibratedEntropyDetector()
    det.fit(X, y)

    path = tmp_path / "calib.pkl"
    det.save(str(path))
    loaded = CalibratedEntropyDetector.load(str(path))

    assert np.allclose(det.predict_proba(X), loaded.predict_proba(X))


def test_percentile_divergence_mode_runs():
    X, y = _make_synthetic(n=200, d=6)
    det = CalibratedEntropyDetector(divergence="percentile")
    det.fit(X, y)
    probs = det.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_invalid_divergence_raises():
    with pytest.raises(ValueError):
        CalibratedEntropyDetector(divergence="nonsense")
