"""Tests for calibrated_entropy_detector.py"""

import numpy as np
import pytest

from calibrated_entropy_detector import (
    CalibratedEntropyDetector,
    RoutingDecision,
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


def test_route_returns_valid_labels_and_thresholds():
    X, y = _make_synthetic(n=400, d=18)
    X_train, y_train = X[:300], y[:300]
    X_test = X[300:]

    det = CalibratedEntropyDetector()
    det.fit(X_train, y_train)
    decisions = det.route(X_test)

    assert len(decisions) == len(X_test)
    assert all(isinstance(d, RoutingDecision) for d in decisions)
    assert all(d.label in ("RELIABLE", "UNCERTAIN", "UNRELIABLE") for d in decisions)
    assert all(d.action in ("accept", "escalate", "reject") for d in decisions)
    assert all(d.threshold_reliable <= d.threshold_unreliable for d in decisions)
    assert all(0.0 <= d.p_hallucination <= 1.0 for d in decisions)


def test_route_separates_correct_from_hallucinated():
    X, y = _make_synthetic(n=400, d=18)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    det = CalibratedEntropyDetector()
    det.fit(X_train, y_train)
    decisions = det.route(X_test)

    correct_labels = [d.label for d, yi in zip(decisions, y_test) if yi == 0]
    halluc_labels = [d.label for d, yi in zip(decisions, y_test) if yi == 1]
    assert correct_labels.count("RELIABLE") / len(correct_labels) > 0.5
    assert halluc_labels.count("UNRELIABLE") / len(halluc_labels) > 0.5


def test_route_one_matches_route_first_row():
    X, y = _make_synthetic(n=100, d=6)
    det = CalibratedEntropyDetector()
    det.fit(X, y)

    batch = det.route(X[:5])
    single = det.route_one(X[0])
    assert single.label == batch[0].label
    assert single.p_hallucination == pytest.approx(batch[0].p_hallucination)


def test_route_with_no_hallucinated_calibration_examples():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 6))
    y = np.zeros(20)  # no y==1 examples at all
    det = CalibratedEntropyDetector()
    det.fit(X, y)  # must not raise
    decisions = det.route(X)
    assert all(d.label != "UNRELIABLE" for d in decisions)  # threshold disabled at 1.0


def test_score_index_auto_picks_the_strongest_column():
    # Column 3 is a strong, clean separator; the rest are pure noise.
    rng = np.random.default_rng(11)
    N, D = 300, 6
    y = rng.integers(0, 2, N).astype(float)
    X = rng.standard_normal((N, D))
    X[:, 3] += 3.0 * y  # only column 3 carries signal

    det = CalibratedEntropyDetector(score_index="auto")
    det.fit(X, y)
    assert det._state.score_index_resolved == 3


def test_score_index_explicit_int_still_works():
    X, y = _make_synthetic(n=200, d=6)
    det = CalibratedEntropyDetector(score_index=2)
    det.fit(X, y)
    assert det._state.score_index_resolved == 2
    probs = det.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_score_index_invalid_type_raises():
    with pytest.raises(TypeError):
        CalibratedEntropyDetector(score_index="not-a-valid-choice")
    with pytest.raises(TypeError):
        CalibratedEntropyDetector(score_index=1.5)


def test_score_index_auto_resolution_persists_through_save_load(tmp_path):
    rng = np.random.default_rng(12)
    N, D = 300, 6
    y = rng.integers(0, 2, N).astype(float)
    X = rng.standard_normal((N, D))
    X[:, 4] += 3.0 * y

    det = CalibratedEntropyDetector(score_index="auto")
    det.fit(X, y)
    resolved = det._state.score_index_resolved

    path = tmp_path / "calib_auto.pkl"
    det.save(str(path))
    loaded = CalibratedEntropyDetector.load(str(path))

    assert loaded.score_index == "auto"  # constructor spec preserved as-is
    assert loaded._state.score_index_resolved == resolved  # but resolution is preserved too
    assert np.allclose(det.predict_proba(X), loaded.predict_proba(X))


def test_route_thresholds_persist_through_save_load(tmp_path):
    X, y = _make_synthetic(n=200, d=6)
    det = CalibratedEntropyDetector(reliable_quantile=0.8, unreliable_quantile=0.2)
    det.fit(X, y)

    path = tmp_path / "calib_routed.pkl"
    det.save(str(path))
    loaded = CalibratedEntropyDetector.load(str(path))

    assert loaded.reliable_quantile == 0.8
    assert loaded.unreliable_quantile == 0.2
    orig = det.route(X)
    restored = loaded.route(X)
    assert [d.label for d in orig] == [d.label for d in restored]
