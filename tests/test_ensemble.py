"""
Tests for ensemble.py -- the detector-stacking ensemble on matched-pair
HaluEval. Synthetic data only (no model/network), mirroring
tests/test_pipeline_integration.py's pattern. Uses a small bilstm_epochs
for speed -- these tests check mechanics/shapes, not accuracy.
"""

import numpy as np
import pytest

from ensemble import (
    BASE_DETECTOR_NAMES,
    VARIANTS,
    average_precision,
    brier_score,
    build_meta_features,
    expected_calibration_error,
    log_loss,
    nested_cv_stacking,
    paired_bootstrap_delta_ci,
    precision_at_recall,
    precision_recall_curve_for_hallucination,
    _active_detector_names,
    _stratified_folds,
)


def _synthetic_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(float)
    # Separable-ish signal so base detectors have something to learn.
    X = rng.standard_normal((n, 24))
    X[y == 1] += 1.0
    X_blackbox = rng.standard_normal((n, 7))
    X_blackbox[y == 1] += 1.0
    L = 4
    X_seq = rng.standard_normal((n, L, 6))
    X_seq[y == 1] += 0.5
    return X, X_blackbox, X_seq, y


# --- _stratified_folds --------------------------------------------------

def test_stratified_folds_partition_exactly():
    y = np.concatenate([np.zeros(40), np.ones(60)])
    folds = _stratified_folds(y, k=5, seed=1)
    assert len(folds) == 5
    all_idx = np.concatenate(folds)
    assert sorted(all_idx.tolist()) == list(range(100))  # covers everything exactly once
    assert len(set(all_idx.tolist())) == 100  # no duplicates -- no leakage across folds


def test_stratified_folds_are_class_balanced():
    y = np.concatenate([np.zeros(50), np.ones(50)])
    folds = _stratified_folds(y, k=5, seed=2)
    for fold in folds:
        frac_positive = y[fold].mean()
        assert 0.3 < frac_positive < 0.7  # roughly balanced, not skewed to one class


# --- build_meta_features -------------------------------------------------

def test_build_meta_features_shapes():
    n = 50
    base_probs = {name: np.random.default_rng(3).random(n) for name in BASE_DETECTOR_NAMES}

    meta_X, names = build_meta_features(base_probs, include_aggregate=False)
    assert meta_X.shape == (n, 10)  # 5 raw + 5 rank
    assert len(names) == 10

    meta_X2, names2 = build_meta_features(base_probs, include_aggregate=True)
    assert meta_X2.shape == (n, 14)  # + mean/std/range/disagreement
    assert len(names2) == 14
    assert "disagreement" in names2


def test_build_meta_features_rank_is_valid_percentile():
    n = 20
    base_probs = {name: np.arange(n, dtype=float) for name in BASE_DETECTOR_NAMES}
    meta_X, names = build_meta_features(base_probs, include_aggregate=False)
    rank_col = names.index("calibrated_entropy_rank")
    assert np.isclose(meta_X[0, rank_col], 0.0)
    assert np.isclose(meta_X[-1, rank_col], 1.0)


def test_disagreement_is_zero_when_all_detectors_agree():
    n = 20
    same = np.random.default_rng(4).random(n)
    base_probs = {name: same.copy() for name in BASE_DETECTOR_NAMES}
    meta_X, names = build_meta_features(base_probs, include_aggregate=True)
    disagreement_col = names.index("disagreement")
    assert np.allclose(meta_X[:, disagreement_col], 0.0)


# --- nested_cv_stacking (end-to-end mechanics, synthetic, small epochs) --

@pytest.mark.parametrize("variant", VARIANTS)
def test_nested_cv_stacking_runs_end_to_end(variant):
    X, X_blackbox, X_seq, y = _synthetic_dataset(n=100, seed=5)
    result = nested_cv_stacking(
        X, X_blackbox, X_seq, y,
        outer_k=3, inner_k=3, seed=5, variant=variant, bilstm_epochs=3,
    )
    assert result["ensemble_probs"].shape == y.shape
    assert np.all((result["ensemble_probs"] >= 0) & (result["ensemble_probs"] <= 1))
    # Not BASE_DETECTOR_NAMES directly -- BiLSTM is absent when torch isn't
    # installed (matches pipeline.py's graceful degradation; see
    # ensemble.py::_active_detector_names).
    for name in _active_detector_names():
        assert result["base_probs"][name].shape == y.shape


def test_nested_cv_stacking_works_without_torch(monkeypatch):
    """Regression test for the exact bug that broke CI (numpy/scipy/pytest
    only, no torch -- .github/workflows/test.yml): the ensemble must not
    hard-require BiLSTM. Forces the torch-unavailable path directly rather
    than relying on the test environment happening to lack torch."""
    import ensemble

    monkeypatch.setattr(ensemble, "_HAS_TORCH", False)
    X, X_blackbox, X_seq, y = _synthetic_dataset(n=60, seed=12)
    result = ensemble.nested_cv_stacking(
        X, X_blackbox, X_seq, y,
        outer_k=3, inner_k=3, seed=12, variant="stacker_disagreement", bilstm_epochs=3,
    )
    assert "bilstm" not in result["base_probs"]
    assert set(result["base_probs"].keys()) == {"calibrated_entropy", "logistic", "mlp", "blackbox"}
    assert result["ensemble_probs"].shape == y.shape


def test_nested_cv_stacking_rejects_unknown_variant():
    X, X_blackbox, X_seq, y = _synthetic_dataset(n=30, seed=6)
    with pytest.raises(ValueError):
        nested_cv_stacking(X, X_blackbox, X_seq, y, outer_k=3, inner_k=3, variant="not_a_variant")


def test_nested_cv_stacking_produces_reasonable_separation():
    """On separable synthetic data, pooled out-of-sample AUROC should be
    well above chance -- a sanity check that the nested loop actually wires
    train/predict correctly rather than, say, silently leaving zeros."""
    from detector import compute_auroc

    X, X_blackbox, X_seq, y = _synthetic_dataset(n=150, seed=7)
    result = nested_cv_stacking(
        X, X_blackbox, X_seq, y,
        outer_k=3, inner_k=3, seed=7, variant="uniform_mean", bilstm_epochs=3,
    )
    auroc = compute_auroc(result["ensemble_probs"], y)
    assert auroc > 0.7


# --- Metrics --------------------------------------------------------------

def test_pr_auc_high_for_separable_scores():
    rng = np.random.default_rng(8)
    probs = np.concatenate([rng.uniform(0.0, 0.4, size=50), rng.uniform(0.6, 1.0, size=50)])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    points = precision_recall_curve_for_hallucination(probs, y)
    assert average_precision(points) > 0.9


def test_pr_auc_near_base_rate_for_random_scores():
    rng = np.random.default_rng(9)
    n = 500
    probs = rng.uniform(0, 1, size=n)
    base_rate = 0.3
    y = (rng.uniform(size=n) < base_rate).astype(float)
    points = precision_recall_curve_for_hallucination(probs, y)
    assert abs(average_precision(points) - base_rate) < 0.1


def test_precision_at_recall_returns_nan_when_unreachable():
    probs = np.array([0.1, 0.2, 0.3])
    y = np.array([0.0, 0.0, 0.0])  # no positive examples at all -- recall undefined/unreachable
    points = precision_recall_curve_for_hallucination(probs, y)
    assert np.isnan(precision_at_recall(points, 0.95))


def test_brier_score_zero_for_perfect_predictions():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, y) == 0.0


def test_log_loss_low_for_confident_correct_predictions():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    probs = np.array([0.01, 0.99, 0.01, 0.99])
    assert log_loss(probs, y) < 0.05


def test_ece_near_zero_for_well_calibrated_predictions():
    rng = np.random.default_rng(10)
    n = 2000
    probs = rng.uniform(0, 1, size=n)
    y = (rng.uniform(size=n) < probs).astype(float)  # labels genuinely match stated confidence
    assert expected_calibration_error(probs, y, n_bins=10) < 0.05


def test_ece_high_for_badly_calibrated_predictions():
    n = 100
    probs = np.full(n, 0.95)  # always very confident
    y = np.zeros(n)  # but always wrong
    assert expected_calibration_error(probs, y, n_bins=10) > 0.8


def test_paired_bootstrap_delta_ci_straddles_zero_for_identical_arrays():
    rng = np.random.default_rng(11)
    n = 200
    probs = rng.uniform(0, 1, size=n)
    y = (rng.uniform(size=n) > 0.5).astype(float)
    delta_mean, (lo, hi) = paired_bootstrap_delta_ci(probs, probs, y, n_boot=200, seed=11)
    assert delta_mean == 0.0
    assert lo <= 0.0 <= hi
