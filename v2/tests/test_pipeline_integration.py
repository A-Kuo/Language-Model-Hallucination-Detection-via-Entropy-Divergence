"""
Tests for the Phase 5 generalizations in v2/pipeline.py:
stratified_kfold_cv's detector_factory param and ablation_study's families
param. Uses only synthetic data (generate_synthetic_dataset) — no model
download or network access required.
"""

import numpy as np

from v2.pipeline import (
    generate_synthetic_dataset,
    stratified_kfold_cv,
    ablation_study,
    DEFAULT_ABLATION_FAMILIES,
)
from v2.detector import HallucinationDetector
from v2.calibrated_entropy_detector import CalibratedEntropyDetector


def test_stratified_kfold_cv_classifier_type_still_works():
    X, y = generate_synthetic_dataset(num_samples=200, seed=1)
    auroc, (lo, hi) = stratified_kfold_cv(X, y, k=5, classifier_type="logistic", seed=1)
    assert 0.0 <= auroc <= 1.0
    assert lo <= auroc + 1e-6 <= hi + 1e-6 or lo <= hi  # sane CI ordering


def test_stratified_kfold_cv_with_detector_factory():
    X, y = generate_synthetic_dataset(num_samples=200, seed=2)
    auroc, (lo, hi) = stratified_kfold_cv(
        X, y, k=5, seed=2, detector_factory=lambda: CalibratedEntropyDetector()
    )
    assert 0.0 <= auroc <= 1.0
    assert lo <= hi


def test_stratified_kfold_cv_factory_and_classifier_type_agree_for_logistic():
    X, y = generate_synthetic_dataset(num_samples=200, seed=3)
    auroc_a, _ = stratified_kfold_cv(X, y, k=5, classifier_type="logistic", seed=3)
    auroc_b, _ = stratified_kfold_cv(
        X, y, k=5, seed=3, detector_factory=lambda: HallucinationDetector(classifier_type="logistic")
    )
    assert auroc_a == auroc_b


def test_ablation_study_default_families_runs(capsys):
    X, y = generate_synthetic_dataset(num_samples=200, seed=4)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    ablation_study(X, y, feature_names)
    out = capsys.readouterr().out
    for family in DEFAULT_ABLATION_FAMILIES:
        assert family in out


def test_ablation_study_custom_families_runs(capsys):
    X, y = generate_synthetic_dataset(num_samples=200, seed=5)
    # Simulate a concatenated 18D attention + 6D entropy feature matrix.
    X_extended = np.hstack([X, np.random.default_rng(5).standard_normal((len(y), 6))])
    feature_names = [f"f{i}" for i in range(X_extended.shape[1])]

    families = dict(DEFAULT_ABLATION_FAMILIES)
    families["entropy_token"] = slice(18, 24)

    ablation_study(X_extended, y, feature_names, families=families)
    out = capsys.readouterr().out
    assert "entropy_token" in out
    assert "Full model (24 features)" in out
