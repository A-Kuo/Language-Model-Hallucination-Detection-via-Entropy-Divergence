"""Tests for v2/abstention.py"""

import numpy as np
import pytest

from v2.abstention import (
    risk_coverage_curve,
    _evaluate_threshold,
    area_under_risk_coverage,
    run_abstention_experiment,
    save_risk_coverage_table,
)
from v2.detector import HallucinationDetector
from v2.pipeline import generate_synthetic_dataset


def test_coverage_monotonic_in_threshold():
    rng = np.random.default_rng(0)
    probs = rng.random(200)
    y = rng.integers(0, 2, 200).astype(float)

    points = risk_coverage_curve(probs, y, n_thresholds=30)
    coverages = [p.coverage for p in points]
    assert all(c1 <= c2 + 1e-9 for c1, c2 in zip(coverages, coverages[1:]))


def test_full_coverage_equals_baseline_accuracy():
    rng = np.random.default_rng(1)
    probs = rng.random(200)
    y = rng.integers(0, 2, 200).astype(float)

    points = risk_coverage_curve(probs, y, n_thresholds=30)
    full_coverage_point = max(points, key=lambda p: p.coverage)
    assert full_coverage_point.coverage == pytest.approx(1.0)

    expected_baseline_accuracy = float((y == 0).mean())
    assert full_coverage_point.accuracy == pytest.approx(expected_baseline_accuracy)


def test_zero_coverage_edge_case_no_divide_by_zero():
    probs = np.array([0.3, 0.4, 0.5])
    y = np.array([0.0, 1.0, 0.0])

    point = _evaluate_threshold(probs, y, tau=0.0)  # below every score
    assert point.n_answered == 0
    assert point.coverage == 0.0
    assert np.isnan(point.accuracy)


def test_aurc_bounds():
    rng = np.random.default_rng(2)
    probs = rng.random(300)
    y = rng.integers(0, 2, 300).astype(float)

    points = risk_coverage_curve(probs, y, n_thresholds=50)
    aurc = area_under_risk_coverage(points)
    assert 0.0 <= aurc <= 1.0


def test_aurc_nan_with_insufficient_valid_points():
    # A single point can't be integrated.
    points = [_evaluate_threshold(np.array([0.5]), np.array([0.0]), tau=0.5)]
    assert np.isnan(area_under_risk_coverage(points))


def test_run_abstention_experiment_reports_gain_on_separable_data():
    X, y = generate_synthetic_dataset(num_samples=400, seed=42)
    result = run_abstention_experiment(
        X, y, detector_factory=lambda: HallucinationDetector(classifier_type="logistic"), seed=42
    )
    assert "aurc" in result
    assert 0.0 <= result["baseline_accuracy"] <= 1.0
    assert 0.0 <= result["headline_accuracy"] <= 1.0
    # On well-separated synthetic data, abstaining on high-risk examples
    # should not hurt accuracy relative to answering everything.
    assert result["headline_gain"] >= -1e-6


def test_save_risk_coverage_table_writes_csv(tmp_path):
    points = risk_coverage_curve(
        np.array([0.1, 0.2, 0.3, 0.9]), np.array([0.0, 0.0, 1.0, 1.0]), n_thresholds=5
    )
    out_path = tmp_path / "curve.csv"
    save_risk_coverage_table(points, str(out_path))

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert lines[0] == "threshold,coverage,accuracy,n_answered,n_abstained"
    assert len(lines) == 1 + len(points)
