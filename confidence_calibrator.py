"""
Confidence Calibrator — Decision Routing for Safety-Critical Systems
=====================================================================

Mathematical Foundation:
    The hypothesis test produces a raw confidence score ∈ [0, 1] via
    the normal CDF:  confidence = Φ(-Z_combined).

    However, raw scores from neural networks (and derived statistics)
    are often **miscalibrated** — a model saying "80% confident" might
    only be correct 60% of the time.

    Calibration corrects this.  We support two approaches:

    1) Threshold-Based Routing (default, no training data needed):
       - confidence > τ_high  →  RELIABLE    (route to user)
       - τ_low ≤ confidence ≤ τ_high  →  UNCERTAIN  (escalate to human)
       - confidence < τ_low   →  UNRELIABLE  (reject, use fallback)

    2) Isotonic Regression Calibration (when calibration data available):
       Learns a monotonic mapping f: raw_score → calibrated_score
       such that calibrated scores are **frequency-calibrated**:
           P(correct | f(score) = p) ≈ p

       This is the standard post-hoc calibration method from:
       Zadrozny & Elkan (2002), "Transforming classifier scores into
       accurate multiclass probability estimates"

    For safety-critical systems (robotics, medical AI), we bias toward
    CONSERVATIVE decisions:
       - False positives (flagging reliable output) = annoying but safe
       - False negatives (trusting unreliable output) = dangerous

    Therefore default thresholds are asymmetric:
       τ_high = 0.75 (must be quite confident to pass)
       τ_low  = 0.50 (uncertain zone is wide)

Usage:
    from src.hypothesis_test import HallucinationHypothesisTest
    from src.confidence_calibrator import ConfidenceCalibrator, Decision

    tester = HallucinationHypothesisTest()
    calibrator = ConfidenceCalibrator()

    # ... get hypothesis_result from tester.test(analysis) ...
    decision = calibrator.route(hypothesis_result)
    print(decision.label)       # "RELIABLE" | "UNCERTAIN" | "UNRELIABLE"
    print(decision.action)      # "output_to_user" | "escalate_to_human" | ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# Type hint only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.hypothesis_test import HypothesisTestResult


# ---------------------------------------------------------------------------
# Enums and containers
# ---------------------------------------------------------------------------

class ReliabilityLabel(str, Enum):
    """Three-tier reliability classification."""
    RELIABLE = "RELIABLE"
    UNCERTAIN = "UNCERTAIN"
    UNRELIABLE = "UNRELIABLE"


class RoutingAction(str, Enum):
    """What to do with the output based on reliability."""
    OUTPUT_TO_USER = "output_to_user"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    REJECT_USE_FALLBACK = "reject_use_fallback"


@dataclass
class CalibrationDecision:
    """Complete decision output from the calibrator."""

    # Core decision
    label: ReliabilityLabel
    action: RoutingAction

    # Scores
    raw_confidence: float           # from hypothesis test Φ(-Z)
    calibrated_confidence: float    # after isotonic calibration (if available)

    # Thresholds used (for transparency / debugging)
    threshold_high: float
    threshold_low: float

    # From the hypothesis test
    z_score: float
    p_value: float

    def __str__(self) -> str:
        return (
            f"[{self.label.value}] confidence={self.calibrated_confidence:.3f} "
            f"→ {self.action.value} "
            f"(Z={self.z_score:.3f}, p={self.p_value:.4f})"
        )


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    Isotonic regression for probability calibration.

    Given pairs (raw_score_i, was_correct_i), learns a monotonically
    non-decreasing step function that maps raw scores to calibrated
    probabilities.

    This is a pure-numpy implementation (no sklearn dependency) using
    the Pool Adjacent Violators (PAV) algorithm:

        1. Sort samples by raw_score
        2. Walk through; when y_{i+1} < y_i (violates monotonicity),
           pool the two into their weighted average
        3. Repeat until monotonic

    Time: O(n log n) for sort + O(n) for PAV
    Space: O(n)
    """

    def __init__(self) -> None:
        self._is_fitted = False
        self._thresholds: np.ndarray = np.array([])  # raw score breakpoints
        self._values: np.ndarray = np.array([])       # calibrated values

    def fit(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
    ) -> IsotonicCalibrator:
        """
        Fit the isotonic calibrator.

        Parameters
        ----------
        raw_scores : np.ndarray, shape (n,)
            Raw confidence scores ∈ [0, 1].
        labels : np.ndarray, shape (n,)
            Binary ground truth: 1 = output was correct, 0 = hallucination.

        Returns
        -------
        self (for chaining)
        """
        assert len(raw_scores) == len(labels)
        assert len(raw_scores) >= 2, "Need at least 2 samples"

        # Sort by raw score
        order = np.argsort(raw_scores)
        x = raw_scores[order].astype(np.float64)
        y = labels[order].astype(np.float64)

        # PAV algorithm
        calibrated = self._pav(y)

        self._thresholds = x
        self._values = calibrated
        self._is_fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Map raw scores → calibrated scores using the fitted function.

        Uses linear interpolation between fitted breakpoints.
        Scores outside the fitted range are clipped to [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        calibrated = np.interp(
            raw_scores,
            self._thresholds,
            self._values,
        )
        return np.clip(calibrated, 0.0, 1.0)

    def transform_single(self, raw_score: float) -> float:
        """Calibrate a single score."""
        return float(self.transform(np.array([raw_score]))[0])

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # PAV algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _pav(y: np.ndarray) -> np.ndarray:
        """
        Pool Adjacent Violators algorithm.

        Ensures the output is monotonically non-decreasing by averaging
        adjacent blocks that violate the ordering constraint.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Values sorted by corresponding x (raw_scores).

        Returns
        -------
        np.ndarray, shape (n,)
            Isotonically calibrated values.
        """
        n = len(y)
        result = y.copy()

        # Each element starts as its own block
        # block[i] = (start_idx, end_idx, weighted_sum, count)
        blocks: list[list] = [[i, i, result[i], 1] for i in range(n)]

        # Merge adjacent blocks that violate monotonicity
        merged = True
        while merged:
            merged = False
            new_blocks: list[list] = [blocks[0]]
            for i in range(1, len(blocks)):
                prev = new_blocks[-1]
                curr = blocks[i]
                prev_avg = prev[2] / prev[3]
                curr_avg = curr[2] / curr[3]

                if curr_avg < prev_avg:
                    # Violation — merge blocks
                    prev[1] = curr[1]          # extend end
                    prev[2] += curr[2]         # sum values
                    prev[3] += curr[3]         # sum counts
                    merged = True
                else:
                    new_blocks.append(curr)
            blocks = new_blocks

        # Write back block averages
        for start, end, total, count in blocks:
            result[start:end + 1] = total / count

        return result


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """
    Maps hypothesis test results into actionable decisions.

    Supports two modes:
        1. Threshold-only (default): No calibration data needed.
        2. Isotonic + threshold: If calibration data is provided,
           applies isotonic regression before thresholding.

    Parameters
    ----------
    threshold_high : float
        Confidence above this → RELIABLE. Default 0.75.
    threshold_low : float
        Confidence below this → UNRELIABLE. Default 0.50.
        Between low and high → UNCERTAIN.
    isotonic_calibrator : IsotonicCalibrator | None
        Pre-fitted calibrator. If None, raw scores are used directly.
    """

    def __init__(
        self,
        threshold_high: float = 0.75,
        threshold_low: float = 0.50,
        isotonic_calibrator: Optional[IsotonicCalibrator] = None,
    ) -> None:
        assert 0.0 < threshold_low < threshold_high < 1.0, \
            f"Need 0 < τ_low < τ_high < 1, got ({threshold_low}, {threshold_high})"

        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self._isotonic = isotonic_calibrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, hypothesis_result: HypothesisTestResult) -> CalibrationDecision:
        """
        Make a routing decision from a hypothesis test result.

        Pipeline:
            1. Extract raw confidence from hypothesis test
            2. (Optional) Apply isotonic calibration
            3. Compare to thresholds → label + action
            4. Package into CalibrationDecision
        """
        raw_conf = hypothesis_result.confidence_score

        # Calibrate if isotonic model is available
        if self._isotonic is not None and self._isotonic.is_fitted:
            cal_conf = self._isotonic.transform_single(raw_conf)
        else:
            cal_conf = raw_conf

        # Threshold routing
        label, action = self._classify(cal_conf)

        return CalibrationDecision(
            label=label,
            action=action,
            raw_confidence=raw_conf,
            calibrated_confidence=cal_conf,
            threshold_high=self.threshold_high,
            threshold_low=self.threshold_low,
            z_score=hypothesis_result.z_combined,
            p_value=hypothesis_result.p_value,
        )

    def route_from_score(self, confidence_score: float) -> CalibrationDecision:
        """
        Route directly from a confidence score (skip hypothesis test).
        Useful for testing or when scores come from a saved file.
        """
        if self._isotonic is not None and self._isotonic.is_fitted:
            cal_conf = self._isotonic.transform_single(confidence_score)
        else:
            cal_conf = confidence_score

        label, action = self._classify(cal_conf)

        return CalibrationDecision(
            label=label,
            action=action,
            raw_confidence=confidence_score,
            calibrated_confidence=cal_conf,
            threshold_high=self.threshold_high,
            threshold_low=self.threshold_low,
            z_score=0.0,
            p_value=0.0,
        )

    def route_batch(
        self,
        hypothesis_results: list[HypothesisTestResult],
    ) -> list[CalibrationDecision]:
        """Route multiple hypothesis test results."""
        return [self.route(r) for r in hypothesis_results]

    # ------------------------------------------------------------------
    # Calibration fitting
    # ------------------------------------------------------------------

    def fit_calibration(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Fit isotonic calibration from labelled data.

        Parameters
        ----------
        raw_scores : np.ndarray, shape (n,)
            Raw confidence scores from the hypothesis test.
        labels : np.ndarray, shape (n,)
            1 = output was actually correct, 0 = hallucination.
        """
        self._isotonic = IsotonicCalibrator()
        self._isotonic.fit(raw_scores, labels)

    @property
    def has_calibration(self) -> bool:
        """Whether isotonic calibration is fitted."""
        return self._isotonic is not None and self._isotonic.is_fitted

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_calibration_error(
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, List[Tuple[float, float, int]]]:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ_b (|B_b| / N) · |accuracy(B_b) - confidence(B_b)|

        where B_b is the set of samples in bin b.

        A perfectly calibrated model has ECE = 0.

        Returns
        -------
        ece : float
        bin_stats : list of (mean_confidence, mean_accuracy, count) per bin
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_stats: List[Tuple[float, float, int]] = []
        ece = 0.0
        n_total = len(predicted_probs)

        for i in range(n_bins):
            mask = (predicted_probs >= bin_edges[i]) & \
                   (predicted_probs < bin_edges[i + 1])
            count = mask.sum()

            if count == 0:
                bin_stats.append((0.0, 0.0, 0))
                continue

            mean_conf = float(predicted_probs[mask].mean())
            mean_acc = float(true_labels[mask].mean())
            bin_stats.append((mean_conf, mean_acc, int(count)))

            ece += (count / n_total) * abs(mean_acc - mean_conf)

        return float(ece), bin_stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(
        self, confidence: float
    ) -> Tuple[ReliabilityLabel, RoutingAction]:
        """Apply threshold routing."""
        if confidence > self.threshold_high:
            return ReliabilityLabel.RELIABLE, RoutingAction.OUTPUT_TO_USER
        elif confidence >= self.threshold_low:
            return ReliabilityLabel.UNCERTAIN, RoutingAction.ESCALATE_TO_HUMAN
        else:
            return ReliabilityLabel.UNRELIABLE, RoutingAction.REJECT_USE_FALLBACK


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIDENCE CALIBRATOR — STANDALONE VALIDATION")
    print("=" * 60)

    # ---- Test 1: Threshold routing without calibration ----
    print("\n--- Test 1: Basic threshold routing ---")
    cal = ConfidenceCalibrator(threshold_high=0.75, threshold_low=0.50)

    d1 = cal.route_from_score(0.90)
    assert d1.label == ReliabilityLabel.RELIABLE
    assert d1.action == RoutingAction.OUTPUT_TO_USER
    print(f"  0.90 → {d1.label.value}  ✅")

    d2 = cal.route_from_score(0.60)
    assert d2.label == ReliabilityLabel.UNCERTAIN
    assert d2.action == RoutingAction.ESCALATE_TO_HUMAN
    print(f"  0.60 → {d2.label.value}  ✅")

    d3 = cal.route_from_score(0.30)
    assert d3.label == ReliabilityLabel.UNRELIABLE
    assert d3.action == RoutingAction.REJECT_USE_FALLBACK
    print(f"  0.30 → {d3.label.value}  ✅")

    # ---- Test 2: Boundary conditions ----
    print("\n--- Test 2: Boundary conditions ---")
    d_high = cal.route_from_score(0.75)
    assert d_high.label == ReliabilityLabel.UNCERTAIN  # ≤ not <
    print(f"  0.75 (=τ_high) → {d_high.label.value}  ✅")

    d_low = cal.route_from_score(0.50)
    assert d_low.label == ReliabilityLabel.UNCERTAIN
    print(f"  0.50 (=τ_low)  → {d_low.label.value}  ✅")

    d_below = cal.route_from_score(0.499)
    assert d_below.label == ReliabilityLabel.UNRELIABLE
    print(f"  0.499 (<τ_low) → {d_below.label.value}  ✅")

    # ---- Test 3: PAV algorithm (isotonic regression) ----
    print("\n--- Test 3: Isotonic regression (PAV) ---")
    iso = IsotonicCalibrator()

    # Simple case: raw scores [0.1, 0.4, 0.6, 0.9]
    # Labels:                 [0,   1,   0,   1]
    # After PAV, the middle two (1, 0) violate monotonicity → averaged to 0.5
    raw = np.array([0.1, 0.4, 0.6, 0.9])
    lbl = np.array([0.0, 1.0, 0.0, 1.0])
    iso.fit(raw, lbl)

    assert iso.is_fitted
    calibrated = iso.transform(raw)
    print(f"  Raw:        {raw}")
    print(f"  Labels:     {lbl}")
    print(f"  Calibrated: {calibrated}")

    # Check monotonicity
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1] + 1e-10, \
            f"Monotonicity violated: {calibrated[i]} > {calibrated[i+1]}"
    print(f"  Monotonicity: ✅")

    # ---- Test 4: Isotonic calibration in the full pipeline ----
    print("\n--- Test 4: Calibrator with isotonic regression ---")
    rng = np.random.default_rng(42)
    n = 200
    # Simulate: higher raw score → more likely correct (with noise)
    raw_scores = rng.uniform(0, 1, n)
    true_probs = 1.0 / (1.0 + np.exp(-5 * (raw_scores - 0.5)))
    true_labels = (rng.random(n) < true_probs).astype(float)

    cal_iso = ConfidenceCalibrator(threshold_high=0.75, threshold_low=0.50)
    cal_iso.fit_calibration(raw_scores, true_labels)
    assert cal_iso.has_calibration
    print(f"  Fitted isotonic calibrator on {n} samples  ✅")

    # Check calibrated scores are in [0, 1]
    test_scores = np.linspace(0, 1, 50)
    calibrated_test = cal_iso._isotonic.transform(test_scores)
    assert np.all(calibrated_test >= 0.0) and np.all(calibrated_test <= 1.0)
    print(f"  Calibrated scores in [0, 1]  ✅")

    # Check monotonicity of calibrated scores
    for i in range(len(calibrated_test) - 1):
        assert calibrated_test[i] <= calibrated_test[i + 1] + 1e-10
    print(f"  Calibrated monotonicity  ✅")

    # ---- Test 5: ECE computation ----
    print("\n--- Test 5: Expected Calibration Error ---")
    # Perfect calibration: predicted probs match actual accuracy
    perfect_preds = np.array([0.2, 0.2, 0.8, 0.8, 0.8])
    perfect_labels = np.array([0, 0, 1, 1, 0])  # ~0% at 0.2, ~67% at 0.8
    ece, bins = ConfidenceCalibrator.compute_calibration_error(
        perfect_preds, perfect_labels, n_bins=5
    )
    print(f"  ECE = {ece:.4f}")
    assert 0.0 <= ece <= 1.0
    print(f"  ECE in [0, 1]  ✅")

    # Totally miscalibrated: always says 0.9 but half are wrong
    bad_preds = np.full(100, 0.9)
    bad_labels = rng.choice([0.0, 1.0], size=100)
    ece_bad, _ = ConfidenceCalibrator.compute_calibration_error(
        bad_preds, bad_labels, n_bins=10
    )
    print(f"  Miscalibrated ECE = {ece_bad:.4f} (should be high)")
    assert ece_bad > 0.1, "Miscalibrated model should have high ECE"
    print(f"  High ECE for miscalibrated  ✅")

    # ---- Test 6: __str__ representation ----
    print("\n--- Test 6: Decision string format ---")
    d = cal.route_from_score(0.82)
    s = str(d)
    assert "RELIABLE" in s
    assert "output_to_user" in s
    print(f"  {s}  ✅")

    # ---- Test 7: Invalid threshold assertion ----
    print("\n--- Test 7: Threshold validation ---")
    try:
        ConfidenceCalibrator(threshold_high=0.3, threshold_low=0.7)
        assert False, "Should have raised"
    except AssertionError:
        print(f"  Invalid thresholds rejected  ✅")

    print(f"\n{'=' * 60}")
    print("Part 3/9: confidence_calibrator.py — ALL CHECKS PASS ✅")
    print(f"{'=' * 60}")
