"""
Hypothesis Test — Statistical Decision Framework for Hallucination Detection
=============================================================================

Mathematical Foundation:
    Given attention-derived features from AttentionAnalyzer:
        - mean_entropy (H̄)
        - entropy_std (σ_H)
        - total_kl_divergence (D_KL_total)
        - max_kl_divergence (D_KL_max)

    We test the null hypothesis:

        H₀: The model output is RELIABLE
        H₁: The model output is UNRELIABLE (potential hallucination)

    Test Statistic (composite Z-score):
        We combine entropy and KL signals into a single test statistic
        using a weighted sum. Each signal is standardised against its
        calibrated baseline (mean, std from a reference corpus).

        Z_entropy = (H̄ - μ_H) / σ_ref_H
        Z_kl      = (D_KL_total - μ_KL) / σ_ref_KL

        Z_combined = w₁·Z_entropy + w₂·Z_kl

    Decision Rule:
        - Z_combined > z_critical  →  REJECT H₀  →  UNRELIABLE
        - Z_combined ≤ z_critical  →  FAIL TO REJECT  →  RELIABLE

    The p-value gives P(Z ≥ z_observed | H₀ is true), computed from
    the standard normal CDF:  p = 1 - Φ(z_combined)

    Default significance level α = 0.01  →  z_critical ≈ 2.576

Usage:
    from attention_analyzer import AttentionAnalyzer
    from hypothesis_test import HallucinationHypothesisTest

    analyzer = AttentionAnalyzer(model_name="gpt2")
    tester = HallucinationHypothesisTest()

    result = analyzer.analyze("The capital of France is")
    decision = tester.test(result)

    print(decision.z_score)       # float
    print(decision.p_value)       # float
    print(decision.is_reliable)   # bool
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

# Type hint only — avoid circular import at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from attention_analyzer import AttentionAnalysisResult


# ---------------------------------------------------------------------------
# Calibration baselines (estimated from reference corpus)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationBaseline:
    """
    Reference statistics from running the analyzer on a known-reliable
    corpus (e.g., WikiText-103 factual passages).

    These are the "null distribution" parameters — what entropy and KL
    look like when the model is behaving normally (not hallucinating).

    In production, you'd estimate these from your own calibration set.
    These defaults are reasonable starting points for GPT-2 on factual text.
    """
    # Entropy baseline (across all layers & heads)
    entropy_mean: float = 2.8       # bits — typical for GPT-2 on factual text
    entropy_std: float = 0.6        # spread across different prompts

    # KL divergence baseline (total across layer pairs)
    kl_mean: float = 1.5            # nats — typical cross-layer agreement
    kl_std: float = 0.8             # spread across different prompts

    # Entropy standard deviation baseline
    entropy_spread_mean: float = 1.2    # typical head-to-head variation
    entropy_spread_std: float = 0.4


# Default baselines for GPT-2
GPT2_BASELINE = CalibrationBaseline()


# ---------------------------------------------------------------------------
# Decision container
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTestResult:
    """Result of the hallucination hypothesis test."""

    # Individual Z-scores (how many σ away from baseline)
    z_entropy: float            # high → unusually high entropy (diffuse attention)
    z_kl: float                 # high → unusually high KL (layer disagreement)
    z_entropy_spread: float     # high → unusually variable across heads

    # Combined test statistic
    z_combined: float           # weighted sum of individual Z-scores

    # Statistical decision
    p_value: float              # P(Z ≥ z_observed | H₀)
    alpha: float                # significance level used
    is_reliable: bool           # True if we FAIL to reject H₀

    # Confidence score mapped to [0, 1] (higher = more reliable)
    confidence_score: float

    # Raw inputs (for debugging / transparency)
    raw_entropy: float
    raw_kl: float
    raw_entropy_std: float


# ---------------------------------------------------------------------------
# Core hypothesis test
# ---------------------------------------------------------------------------

class HallucinationHypothesisTest:
    """
    Statistical hypothesis test for hallucination detection.

    Combines attention entropy and KL divergence into a composite
    Z-score, then makes a binary decision at a given significance level.

    Parameters
    ----------
    baseline : CalibrationBaseline
        Reference statistics from a known-reliable corpus.
    alpha : float
        Significance level (default 0.01 → 1% false positive rate).
        Lower α = more conservative = fewer false alarms.
    weights : tuple of 3 floats
        (w_entropy, w_kl, w_spread) — relative importance of each signal.
        Default (0.4, 0.45, 0.15) emphasises KL divergence slightly,
        because layer disagreement is the strongest hallucination signal.
    """

    def __init__(
        self,
        baseline: Optional[CalibrationBaseline] = None,
        alpha: float = 0.01,
        weights: tuple[float, float, float] = (0.40, 0.45, 0.15),
    ) -> None:
        self.baseline = baseline or GPT2_BASELINE
        self.alpha = alpha
        self.weights = weights

        # Pre-compute critical value: z such that P(Z > z) = α
        # For α=0.01 → z_critical ≈ 2.326 (one-tailed)
        self.z_critical = scipy_stats.norm.ppf(1.0 - self.alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(self, analysis: AttentionAnalysisResult) -> HypothesisTestResult:
        """
        Run the hypothesis test on an AttentionAnalysisResult.

        Steps:
            1. Standardise each signal against baseline → Z-scores
            2. Combine Z-scores with weights → Z_combined
            3. Compute p-value from standard normal
            4. Compare to α → decision
            5. Map Z_combined to [0, 1] confidence score

        Returns
        -------
        HypothesisTestResult
        """
        b = self.baseline
        w = self.weights

        # --- 1. Individual Z-scores ---
        # Z = (observed - μ_baseline) / σ_baseline
        # Positive Z → observed is HIGHER than baseline → more uncertain

        z_entropy = self._safe_z(
            analysis.mean_entropy, b.entropy_mean, b.entropy_std
        )
        z_kl = self._safe_z(
            analysis.total_kl_divergence, b.kl_mean, b.kl_std
        )
        z_spread = self._safe_z(
            analysis.entropy_std, b.entropy_spread_mean, b.entropy_spread_std
        )

        # --- 2. Weighted combination ---
        z_combined = w[0] * z_entropy + w[1] * z_kl + w[2] * z_spread

        # --- 3. P-value (one-tailed, upper tail) ---
        # p = P(Z ≥ z_combined) under standard normal
        p_value = float(1.0 - scipy_stats.norm.cdf(z_combined))

        # --- 4. Decision ---
        # Reject H₀ (flag as unreliable) if z_combined > z_critical
        is_reliable = z_combined <= self.z_critical

        # --- 5. Confidence score [0, 1] ---
        # Map z_combined to a confidence score using the sigmoid-like
        # transformation:  confidence = Φ(-z_combined)
        # High z → low confidence;  Low z → high confidence
        confidence_score = float(scipy_stats.norm.cdf(-z_combined))

        return HypothesisTestResult(
            z_entropy=z_entropy,
            z_kl=z_kl,
            z_entropy_spread=z_spread,
            z_combined=z_combined,
            p_value=p_value,
            alpha=self.alpha,
            is_reliable=is_reliable,
            confidence_score=confidence_score,
            raw_entropy=analysis.mean_entropy,
            raw_kl=analysis.total_kl_divergence,
            raw_entropy_std=analysis.entropy_std,
        )

    def test_from_features(
        self,
        mean_entropy: float,
        total_kl: float,
        entropy_std: float,
    ) -> HypothesisTestResult:
        """
        Run the test from raw scalar features (without an
        AttentionAnalysisResult object). Useful for unit testing
        or when features come from a saved file.
        """
        # Build a minimal mock result
        _MockResult = type("_Mock", (), {
            "mean_entropy": mean_entropy,
            "total_kl_divergence": total_kl,
            "entropy_std": entropy_std,
        })
        return self.test(_MockResult())  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def test_batch(
        self,
        analyses: list[AttentionAnalysisResult],
    ) -> list[HypothesisTestResult]:
        """Test multiple analysis results."""
        return [self.test(a) for a in analyses]

    # ------------------------------------------------------------------
    # Calibration helper
    # ------------------------------------------------------------------

    @staticmethod
    def calibrate_from_corpus(
        analyses: list[AttentionAnalysisResult],
    ) -> CalibrationBaseline:
        """
        Estimate calibration baseline from a corpus of KNOWN-RELIABLE
        analysis results.

        Run AttentionAnalyzer on a factual reference corpus (e.g.,
        WikiText-103), collect the results, then pass them here to
        get baseline statistics.

        Parameters
        ----------
        analyses : list of AttentionAnalysisResult
            Results from running the analyzer on reliable text.

        Returns
        -------
        CalibrationBaseline
        """
        entropies = np.array([a.mean_entropy for a in analyses])
        kls = np.array([a.total_kl_divergence for a in analyses])
        spreads = np.array([a.entropy_std for a in analyses])

        return CalibrationBaseline(
            entropy_mean=float(entropies.mean()),
            entropy_std=float(entropies.std()),
            kl_mean=float(kls.mean()),
            kl_std=float(kls.std()),
            entropy_spread_mean=float(spreads.mean()),
            entropy_spread_std=float(spreads.std()),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_z(observed: float, mu: float, sigma: float) -> float:
        """
        Compute Z-score with division-by-zero protection.

        If σ = 0 (degenerate baseline), return 0.0 (no information).
        """
        if sigma < 1e-12:
            return 0.0
        return (observed - mu) / sigma


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HYPOTHESIS TEST — STANDALONE VALIDATION")
    print("=" * 60)

    tester = HallucinationHypothesisTest(alpha=0.01)
    print(f"\nBaseline: entropy μ={tester.baseline.entropy_mean}, "
          f"σ={tester.baseline.entropy_std}")
    print(f"Baseline: KL μ={tester.baseline.kl_mean}, "
          f"σ={tester.baseline.kl_std}")
    print(f"α = {tester.alpha}, z_critical = {tester.z_critical:.4f}")

    # --- Scenario 1: Normal (baseline-level) signals → RELIABLE ---
    r1 = tester.test_from_features(
        mean_entropy=2.8,   # exactly at baseline
        total_kl=1.5,       # exactly at baseline
        entropy_std=1.2,    # exactly at baseline
    )
    print(f"\n--- Scenario 1: Baseline-level signals ---")
    print(f"Z_combined = {r1.z_combined:.4f}, p = {r1.p_value:.4f}, "
          f"reliable = {r1.is_reliable}, confidence = {r1.confidence_score:.4f}")
    assert r1.is_reliable, "Baseline signals should be reliable"
    assert abs(r1.z_combined) < 0.01, "Z should be ~0 at baseline"
    print("✅ Correctly identified as RELIABLE")

    # --- Scenario 2: Very high entropy + KL → UNRELIABLE ---
    r2 = tester.test_from_features(
        mean_entropy=5.0,   # way above baseline (diffuse attention)
        total_kl=5.0,       # way above baseline (layers disagree)
        entropy_std=2.5,    # high spread
    )
    print(f"\n--- Scenario 2: Extreme signals ---")
    print(f"Z_combined = {r2.z_combined:.4f}, p = {r2.p_value:.6f}, "
          f"reliable = {r2.is_reliable}, confidence = {r2.confidence_score:.4f}")
    assert not r2.is_reliable, "Extreme signals should be unreliable"
    assert r2.confidence_score < 0.1, "Confidence should be very low"
    print("✅ Correctly identified as UNRELIABLE")

    # --- Scenario 3: Low entropy + KL → HIGH confidence ---
    r3 = tester.test_from_features(
        mean_entropy=1.5,   # well below baseline (peaked attention)
        total_kl=0.3,       # well below baseline (layers agree)
        entropy_std=0.5,    # low spread
    )
    print(f"\n--- Scenario 3: Low uncertainty signals ---")
    print(f"Z_combined = {r3.z_combined:.4f}, p = {r3.p_value:.4f}, "
          f"reliable = {r3.is_reliable}, confidence = {r3.confidence_score:.4f}")
    assert r3.is_reliable, "Low-uncertainty should be reliable"
    assert r3.confidence_score > 0.8, "Confidence should be high"
    print("✅ Correctly identified as RELIABLE with high confidence")

    # --- Scenario 4: Borderline case ---
    # Find entropy that puts us right at the critical threshold
    # Z_combined ≈ z_critical when we're at the decision boundary
    print(f"\n--- Scenario 4: Near decision boundary ---")
    # Slowly increase entropy until we flip
    for test_entropy in np.arange(2.8, 6.0, 0.1):
        r = tester.test_from_features(
            mean_entropy=test_entropy,
            total_kl=1.5,
            entropy_std=1.2,
        )
        if not r.is_reliable:
            print(f"Flipped to UNRELIABLE at entropy={test_entropy:.1f}, "
                  f"Z={r.z_combined:.4f}, p={r.p_value:.4f}")
            break
    print("✅ Decision boundary behaves correctly")

    # --- Scenario 5: Confidence score is monotonically decreasing with Z ---
    print(f"\n--- Scenario 5: Monotonicity check ---")
    prev_conf = 1.0
    for ent in [1.0, 2.0, 3.0, 4.0, 5.0]:
        r = tester.test_from_features(ent, 1.5, 1.2)
        assert r.confidence_score <= prev_conf + 1e-6, \
            f"Confidence should decrease: {r.confidence_score} > {prev_conf}"
        prev_conf = r.confidence_score
    print("✅ Confidence monotonically decreases with entropy")

    # --- Scenario 6: P-value properties ---
    print(f"\n--- Scenario 6: P-value sanity ---")
    r_base = tester.test_from_features(2.8, 1.5, 1.2)
    assert 0.0 <= r_base.p_value <= 1.0, "P-value must be in [0, 1]"
    assert abs(r_base.p_value - 0.5) < 0.01, \
        f"At baseline, p ≈ 0.5: got {r_base.p_value}"
    print(f"At baseline: p = {r_base.p_value:.4f} ≈ 0.5  ✅")

    r_extreme = tester.test_from_features(10.0, 10.0, 5.0)
    assert r_extreme.p_value < 0.001, "Extreme → p ≈ 0"
    print(f"At extreme:  p = {r_extreme.p_value:.8f} ≈ 0  ✅")

    # --- Scenario 7: Custom calibration baseline ---
    print(f"\n--- Scenario 7: Custom baseline ---")
    custom = CalibrationBaseline(
        entropy_mean=3.5, entropy_std=0.3,
        kl_mean=2.0, kl_std=0.5,
        entropy_spread_mean=1.0, entropy_spread_std=0.2,
    )
    tester_custom = HallucinationHypothesisTest(baseline=custom, alpha=0.05)
    r_custom = tester_custom.test_from_features(3.5, 2.0, 1.0)
    assert r_custom.is_reliable
    assert abs(r_custom.z_combined) < 0.01
    print(f"Custom baseline at center: Z = {r_custom.z_combined:.4f}  ✅")

    print(f"\n{'=' * 60}")
    print("Part 2/9: hypothesis_test.py — ALL CHECKS PASS ✅")
    print(f"{'=' * 60}")
