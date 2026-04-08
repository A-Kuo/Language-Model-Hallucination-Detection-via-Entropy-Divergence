"""
Tests for v2 Feature Engineer
================================

Strategy:
    Pure numpy — no torch, no model loading, no external data.
    Mirrors the structure of v1/test_attention_analyzer.py.

    - Section 1:  compute_entropy_features
    - Section 2:  compute_lookback_features
    - Section 3:  compute_frequency_features
    - Section 4:  compute_spectral_features
    - Section 5:  compute_kl_features
    - Section 6:  AttentionFeatureEngineer (unified extractor)
    - Section 7:  Numerical stability edge cases

Run:
    pytest v2/test_feature_engineer.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineer import (
    EPS,
    AttentionFeatureEngineer,
    FeatureConfig,
    compute_entropy_features,
    compute_frequency_features,
    compute_kl_features,
    compute_lookback_features,
    compute_spectral_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attn(L=4, H=4, T=8, rng=None, seed=42):
    """Return a valid softmax-normalised attention tensor (L, H, T, T)."""
    if rng is None:
        rng = np.random.default_rng(seed)
    logits = rng.standard_normal((L, H, T, T))
    logits -= logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


def _uniform_attn(L=4, H=4, T=8):
    """Return perfectly uniform attention (all 1/T)."""
    return np.ones((L, H, T, T)) / T


def _peaked_attn(L=4, H=4, T=8, pos=0):
    """Return peaked attention: all mass on position `pos`."""
    a = np.zeros((L, H, T, T))
    a[:, :, :, pos] = 1.0
    return a


# =========================================================================
# Section 1: compute_entropy_features
# =========================================================================

class TestComputeEntropyFeatures:
    """Tests for the entropy feature family."""

    def test_shape(self):
        """Output shape is (3,)."""
        a = _make_attn()
        feat = compute_entropy_features(a)
        assert feat.shape == (3,)

    def test_uniform_entropy_equals_log2T(self):
        """Uniform attention over T tokens → mean entropy = log₂(T)."""
        for T in [4, 8, 16, 32]:
            a = _uniform_attn(T=T)
            feat = compute_entropy_features(a)
            expected = np.log2(T)
            assert abs(feat[0] - expected) < 1e-5, \
                f"T={T}: mean_entropy={feat[0]:.6f}, expected {expected:.6f}"

    def test_peaked_entropy_near_zero(self):
        """All-on-one-token → entropy ≈ 0."""
        a = _peaked_attn()
        feat = compute_entropy_features(a)
        assert feat[0] < 0.01, f"mean_entropy={feat[0]}"
        assert feat[1] < 0.01, f"max_entropy={feat[1]}"

    def test_non_negative(self):
        """All 3 values ≥ 0 for random valid attention (50 trials)."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = _make_attn(rng=rng)
            feat = compute_entropy_features(a)
            assert np.all(feat >= -1e-10), f"Negative value: {feat}"

    def test_std_zero_for_uniform(self):
        """Identical distributions across all heads → std ≈ 0."""
        a = _uniform_attn()
        feat = compute_entropy_features(a)
        # All heads identical → entropy_std = 0
        assert feat[2] < 1e-6, f"std={feat[2]}"

    def test_max_geq_mean(self):
        """max_entropy ≥ mean_entropy always (100 trials)."""
        rng = np.random.default_rng(7)
        for _ in range(100):
            a = _make_attn(rng=rng)
            feat = compute_entropy_features(a)
            assert feat[1] >= feat[0] - 1e-10, \
                f"max={feat[1]} < mean={feat[0]}"


# =========================================================================
# Section 2: compute_lookback_features
# =========================================================================

class TestComputeLookbackFeatures:
    """Tests for the lookback ratio feature family."""

    def test_shape(self):
        """Output shape is (4,)."""
        a = _make_attn(T=16)
        feat = compute_lookback_features(a, context_length=8)
        assert feat.shape == (4,)

    def test_all_context(self):
        """Attention entirely on context tokens → ratio_mean ≈ 1."""
        T, context = 16, 8
        a = np.zeros((4, 4, T, T))
        a[:, :, :, :context] = 1.0 / context
        feat = compute_lookback_features(a, context_length=context)
        assert abs(feat[0] - 1.0) < 1e-6, f"ratio_mean={feat[0]}"

    def test_all_generated(self):
        """Attention entirely on generated tokens → ratio_mean ≈ 0."""
        T, context = 16, 8
        a = np.zeros((4, 4, T, T))
        a[:, :, :, context:] = 1.0 / (T - context)
        feat = compute_lookback_features(a, context_length=context)
        assert feat[0] < 1e-6, f"ratio_mean={feat[0]}"

    def test_ratio_in_unit_interval(self):
        """Ratios ∈ [0, 1] for 50 random inputs."""
        rng = np.random.default_rng(11)
        for _ in range(50):
            T = rng.integers(4, 20)
            ctx = int(T // 2)
            a = _make_attn(T=T, rng=rng)
            feat = compute_lookback_features(a, context_length=ctx)
            # ratio_mean and ratio_min should be in [0, 1]
            assert 0.0 - 1e-9 <= feat[0] <= 1.0 + 1e-9, \
                f"ratio_mean out of range: {feat[0]}"
            assert 0.0 - 1e-9 <= feat[1] <= 1.0 + 1e-9, \
                f"ratio_min out of range: {feat[1]}"

    def test_entropy_non_negative(self):
        """ratio_entropy ≥ 0 (binary entropy is always ≥ 0)."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            a = _make_attn(T=16, rng=rng)
            feat = compute_lookback_features(a, context_length=8)
            assert feat[3] >= -1e-10, f"ratio_entropy={feat[3]}"

    def test_half_split(self):
        """Equal context/generated attention → ratio_mean ≈ 0.5."""
        T, context = 16, 8
        # Uniform over all T tokens → exactly half on context
        a = _uniform_attn(T=T)
        feat = compute_lookback_features(a, context_length=context)
        assert abs(feat[0] - 0.5) < 1e-6, f"ratio_mean={feat[0]}"


# =========================================================================
# Section 3: compute_frequency_features
# =========================================================================

class TestComputeFrequencyFeatures:
    """Tests for the frequency-domain feature family."""

    def test_shape(self):
        """Output shape is (4,)."""
        a = _make_attn()
        feat = compute_frequency_features(a)
        assert feat.shape == (4,)

    def test_all_finite(self):
        """No NaN or Inf for 50 random inputs."""
        rng = np.random.default_rng(21)
        for _ in range(50):
            a = _make_attn(rng=rng)
            feat = compute_frequency_features(a)
            assert np.all(np.isfinite(feat)), f"Non-finite: {feat}"

    def test_high_freq_ratio_bounded(self):
        """high_freq_ratio_mean ∈ [0, 1]."""
        rng = np.random.default_rng(33)
        for _ in range(50):
            a = _make_attn(rng=rng)
            feat = compute_frequency_features(a)
            assert 0.0 - 1e-9 <= feat[0] <= 1.0 + 1e-9, \
                f"high_freq_ratio out of [0,1]: {feat[0]}"

    def test_dc_signal_low_high_freq(self):
        """
        Constant (uniform) attention → DC-dominated spectrum → low
        high-freq energy ratio.
        """
        # Uniform attention: signal is constant across key positions
        a = _uniform_attn(T=16)
        feat = compute_frequency_features(a)
        # All energy at DC (k=0); high-freq ratio should be very low
        assert feat[0] < 0.2, f"high_freq_ratio_mean={feat[0]}"

    def test_spectral_entropy_non_negative(self):
        """spectral_entropy_mean ≥ 0."""
        rng = np.random.default_rng(55)
        for _ in range(50):
            a = _make_attn(rng=rng)
            feat = compute_frequency_features(a)
            assert feat[3] >= -1e-10, f"spectral_entropy_mean={feat[3]}"


# =========================================================================
# Section 4: compute_spectral_features
# =========================================================================

class TestComputeSpectralFeatures:
    """Tests for the Laplacian spectral feature family."""

    def test_shape(self):
        """Output shape is (4,)."""
        a = _make_attn()
        feat = compute_spectral_features(a)
        assert feat.shape == (4,)

    def test_fiedler_non_negative(self):
        """Fiedler value (λ₂) ≥ 0 always."""
        rng = np.random.default_rng(77)
        for _ in range(50):
            a = _make_attn(rng=rng)
            feat = compute_spectral_features(a)
            assert feat[0] >= -1e-9, f"fiedler_mean={feat[0]}"

    def test_all_finite(self):
        """No NaN or Inf for 20 random inputs."""
        rng = np.random.default_rng(88)
        for _ in range(20):
            a = _make_attn(rng=rng)
            feat = compute_spectral_features(a)
            assert np.all(np.isfinite(feat)), f"Non-finite: {feat}"

    def test_uniform_graph_fiedler_positive(self):
        """
        Uniform attention creates a fully-connected symmetric graph.
        A fully-connected graph is well-connected → Fiedler > 0.
        """
        a = _uniform_attn(T=8)
        feat = compute_spectral_features(a)
        assert feat[0] > 0, f"fiedler_mean should be > 0 for uniform graph: {feat[0]}"


# =========================================================================
# Section 5: compute_kl_features
# =========================================================================

class TestComputeKLFeatures:
    """Tests for the cross-layer KL divergence feature family."""

    def test_shape(self):
        """Output shape is (3,)."""
        a = _make_attn()
        feat = compute_kl_features(a)
        assert feat.shape == (3,)

    def test_identical_layers_zero_kl(self):
        """All identical layers → KL = 0."""
        # All layers same uniform distribution
        a = _uniform_attn(L=6, T=8)
        feat = compute_kl_features(a)
        assert feat[0] < 1e-6, f"total_kl={feat[0]} (expected ~0)"
        assert feat[1] < 1e-6, f"max_kl={feat[1]} (expected ~0)"

    def test_non_negative(self):
        """KL values ≥ 0 for 100 random inputs (Gibbs' inequality)."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = _make_attn(rng=rng)
            feat = compute_kl_features(a)
            assert np.all(feat >= -1e-10), f"Negative KL: {feat}"

    def test_single_layer_returns_zeros(self):
        """L=1 → no consecutive pairs → [0.0, 0.0, 0.0]."""
        a = _make_attn(L=1)
        feat = compute_kl_features(a)
        assert np.allclose(feat, 0.0), f"Expected zeros: {feat}"

    def test_divergent_layers_high_kl(self):
        """
        Alternating peaked layers (mass on position 0 vs. position 7) →
        high cross-layer KL.
        """
        L, H, T = 6, 4, 8
        a = np.zeros((L, H, T, T))
        for l in range(L):
            pos = 0 if l % 2 == 0 else T - 1
            a[l, :, :, pos] = 1.0

        feat = compute_kl_features(a)
        assert feat[0] > 1.0, f"total_kl={feat[0]} (expected > 1)"


# =========================================================================
# Section 6: AttentionFeatureEngineer
# =========================================================================

class TestAttentionFeatureEngineer:
    """Tests for the unified multi-family feature extractor."""

    def _make_eng(self, context_length=8, **cfg_kwargs):
        config = FeatureConfig(**cfg_kwargs) if cfg_kwargs else None
        return AttentionFeatureEngineer(context_length=context_length, config=config)

    def test_feature_dim_all_families(self):
        """All families enabled → 18-dimensional output."""
        eng = self._make_eng()
        assert eng.feature_dim == 18, f"Expected 18, got {eng.feature_dim}"

    def test_feature_dim_no_context(self):
        """context_length=0 → lookback auto-disabled → 14 features."""
        eng = self._make_eng(context_length=0)
        assert eng.feature_dim == 14, f"Expected 14, got {eng.feature_dim}"

    def test_feature_names_length_matches_dim(self):
        """len(feature_names) == feature_dim."""
        eng = self._make_eng()
        assert len(eng.feature_names) == eng.feature_dim

    def test_extract_all_contains_combined(self):
        """extract_all always returns a 'combined' key."""
        eng = self._make_eng()
        a = _make_attn(T=16)
        feats = eng.extract_all(a)
        assert "combined" in feats

    def test_combined_shape(self):
        """combined.shape[0] == feature_dim."""
        eng = self._make_eng()
        a = _make_attn(T=16)
        feats = eng.extract_all(a)
        assert feats["combined"].shape[0] == eng.feature_dim

    def test_all_finite(self):
        """No NaN or Inf in combined vector for random attention."""
        rng = np.random.default_rng(13)
        eng = self._make_eng()
        for _ in range(20):
            a = _make_attn(T=16, rng=rng)
            feats = eng.extract_all(a)
            assert np.all(np.isfinite(feats["combined"])), \
                f"Non-finite: {feats['combined']}"

    def test_uniform_entropy_value(self):
        """Uniform attention → ent_mean ≈ log₂(T)."""
        T = 16
        eng = self._make_eng()
        a = _uniform_attn(T=T)
        feats = eng.extract_all(a)
        expected = np.log2(T)
        assert abs(feats["entropy"][0] - expected) < 1e-4, \
            f"ent_mean={feats['entropy'][0]:.4f}, expected {expected:.4f}"

    def test_uniform_zero_kl(self):
        """Uniform attention (all layers identical) → kl_total ≈ 0."""
        eng = self._make_eng()
        a = _uniform_attn(T=16)
        feats = eng.extract_all(a)
        assert feats["cross_layer_kl"][0] < 1e-6, \
            f"kl_total={feats['cross_layer_kl'][0]}"

    def test_minimal_config(self):
        """entropy + cross_layer_kl only → 6 features, no other family keys."""
        eng = AttentionFeatureEngineer(
            context_length=0,
            config=FeatureConfig(
                entropy=True,
                lookback=False,
                frequency=False,
                spectral=False,
                cross_layer_kl=True,
            ),
        )
        a = _make_attn(T=16)
        feats = eng.extract_all(a)
        assert "lookback" not in feats
        assert "frequency" not in feats
        assert "spectral" not in feats
        assert feats["combined"].shape[0] == 6  # 3 + 3

    def test_numpy_tuple_equivalence(self):
        """
        Passing a numpy array and an equivalent HF-style tuple give the same
        combined feature vector.
        """
        L, H, T = 4, 4, 8
        a_np = _make_attn(L=L, H=H, T=T)

        # Build HF-style tuple: each element is (1, H, T, T)
        a_tuple = tuple(a_np[l:l + 1] for l in range(L))

        eng = AttentionFeatureEngineer(context_length=4)
        v_np = eng.extract_vector(a_np)
        v_tp = eng.extract_vector(a_tuple)

        assert np.allclose(v_np, v_tp, atol=1e-6), \
            f"numpy vs tuple mismatch:\n{v_np}\n{v_tp}"


# =========================================================================
# Section 7: Numerical stability edge cases
# =========================================================================

class TestNumericalStability:
    """Edge cases that could produce NaN / Inf."""

    def test_near_zero_attention(self):
        """Attention values at EPS → all outputs finite."""
        L, H, T = 4, 4, 8
        a = np.full((L, H, T, T), EPS)
        a = a / a.sum(axis=-1, keepdims=True)
        for fn in [compute_entropy_features, compute_frequency_features,
                   compute_kl_features]:
            out = fn(a)
            assert np.all(np.isfinite(out)), f"{fn.__name__} non-finite: {out}"
        out = compute_lookback_features(a, context_length=4)
        assert np.all(np.isfinite(out))
        out = compute_spectral_features(a)
        assert np.all(np.isfinite(out))

    def test_large_T(self):
        """T=512 → no overflow in any family."""
        a = _uniform_attn(L=2, H=2, T=512)
        eng = AttentionFeatureEngineer(context_length=256)
        feats = eng.extract_all(a)
        assert np.all(np.isfinite(feats["combined"]))

    def test_single_layer(self):
        """L=1 → graceful output (KL returns zeros)."""
        a = _make_attn(L=1, T=8)
        kl = compute_kl_features(a)
        assert np.allclose(kl, 0.0)

        eng = AttentionFeatureEngineer(context_length=4)
        feats = eng.extract_all(a)
        assert np.all(np.isfinite(feats["combined"]))

    def test_single_head(self):
        """H=1 → correct shapes for all families."""
        a = _make_attn(L=4, H=1, T=8)
        assert compute_entropy_features(a).shape == (3,)
        assert compute_lookback_features(a, context_length=4).shape == (4,)
        assert compute_frequency_features(a).shape == (4,)
        assert compute_spectral_features(a).shape == (4,)
        assert compute_kl_features(a).shape == (3,)


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
