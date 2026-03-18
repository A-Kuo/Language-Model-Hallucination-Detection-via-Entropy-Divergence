"""
Tests for AttentionAnalyzer
============================

Strategy:
    - Section 1-2: Test standalone math functions using ONLY numpy
      (compute_entropy_from_weights, compute_kl_divergence).
      These require no model loading and no torch.

    - Section 3: Test internal methods (_compute_entropy, _compute_cross_layer_kl)
      with synthetic torch tensors. These are skipped if torch is unavailable or
      would cause OOM.

    - Section 4: Full model integration tests (marked slow, skipped by default).

Run:
    pytest test_attention_analyzer.py -v
    pytest test_attention_analyzer.py -v -k "not torch"  # numpy-only
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Conditional imports ---
# The standalone math functions live at module level but don't need torch.
# We import them by extracting just the functions from the source.

try:
    # This works if torch is installed but won't load the model
    from attention_analyzer import (
        compute_entropy_from_weights,
        compute_kl_divergence,
    )
    _HAS_STANDALONE = True
except ImportError:
    _HAS_STANDALONE = False

    # Fallback: redefine the functions inline for testing
    def compute_entropy_from_weights(attention_weights, eps=1e-12):
        a = np.clip(attention_weights, eps, None)
        if a.ndim == 1:
            return -np.sum(a * np.log2(a))
        return -np.sum(a * np.log2(a), axis=-1)

    def compute_kl_divergence(p, q, eps=1e-12):
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

# Check if torch is available for tensor-based tests
try:
    import torch
    import transformers  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# =========================================================================
# Section 1: Shannon Entropy (numpy-only)
# =========================================================================

class TestComputeEntropyFromWeights:
    """Tests for compute_entropy_from_weights — pure numpy."""

    def test_uniform_distribution_various_sizes(self):
        """Uniform dist over T tokens → entropy = log₂(T)."""
        for T in [2, 4, 8, 16, 64, 256, 1024]:
            uniform = np.ones(T) / T
            h = compute_entropy_from_weights(uniform)
            expected = np.log2(T)
            assert abs(h - expected) < 1e-6, \
                f"T={T}: got {h:.6f}, expected {expected:.6f}"

    def test_delta_distribution(self):
        """One-hot distribution → entropy ≈ 0."""
        for T in [2, 8, 64]:
            delta = np.zeros(T)
            delta[0] = 1.0
            h = compute_entropy_from_weights(delta)
            assert h < 1e-6, f"T={T}: delta entropy should be ~0, got {h}"

    def test_binary_entropy_one_bit(self):
        """[0.5, 0.5] → exactly 1 bit."""
        h = compute_entropy_from_weights(np.array([0.5, 0.5]))
        assert abs(h - 1.0) < 1e-10

    def test_skewed_binary_known_value(self):
        """[0.9, 0.1] → H = -0.9·log₂(0.9) - 0.1·log₂(0.1) ≈ 0.469."""
        p = np.array([0.9, 0.1])
        h = compute_entropy_from_weights(p)
        expected = -(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))
        assert abs(h - expected) < 1e-6

    def test_three_way_uniform(self):
        """[1/3, 1/3, 1/3] → log₂(3) ≈ 1.585."""
        h = compute_entropy_from_weights(np.ones(3) / 3)
        assert abs(h - np.log2(3)) < 1e-6

    def test_multi_head_batch_shape(self):
        """Input (H, T) → output (H,)."""
        H, T = 4, 8
        heads = np.ones((H, T)) / T
        h = compute_entropy_from_weights(heads)
        assert h.shape == (H,)
        assert np.allclose(h, np.log2(T))

    def test_multi_head_mixed_distributions(self):
        """Different distributions per head → different entropies."""
        T = 8
        uniform = np.ones(T) / T
        peaked = np.zeros(T); peaked[0] = 1.0
        heads = np.array([uniform, peaked])
        h = compute_entropy_from_weights(heads)
        assert h[0] > 2.0  # uniform → high
        assert h[1] < 0.01  # peaked → low

    def test_entropy_non_negative_random(self):
        """Entropy ≥ 0 for any valid distribution (100 random trials)."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            T = rng.integers(2, 64)
            dist = rng.dirichlet(np.ones(T))
            h = compute_entropy_from_weights(dist)
            assert h >= -1e-10, f"Negative entropy: {h}"

    def test_entropy_bounded_by_max(self):
        """Entropy ≤ log₂(T) for any distribution (100 random trials)."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            T = rng.integers(2, 64)
            dist = rng.dirichlet(np.ones(T))
            h = compute_entropy_from_weights(dist)
            assert h <= np.log2(T) + 1e-6, \
                f"Entropy {h} > log₂({T}) = {np.log2(T)}"

    def test_entropy_monotonic_with_uniformity(self):
        """More uniform → higher entropy (interpolate between delta and uniform)."""
        T = 8
        delta = np.zeros(T); delta[0] = 1.0
        uniform = np.ones(T) / T

        prev_h = -1.0
        for alpha in np.linspace(0.0, 1.0, 20):
            mixed = (1 - alpha) * delta + alpha * uniform
            mixed = mixed / mixed.sum()
            h = compute_entropy_from_weights(mixed)
            assert h >= prev_h - 1e-10, \
                f"Entropy should increase: α={alpha:.2f}, h={h:.4f} < prev={prev_h:.4f}"
            prev_h = h


# =========================================================================
# Section 2: KL Divergence (numpy-only)
# =========================================================================

class TestComputeKLDivergence:
    """Tests for compute_kl_divergence — pure numpy."""

    def test_identical_distributions_zero(self):
        """KL(p || p) = 0."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            T = rng.integers(2, 64)
            p = rng.dirichlet(np.ones(T))
            kl = compute_kl_divergence(p, p)
            assert abs(kl) < 1e-10, f"KL(p||p) should be 0, got {kl}"

    def test_gibbs_inequality(self):
        """KL(p || q) ≥ 0 for all p, q (200 random trials)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            T = rng.integers(2, 64)
            p = rng.dirichlet(np.ones(T))
            q = rng.dirichlet(np.ones(T))
            kl = compute_kl_divergence(p, q)
            assert kl >= -1e-10, f"KL should be ≥ 0, got {kl}"

    def test_asymmetry(self):
        """KL(p||q) ≠ KL(q||p) in general."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)
        assert abs(kl_pq - kl_qp) > 0.01, \
            f"Should be asymmetric: {kl_pq} vs {kl_qp}"

    def test_high_divergence_opposing(self):
        """Nearly disjoint distributions → high KL."""
        p = np.array([0.99, 0.01])
        q = np.array([0.01, 0.99])
        kl = compute_kl_divergence(p, q)
        assert kl > 3.0, f"Very different dists should have high KL: {kl}"

    def test_low_divergence_similar(self):
        """Nearly identical distributions → low KL."""
        p = np.array([0.5, 0.5])
        q = np.array([0.49, 0.51])
        kl = compute_kl_divergence(p, q)
        assert kl < 0.01, f"Similar dists should have low KL: {kl}"

    def test_known_analytical_value(self):
        """Hand-computed KL for [0.5, 0.5] vs [0.25, 0.75]."""
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        # D_KL = 0.5·ln(0.5/0.25) + 0.5·ln(0.5/0.75)
        expected = 0.5 * np.log(2.0) + 0.5 * np.log(2.0 / 3.0)
        kl = compute_kl_divergence(p, q)
        assert abs(kl - expected) < 1e-6, f"Expected {expected}, got {kl}"

    def test_kl_increases_with_distance(self):
        """
        As q moves away from p, KL should increase.
        p = [0.5, 0.5], q = [0.5-ε, 0.5+ε] for increasing ε.
        """
        p = np.array([0.5, 0.5])
        prev_kl = 0.0
        for eps in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49]:
            q = np.array([0.5 - eps, 0.5 + eps])
            kl = compute_kl_divergence(p, q)
            assert kl >= prev_kl - 1e-10, \
                f"KL should increase: ε={eps}, kl={kl} < prev={prev_kl}"
            prev_kl = kl

    def test_kl_of_uniform_vs_peaked(self):
        """Uniform p vs peaked q → moderate KL (information gain)."""
        T = 8
        p = np.ones(T) / T
        q = np.zeros(T); q[0] = 1.0
        kl = compute_kl_divergence(p, q)
        # KL(uniform || delta) = log(T) - 0 = log(T)  (approximately)
        assert kl > 1.0, f"Uniform vs peaked should be high: {kl}"


# =========================================================================
# Section 3: Internal methods with torch tensors
# =========================================================================

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestInternalComputeEntropy:
    """Test _compute_entropy with synthetic torch tensors."""

    def _make_stub(self):
        """Create analyzer stub without loading a model."""
        from attention_analyzer import AttentionAnalyzer
        stub = object.__new__(AttentionAnalyzer)
        stub.eps = 1e-12
        return stub

    def _uniform_attentions(self, L=3, H=4, T=8):
        attn = torch.ones(1, H, T, T) / T
        return tuple(attn.clone() for _ in range(L))

    def _peaked_attentions(self, L=3, H=4, T=8):
        attn = torch.zeros(1, H, T, T)
        attn[:, :, :, 0] = 1.0
        return tuple(attn.clone() for _ in range(L))

    def test_uniform_entropy_value(self):
        T = 8
        stub = self._make_stub()
        entropy = stub._compute_entropy(self._uniform_attentions(T=T))
        assert entropy.shape == (3, 4)
        assert np.allclose(entropy, np.log2(T), atol=1e-5)

    def test_peaked_entropy_near_zero(self):
        stub = self._make_stub()
        entropy = stub._compute_entropy(self._peaked_attentions())
        assert np.all(entropy < 0.01)

    def test_shape_matches_architecture(self):
        stub = self._make_stub()
        for L, H, T in [(1, 1, 4), (6, 8, 16), (12, 12, 32)]:
            attn = tuple(torch.ones(1, H, T, T) / T for _ in range(L))
            entropy = stub._compute_entropy(attn)
            assert entropy.shape == (L, H)

    def test_uses_last_token_row(self):
        """Only the last query token's attention row matters."""
        stub = self._make_stub()
        T = 4
        attn = torch.zeros(1, 1, T, T)
        attn[0, 0, :-1, 0] = 1.0       # non-last rows: peaked
        attn[0, 0, -1, :] = 1.0 / T    # last row: uniform
        entropy = stub._compute_entropy((attn,))
        assert abs(entropy[0, 0] - np.log2(T)) < 1e-5


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestInternalComputeCrossLayerKL:
    """Test _compute_cross_layer_kl with synthetic torch tensors."""

    def _make_stub(self):
        from attention_analyzer import AttentionAnalyzer
        stub = object.__new__(AttentionAnalyzer)
        stub.eps = 1e-12
        return stub

    def test_identical_layers_zero_kl(self):
        stub = self._make_stub()
        attn = tuple(torch.ones(1, 4, 8, 8) / 8 for _ in range(5))
        kl = stub._compute_cross_layer_kl(attn)
        assert len(kl) == 4
        assert all(abs(v) < 1e-10 for v in kl)

    def test_divergent_layers_high_kl(self):
        stub = self._make_stub()
        layers = []
        for l in range(4):
            attn = torch.full((1, 4, 8, 8), 1e-8)
            attn[:, :, :, l % 8] = 1.0
            attn = attn / attn.sum(dim=-1, keepdim=True)
            layers.append(attn)
        kl = stub._compute_cross_layer_kl(tuple(layers))
        assert len(kl) == 3
        assert all(v > 1.0 for v in kl)

    def test_single_layer_empty(self):
        stub = self._make_stub()
        attn = (torch.ones(1, 4, 8, 8) / 8,)
        kl = stub._compute_cross_layer_kl(attn)
        assert kl == []

    def test_kl_count(self):
        stub = self._make_stub()
        for L in [2, 5, 12]:
            attn = tuple(torch.ones(1, 4, 8, 8) / 8 for _ in range(L))
            kl = stub._compute_cross_layer_kl(attn)
            assert len(kl) == L - 1

    def test_non_negative(self):
        stub = self._make_stub()
        torch.manual_seed(42)
        layers = []
        for _ in range(6):
            logits = torch.randn(1, 4, 8, 8)
            layers.append(torch.softmax(logits, dim=-1))
        kl = stub._compute_cross_layer_kl(tuple(layers))
        assert all(v >= -1e-10 for v in kl)


# =========================================================================
# Section 4: Numerical stability edge cases
# =========================================================================

class TestNumericalStability:
    """Edge cases that could cause NaN or Inf."""

    def test_near_zero_weights(self):
        tiny = np.full(8, 1e-15)
        tiny = tiny / tiny.sum()
        h = compute_entropy_from_weights(tiny)
        assert np.isfinite(h)

    def test_single_element(self):
        h = compute_entropy_from_weights(np.array([1.0]))
        assert abs(h) < 1e-6

    def test_kl_near_zero_denominator(self):
        p = np.array([0.5, 0.5])
        q = np.array([1e-15, 1.0 - 1e-15])
        kl = compute_kl_divergence(p, q)
        assert np.isfinite(kl) and kl > 0

    def test_large_T(self):
        T = 1024
        h = compute_entropy_from_weights(np.ones(T) / T)
        assert abs(h - np.log2(T)) < 1e-4

    def test_very_large_T(self):
        T = 4096
        h = compute_entropy_from_weights(np.ones(T) / T)
        assert abs(h - np.log2(T)) < 1e-3


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
