"""
Multi-Family Attention Feature Engineering
=============================================

Unifies five research-grade feature families for hallucination detection,
each operating on raw attention tensors (L, H, T, T):

    1. ENTROPY — Shannon entropy of attention distributions (our v1)
    2. LOOKBACK RATIO — Context vs. generation attention (Chuang et al., EMNLP 2024)
    3. FREQUENCY DOMAIN — DFT high-frequency energy (Qi et al., 2026)
    4. SPECTRAL — Laplacian eigenvalues of attention-as-graph (Barbero et al.)
    5. CROSS-LAYER KL — Divergence between consecutive layers (our v1)

Each family extracts a fixed-size feature vector. Concatenated, they form
the full feature vector for a lightweight classifier.

Key insight: No single feature family captures all hallucination signals.
Entropy catches diffuse attention but misses context-grounding failures.
Lookback ratio catches grounding failures but misses cross-layer instability.
Frequency features catch temporal instability invisible to static metrics.

Important limitation (Batson et al., 2025 — Anthropic circuit tracing):
    All five families measure downstream ATTENTION PATTERNS, not the causal
    circuit (known-entity detector misfiring → refusal suppression). This
    means confident hallucination (motivated reasoning with focused, low-entropy
    attention) will evade detection. Attention-based features are a first-pass
    filter; circuit-level activation probing is needed for the remaining cases.

Usage:
    engineer = AttentionFeatureEngineer(context_length=42)
    features = engineer.extract_all(attention_tuple)
    # features is a dict mapping family → np.ndarray

Mathematical notation:
    L = num_layers, H = num_heads, T = sequence_length
    A^{l,h} ∈ R^{T×T} = attention matrix for layer l, head h
    a^{l,h}_{t} = A^{l,h}[t, :] = attention distribution for query token t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


EPS = 1e-12  # numerical floor for log operations


# =========================================================================
# Feature Family 1: ENTROPY
# =========================================================================

def compute_entropy_features(
    attentions: np.ndarray,
    query_idx: int = -1,
) -> np.ndarray:
    """
    Shannon entropy of the attention distribution for a query token.

    H(a^{l,h}_t) = -Σ_i a^{l,h}_{t,i} · log₂(a^{l,h}_{t,i})

    Parameters
    ----------
    attentions : np.ndarray, shape (L, H, T, T)
        Attention weights for all layers and heads.
    query_idx : int
        Which query position to analyze (-1 = last token).

    Returns
    -------
    np.ndarray, shape (3,)
        [mean_entropy, max_entropy, entropy_std] across all heads.
    """
    L, H, T, _ = attentions.shape
    a = np.clip(attentions[:, :, query_idx, :], EPS, None)  # (L, H, T)
    per_head = -np.sum(a * np.log2(a), axis=-1)  # (L, H)

    return np.array([
        per_head.mean(),
        per_head.max(),
        per_head.std(),
    ])


# =========================================================================
# Feature Family 2: LOOKBACK RATIO (Chuang et al., EMNLP 2024)
# =========================================================================

def compute_lookback_features(
    attentions: np.ndarray,
    context_length: int,
    query_idx: int = -1,
) -> np.ndarray:
    """
    Lookback ratio: fraction of attention on context vs. generated tokens.

    For each head h at layer l:
        r^{l,h} = Σ_{i ∈ context} a^{l,h}_{t,i} / Σ_{i ∈ all} a^{l,h}_{t,i}

    High lookback ratio → model grounds in context → likely reliable.
    Low lookback ratio → model attends to own generations → hallucination risk.

    Per Lookback Lens: "a linear classifier based on these lookback ratio
    features is as effective as a richer detector that utilizes the entire
    hidden states of an LLM."

    Parameters
    ----------
    attentions : np.ndarray, shape (L, H, T, T)
    context_length : int
        Number of tokens in the original context/prompt.
    query_idx : int
        Which query position to analyze.

    Returns
    -------
    np.ndarray, shape (4,)
        [mean_ratio, min_ratio, ratio_std, ratio_entropy]
    """
    L, H, T, _ = attentions.shape
    # Attention from query token to all positions
    a = attentions[:, :, query_idx, :]  # (L, H, T)

    # Attention mass on context tokens vs. all tokens
    context_attn = a[:, :, :context_length].sum(axis=-1)  # (L, H)
    total_attn = a.sum(axis=-1)  # (L, H), should be ~1.0

    ratios = context_attn / np.clip(total_attn, EPS, None)  # (L, H)

    # Lookback entropy: how dispersed is the ratio across heads?
    # High entropy = inconsistent grounding across heads
    flat_ratios = ratios.flatten()
    flat_ratios = np.clip(flat_ratios, EPS, 1.0 - EPS)
    ratio_entropy = -np.mean(
        flat_ratios * np.log2(flat_ratios) +
        (1 - flat_ratios) * np.log2(1 - flat_ratios)
    )

    return np.array([
        ratios.mean(),
        ratios.min(),
        ratios.std(),
        ratio_entropy,
    ])


# =========================================================================
# Feature Family 3: FREQUENCY DOMAIN (Qi et al., 2026)
# =========================================================================

def compute_frequency_features(
    attentions: np.ndarray,
    query_idx: int = -1,
    top_k_freq: int = 5,
) -> np.ndarray:
    """
    Frequency-domain analysis of attention via Discrete Fourier Transform.

    Core insight from Qi et al.: "hallucinated tokens are associated with
    high-frequency attention energy, reflecting fragmented and unstable
    grounding behavior."

    We treat each head's attention vector a^{l,h}_t as a discrete signal
    over the key positions, then compute its DFT. High-frequency energy
    indicates rapid, unstable shifts in attention — a hallucination signal.

    Mathematically:
        X_k = Σ_{n=0}^{T-1} a_n · e^{-j2πkn/T}    (DFT)
        E_high = Σ_{k=T/2}^{T-1} |X_k|²             (high-freq energy)
        E_total = Σ_{k=0}^{T-1} |X_k|²              (Parseval's)
        ratio = E_high / E_total                      (instability metric)

    Parameters
    ----------
    attentions : np.ndarray, shape (L, H, T, T)
    query_idx : int
    top_k_freq : int
        Number of top frequency magnitudes to include as features.

    Returns
    -------
    np.ndarray, shape (4,)
        [high_freq_ratio_mean, high_freq_ratio_max,
         spectral_centroid_mean, spectral_entropy_mean]
    """
    L, H, T, _ = attentions.shape
    a = attentions[:, :, query_idx, :]  # (L, H, T)

    high_ratios = np.zeros((L, H))
    centroids = np.zeros((L, H))
    spectral_entropies = np.zeros((L, H))

    for l in range(L):
        for h in range(H):
            signal = a[l, h]  # (T,)

            # DFT
            spectrum = np.fft.rfft(signal)
            magnitudes = np.abs(spectrum) ** 2
            total_energy = magnitudes.sum() + EPS

            # High-frequency ratio: energy in upper half of spectrum
            midpoint = len(magnitudes) // 2
            high_energy = magnitudes[midpoint:].sum()
            high_ratios[l, h] = high_energy / total_energy

            # Spectral centroid: "center of mass" of the spectrum
            freqs = np.arange(len(magnitudes))
            centroids[l, h] = (freqs * magnitudes).sum() / total_energy

            # Spectral entropy: how spread the energy is across frequencies
            p = magnitudes / total_energy
            p = np.clip(p, EPS, None)
            spectral_entropies[l, h] = -np.sum(p * np.log2(p))

    return np.array([
        high_ratios.mean(),
        high_ratios.max(),
        centroids.mean(),
        spectral_entropies.mean(),
    ])


# =========================================================================
# Feature Family 4: SPECTRAL / LAPLACIAN EIGENVALUES
# =========================================================================

def compute_spectral_features(
    attentions: np.ndarray,
    query_idx: int = -1,
    top_k_eigenvalues: int = 5,
) -> np.ndarray:
    """
    Spectral features from the attention-as-graph Laplacian.

    Following recent work on LapEigvals: "eigenvalues of a Laplacian matrix
    derived from attention maps serve as good predictors of hallucinations."

    Treating each layer's mean-over-heads attention matrix as a weighted
    adjacency graph:
        W = mean_h(A^{l,h})               (T × T adjacency)
        D = diag(W · 1)                   (degree matrix)
        L = D - W                          (unnormalised Laplacian)
        λ_1 ≤ λ_2 ≤ ... ≤ λ_T            (Laplacian spectrum)

    λ₂ (Fiedler value / algebraic connectivity) indicates how well-connected
    the attention graph is. Low λ₂ → bottlenecks → fragmented attention.

    Parameters
    ----------
    attentions : np.ndarray, shape (L, H, T, T)
    top_k_eigenvalues : int
        Number of smallest non-zero eigenvalues to return.

    Returns
    -------
    np.ndarray, shape (4,)
        [fiedler_mean, fiedler_std, spectral_gap_mean, laplacian_energy_mean]
    """
    L, H, T, _ = attentions.shape

    fiedler_values = []
    spectral_gaps = []
    laplacian_energies = []

    for l in range(L):
        # Average attention across heads for this layer
        W = attentions[l].mean(axis=0)  # (T, T)

        # Symmetrise (attention is not symmetric)
        W = (W + W.T) / 2

        # Degree matrix
        degrees = W.sum(axis=1)
        D = np.diag(degrees)

        # Unnormalised graph Laplacian
        Lap = D - W

        # Eigenvalues (sorted ascending)
        eigenvalues = np.linalg.eigvalsh(Lap)
        eigenvalues = np.sort(np.abs(eigenvalues))

        # Fiedler value (second-smallest eigenvalue)
        fiedler = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        fiedler_values.append(fiedler)

        # Spectral gap: λ₂ / λ_max
        lam_max = eigenvalues[-1] if eigenvalues[-1] > EPS else 1.0
        spectral_gaps.append(fiedler / lam_max)

        # Laplacian energy: sum of |λ_i - mean_degree|
        mean_deg = degrees.mean()
        laplacian_energies.append(np.abs(eigenvalues - mean_deg).sum())

    return np.array([
        np.mean(fiedler_values),
        np.std(fiedler_values),
        np.mean(spectral_gaps),
        np.mean(laplacian_energies),
    ])


# =========================================================================
# Feature Family 5: CROSS-LAYER KL DIVERGENCE
# =========================================================================

def compute_kl_features(
    attentions: np.ndarray,
    query_idx: int = -1,
) -> np.ndarray:
    """
    KL divergence between consecutive layers' attention distributions.

    D_KL(a^{l} || a^{l+1}) = Σ_i ā^l_i · log(ā^l_i / ā^{l+1}_i)

    where ā^l = mean_h(a^{l,h}_t) is the head-averaged attention at layer l.

    High cross-layer KL → layers disagree → internal inconsistency.

    Parameters
    ----------
    attentions : np.ndarray, shape (L, H, T, T)

    Returns
    -------
    np.ndarray, shape (3,)
        [total_kl, max_kl, kl_std]
    """
    L, H, T, _ = attentions.shape

    if L < 2:
        return np.array([0.0, 0.0, 0.0])

    # Head-averaged attention for the query token
    a = attentions[:, :, query_idx, :].mean(axis=1)  # (L, T)
    a = np.clip(a, EPS, None)

    kls = []
    for l in range(L - 1):
        p = a[l] / a[l].sum()
        q = a[l + 1] / a[l + 1].sum()
        kl = float(np.sum(p * np.log(p / q)))
        kls.append(max(0.0, kl))  # clamp numerical noise

    kls = np.array(kls) if kls else np.array([0.0])

    return np.array([
        kls.sum(),
        kls.max(),
        kls.std(),
    ])


# =========================================================================
# Unified Feature Engineer
# =========================================================================

@dataclass
class FeatureConfig:
    """Configuration for which feature families to extract."""
    entropy: bool = True
    lookback: bool = True
    frequency: bool = True
    spectral: bool = True
    cross_layer_kl: bool = True
    query_idx: int = -1  # -1 = last token (the generated token)


class AttentionFeatureEngineer:
    """
    Unified multi-family feature extraction from attention tensors.

    Combines insights from:
        - Lookback Lens (Chuang et al., EMNLP 2024)
        - Frequency-Aware Attention (Qi et al., 2026)
        - Laplacian Eigenvalue features (Barbero et al., 2025)
        - Our v1 entropy + KL divergence
    """

    # Feature sizes per family
    FEATURE_SIZES = {
        "entropy": 3,       # mean, max, std
        "lookback": 4,      # mean_ratio, min_ratio, ratio_std, ratio_entropy
        "frequency": 4,     # high_freq_ratio_mean/max, centroid, spectral_entropy
        "spectral": 4,      # fiedler_mean/std, spectral_gap, laplacian_energy
        "cross_layer_kl": 3, # total, max, std
    }

    def __init__(
        self,
        context_length: int = 0,
        config: Optional[FeatureConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        context_length : int
            Number of tokens in the input context (prompt). Required for
            lookback ratio features. Set to 0 to skip lookback.
        config : FeatureConfig
            Which families to extract.
        """
        self.context_length = context_length
        self.config = config or FeatureConfig()

        if context_length == 0:
            self.config.lookback = False

    @property
    def feature_dim(self) -> int:
        """Total dimension of the concatenated feature vector."""
        dim = 0
        for family, size in self.FEATURE_SIZES.items():
            if getattr(self.config, family, False):
                dim += size
        return dim

    @property
    def feature_names(self) -> List[str]:
        """Human-readable names for each feature dimension."""
        names = []
        family_labels = {
            "entropy": ["ent_mean", "ent_max", "ent_std"],
            "lookback": ["lb_ratio_mean", "lb_ratio_min", "lb_ratio_std", "lb_ratio_entropy"],
            "frequency": ["freq_high_mean", "freq_high_max", "freq_centroid", "freq_spectral_ent"],
            "spectral": ["spec_fiedler_mean", "spec_fiedler_std", "spec_gap", "spec_lap_energy"],
            "cross_layer_kl": ["kl_total", "kl_max", "kl_std"],
        }
        for family in self.FEATURE_SIZES:
            if getattr(self.config, family, False):
                names.extend(family_labels[family])
        return names

    def extract_all(
        self,
        attentions: Union[tuple, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Extract all configured feature families from attention tensors.

        Parameters
        ----------
        attentions : tuple of tensors or np.ndarray
            If tuple: HuggingFace format, each element is (batch, H, T, T).
            If np.ndarray: shape (L, H, T, T).

        Returns
        -------
        Dict[str, np.ndarray]
            Maps family name → feature vector.
            Also includes "combined" key with all features concatenated.
        """
        a = self._to_numpy(attentions)
        q = self.config.query_idx

        features = {}

        if self.config.entropy:
            features["entropy"] = compute_entropy_features(a, q)

        if self.config.lookback and self.context_length > 0:
            features["lookback"] = compute_lookback_features(a, self.context_length, q)

        if self.config.frequency:
            features["frequency"] = compute_frequency_features(a, q)

        if self.config.spectral:
            features["spectral"] = compute_spectral_features(a, q)

        if self.config.cross_layer_kl:
            features["cross_layer_kl"] = compute_kl_features(a, q)

        # Concatenate into a single vector
        features["combined"] = np.concatenate(list(features.values()))

        return features

    def extract_vector(
        self,
        attentions: Union[tuple, np.ndarray],
    ) -> np.ndarray:
        """Extract and return only the concatenated feature vector."""
        return self.extract_all(attentions)["combined"]

    @staticmethod
    def _to_numpy(attentions: Union[tuple, np.ndarray]) -> np.ndarray:
        """
        Convert HuggingFace attention tuple to numpy (L, H, T, T).

        HuggingFace returns: tuple of L tensors, each (batch, H, T, T).
        We take batch[0] and stack into (L, H, T, T).
        """
        if isinstance(attentions, np.ndarray):
            return attentions

        # Tuple of tensors (HuggingFace format)
        layers = []
        for layer_attn in attentions:
            if hasattr(layer_attn, "numpy"):
                arr = layer_attn[0].numpy()  # (H, T, T)
            elif hasattr(layer_attn, "detach"):
                arr = layer_attn[0].detach().cpu().numpy()
            else:
                arr = np.array(layer_attn[0])
            layers.append(arr)

        return np.stack(layers)  # (L, H, T, T)


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE ENGINEER — STANDALONE VALIDATION")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate GPT-2 sized attention: 12 layers, 12 heads, 16 tokens
    L, H, T = 12, 12, 16
    context_len = 10

    # Create synthetic attention (softmax of random logits)
    logits = rng.standard_normal((L, H, T, T))
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    attn = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    print(f"\nSynthetic attention: ({L}, {H}, {T}, {T})")
    print(f"Context length: {context_len} tokens")

    # --- Test 1: All families individually ---
    print("\n--- Test 1: Individual feature families ---")

    ent = compute_entropy_features(attn)
    print(f"  Entropy:    {ent} (shape {ent.shape})")
    assert ent.shape == (3,)
    assert ent[0] > 0  # mean entropy positive

    lb = compute_lookback_features(attn, context_len)
    print(f"  Lookback:   {lb} (shape {lb.shape})")
    assert lb.shape == (4,)
    assert 0 <= lb[0] <= 1  # ratio in [0, 1]

    freq = compute_frequency_features(attn)
    print(f"  Frequency:  {freq} (shape {freq.shape})")
    assert freq.shape == (4,)
    assert 0 <= freq[0] <= 1  # high-freq ratio in [0, 1]

    spec = compute_spectral_features(attn)
    print(f"  Spectral:   {spec} (shape {spec.shape})")
    assert spec.shape == (4,)
    assert spec[0] >= 0  # Fiedler value non-negative

    kl = compute_kl_features(attn)
    print(f"  Cross-KL:   {kl} (shape {kl.shape})")
    assert kl.shape == (3,)
    assert kl[0] >= 0  # total KL non-negative

    print("  All families produce correct shapes ✅")

    # --- Test 2: Unified extraction ---
    print("\n--- Test 2: Unified feature engineer ---")
    eng = AttentionFeatureEngineer(context_length=context_len)
    feats = eng.extract_all(attn)
    print(f"  Feature families: {list(feats.keys())}")
    print(f"  Combined vector dim: {feats['combined'].shape[0]}")
    print(f"  Feature names: {eng.feature_names}")
    assert feats["combined"].shape[0] == eng.feature_dim == 18
    assert np.all(np.isfinite(feats["combined"]))
    print(f"  All 18 features finite ✅")

    # --- Test 3: Uniform attention (baseline) ---
    print("\n--- Test 3: Uniform attention → max entropy, low KL ---")
    uniform = np.ones((L, H, T, T)) / T
    feats_u = eng.extract_all(uniform)
    print(f"  Entropy mean: {feats_u['entropy'][0]:.4f} (expected ~{np.log2(T):.4f})")
    print(f"  Cross-KL total: {feats_u['cross_layer_kl'][0]:.6f} (expected ~0)")
    assert abs(feats_u["entropy"][0] - np.log2(T)) < 0.01
    assert feats_u["cross_layer_kl"][0] < 1e-6
    print("  Baseline correct ✅")

    # --- Test 4: Peaked attention (confident) ---
    print("\n--- Test 4: Peaked attention → low entropy, high lookback ---")
    peaked = np.full((L, H, T, T), EPS)
    peaked[:, :, :, 0] = 1.0  # all attention on first (context) token
    peaked = peaked / peaked.sum(axis=-1, keepdims=True)
    feats_p = eng.extract_all(peaked)
    print(f"  Entropy mean: {feats_p['entropy'][0]:.4f} (expected ~0)")
    print(f"  Lookback mean: {feats_p['lookback'][0]:.4f} (expected ~1)")
    assert feats_p["entropy"][0] < 0.5
    assert feats_p["lookback"][0] > 0.9
    print("  Peaked attention correct ✅")

    # --- Test 5: Config toggle ---
    print("\n--- Test 5: Disable families ---")
    minimal = AttentionFeatureEngineer(
        context_length=0,
        config=FeatureConfig(
            entropy=True,
            lookback=False,
            frequency=False,
            spectral=False,
            cross_layer_kl=True,
        ),
    )
    feats_m = minimal.extract_all(attn)
    assert "lookback" not in feats_m
    assert "frequency" not in feats_m
    assert feats_m["combined"].shape[0] == 6  # 3 + 3
    print(f"  Minimal config: {feats_m['combined'].shape[0]} features ✅")

    # --- Test 6: HuggingFace tuple format ---
    print("\n--- Test 6: Tuple input (HuggingFace format) ---")
    attn_tuple = tuple(
        attn[l:l+1]  # simulate (1, H, T, T) per layer
        for l in range(L)
    )
    feats_t = eng.extract_vector(attn_tuple)
    # Should match numpy input
    feats_n = eng.extract_vector(attn)
    assert np.allclose(feats_t, feats_n, atol=1e-6)
    print("  Tuple ↔ numpy equivalence ✅")

    print(f"\n{'=' * 60}")
    print(f"Feature Engineer — ALL 6 CHECKS PASS ✅")
    print(f"{'=' * 60}")
