"""
Token-Level Entropy Baselines
================================

Classic single-pass predictive-entropy features computed directly from a
model's next-token output distribution (logits), as distinct from the
attention-pattern features in feature_engineer.py. This is the family of
signals the root README already describes conceptually (Shannon entropy of
p_t over the vocabulary, perplexity) but that had never actually been
implemented in code — everything in this repo prior to this module operated on
attention weights, not on the output token-probability distribution.

These are single-pass (no multi-sample generation, no MC dropout) — cheap
"established baseline" uncertainty signals, one well-known step short of
full semantic entropy:

    entropy_mean / entropy_max / entropy_std — Shannon entropy of the
        per-token predictive distribution, aggregated over the answer span.
        H_t = -sum_v p_t(v) * log p_t(v)
    perplexity — exp(mean negative log-likelihood) of the actually-generated
        answer tokens under the model's own distribution.
    topk_entropy_mean — entropy restricted to the renormalized top-k mass
        (same estimator later reused, restricted-K, by blackbox_detector.py
        to validate that approximation against full-vocab logits in tests).
    margin_mean — mean (top1 logprob - top2 logprob) over the answer span;
        a cheap, robust confidence-margin signal.

Usage:
    extractor = EntropyFeatureExtractor()
    feats = extractor.extract(logits, answer_start=12, target_ids=token_ids)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

EPS = 1e-12

FEATURE_NAMES: List[str] = [
    "entropy_mean",
    "entropy_max",
    "entropy_std",
    "perplexity",
    "topk_entropy_mean",
    "margin_mean",
]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (exp.sum(axis=axis, keepdims=True) + EPS)


def token_entropy(probs: np.ndarray, use_bits: bool = False) -> np.ndarray:
    """
    Shannon entropy per token position.

    Parameters
    ----------
    probs : np.ndarray, shape (T, V)
    use_bits : bool
        If True, use log2 (bits); default natural log (nats), matching the
        perplexity convention (exp(mean NLL)).

    Returns
    -------
    np.ndarray, shape (T,)
    """
    log = np.log2 if use_bits else np.log
    return -np.sum(probs * log(probs + EPS), axis=-1)


def _topk_renormalized_entropy(probs: np.ndarray, top_k: int, use_bits: bool = False) -> np.ndarray:
    """
    Entropy computed only over the top-k probabilities per position,
    renormalized to sum to 1. Underestimates true entropy (ignores tail
    mass) but is cheap and monotonically related to true entropy for
    concentrated distributions.

    probs : (T, V) -> returns (T,)
    """
    k = min(top_k, probs.shape[-1])
    topk = np.sort(probs, axis=-1)[:, -k:]
    topk_norm = topk / (topk.sum(axis=-1, keepdims=True) + EPS)
    log = np.log2 if use_bits else np.log
    return -np.sum(topk_norm * log(topk_norm + EPS), axis=-1)


def _margin(logits: np.ndarray) -> np.ndarray:
    """Per-position (top1 - top2) logit gap. logits: (T, V) -> (T,)."""
    top2 = np.sort(logits, axis=-1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def compute_entropy_baseline_features(
    logits: np.ndarray,
    target_ids: Optional[np.ndarray] = None,
    top_k: int = 5,
    use_bits: bool = False,
) -> np.ndarray:
    """
    Compute the fixed 6-dim entropy-baseline feature vector.

    Parameters
    ----------
    logits : np.ndarray, shape (T, V)
        Pre-softmax logits for each position of interest (already sliced to
        the answer span by the caller — see EntropyFeatureExtractor.extract).
    target_ids : np.ndarray, shape (T,), optional
        The actually-realized token id that logits[t] should have predicted
        (i.e. teacher-forcing targets, already aligned/shifted by the
        caller). Required for the `perplexity` feature; if omitted,
        perplexity is set to NaN.
    top_k : int
        K for the top-k-renormalized entropy estimator.
    use_bits : bool
        Entropy unit (bits vs nats). Default nats, matching perplexity.

    Returns
    -------
    np.ndarray, shape (6,)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (T, V), got shape {logits.shape}")
    if logits.shape[0] == 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    probs = softmax(logits, axis=-1)
    ent = token_entropy(probs, use_bits=use_bits)

    if target_ids is not None:
        target_ids = np.asarray(target_ids)
        if len(target_ids) != logits.shape[0]:
            raise ValueError(
                f"target_ids length {len(target_ids)} != logits length {logits.shape[0]}"
            )
        chosen_probs = probs[np.arange(len(target_ids)), target_ids]
        nll = -np.log(chosen_probs + EPS)
        perplexity = float(np.exp(nll.mean()))
    else:
        perplexity = float("nan")

    topk_ent = _topk_renormalized_entropy(probs, top_k=top_k, use_bits=use_bits)
    margin = _margin(logits)

    return np.array([
        float(ent.mean()),
        float(ent.max()),
        float(ent.std()),
        perplexity,
        float(topk_ent.mean()),
        float(margin.mean()),
    ])


@dataclass
class EntropyBaselineConfig:
    top_k: int = 5
    use_bits: bool = False  # nats by default, matches perplexity convention


class EntropyFeatureExtractor:
    """
    Extracts single-pass token-entropy features from output logits.

    Mirrors AttentionFeatureEngineer's API shape (feature_dim, feature_names,
    extract()) so it plugs into pipeline.py symmetrically alongside the
    attention-based feature engineer.
    """

    def __init__(self, config: Optional[EntropyBaselineConfig] = None) -> None:
        self.config = config or EntropyBaselineConfig()

    @property
    def feature_dim(self) -> int:
        return len(FEATURE_NAMES)

    @property
    def feature_names(self) -> List[str]:
        return list(FEATURE_NAMES)

    def extract(
        self,
        logits: np.ndarray,
        answer_start: int = 0,
        target_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Slice `logits` (and `target_ids`, if given) to the answer span
        starting at `answer_start`, then compute the feature vector.
        """
        logits_slice = logits[answer_start:]
        targets_slice = target_ids[answer_start:] if target_ids is not None else None
        return compute_entropy_baseline_features(
            logits_slice,
            target_ids=targets_slice,
            top_k=self.config.top_k,
            use_bits=self.config.use_bits,
        )


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTROPY BASELINES — STANDALONE VALIDATION")
    print("=" * 60)

    rng = np.random.default_rng(42)
    V = 50

    # Test 1: uniform distribution -> max entropy = log(V)
    print("\n--- Test 1: Uniform distribution entropy ---")
    uniform_logits = np.zeros((5, V))
    probs = softmax(uniform_logits)
    ent = token_entropy(probs)
    expected = np.log(V)
    assert np.allclose(ent, expected, atol=1e-4), f"{ent} != {expected}"
    print(f"  Uniform entropy: {ent[0]:.4f}  (expected log({V})={expected:.4f}) ✅")

    # Test 2: near-one-hot -> entropy near 0
    print("\n--- Test 2: Near-deterministic distribution entropy ---")
    onehot_logits = np.full((5, V), -20.0)
    onehot_logits[:, 0] = 20.0
    probs = softmax(onehot_logits)
    ent = token_entropy(probs)
    assert np.all(ent < 1e-3), f"expected near-zero entropy, got {ent}"
    print(f"  Near-one-hot entropy: {ent[0]:.6f} ✅")

    # Test 3: feature vector shape/names
    print("\n--- Test 3: Feature extraction ---")
    extractor = EntropyFeatureExtractor()
    logits = rng.standard_normal((20, V))
    target_ids = rng.integers(0, V, 20)
    feats = extractor.extract(logits, answer_start=5, target_ids=target_ids)
    assert feats.shape == (extractor.feature_dim,)
    assert len(extractor.feature_names) == extractor.feature_dim
    print(f"  Features: {dict(zip(extractor.feature_names, np.round(feats, 4)))}")
    print("  Feature extraction works ✅")

    # Test 4: perplexity is NaN without target_ids
    print("\n--- Test 4: Perplexity requires target_ids ---")
    feats_no_targets = extractor.extract(logits, answer_start=5)
    assert np.isnan(feats_no_targets[FEATURE_NAMES.index("perplexity")])
    print("  Perplexity correctly NaN without target_ids ✅")

    print(f"\n{'=' * 60}")
    print("Entropy baselines — ALL CHECKS PASS ✅")
    print(f"{'=' * 60}")
