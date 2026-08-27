"""
Black-Box Top-K Logprob Detector
====================================

The other detectors in this repo (feature_engineer.py, entropy_baselines.py)
assume white-box access — full attention weights or full-vocabulary logits.
Commercial completion APIs never expose that; the most a caller typically
gets is a small fixed-K list of top token logprobs per generated position
(e.g. OpenAI's `logprobs=True, top_logprobs=5`). This module builds a
detector that works under that realistic black-box constraint only.

Given only the top-K logprobs, exact Shannon entropy over the full
vocabulary is not computable — only a lower-bound/truncated estimate is
(the same `_topk_renormalized_entropy` estimator entropy_baselines.py uses
against full logits, applied here to the smaller K actually available).
This is documented explicitly: the estimate systematically UNDERESTIMATES
true entropy by ignoring tail mass, and the bias is worst exactly for
diffuse (high-entropy, hallucination-leaning) distributions — the case
where it matters most. It is still informative because a low top-K mass
already tells you a lot of probability lies outside the top-K, which itself
correlates with high true entropy.

Two logprob sources feed the same feature extraction:
    - simulate_topk_from_full_logits() — derives what an API's top_logprobs
      response would have looked like from full local logits (offline,
      no network, used by tests/CI and the synthetic/HaluEval pipeline).
    - fetch_topk_logprobs_openai() — real API call (lazy-imported optional
      dependency, requires OPENAI_API_KEY), used only for live demos.

Usage:
    seq = simulate_topk_from_full_logits(logits, chosen_token_ids, top_k=5)
    feats = extract_blackbox_features(seq)
    detector = BlackBoxEntropyDetector()
    detector.fit(X, y)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from detector import DetectorMetrics, HallucinationDetector

EPS = 1e-12

FEATURE_NAMES: List[str] = [
    "topk_entropy_mean",
    "topk_entropy_max",
    "topk_entropy_std",
    "topk_mass_mean",
    "topk_mass_min",
    "margin_mean",
    "margin_min",
]

try:
    import openai  # noqa: F401
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# =========================================================================
# Data structure
# =========================================================================

@dataclass
class TokenTopK:
    """One generated token's top-K logprob response, in the shape a
    commercial completions API returns (e.g. OpenAI's `top_logprobs`)."""
    token: str
    logprob: float                          # chosen token's own logprob
    top_logprobs: List[float] = field(default_factory=list)  # sorted descending, includes `logprob`


# =========================================================================
# Feature functions — computed only from top-K mass
# =========================================================================

def topk_entropy_lower_bound(top_logprobs: List[float]) -> float:
    """
    Entropy computed only over the observed top-k probabilities, renormalized
    to sum to 1:
        p_hat_i = p_i / sum_{j=1}^{k} p_j     (i = 1, ..., k)
        H^(k) = -sum_{i=1}^{k} p_hat_i * log(p_hat_i)
    This UNDERESTIMATES true entropy H^(V) = -sum_{v in V} p_v log p_v
    (ignores tail mass): H^(k) <= H^(V) always. Do not treat it as an
    approximation of full entropy for diffuse distributions; it is a lower
    bound, tightest when the true distribution is already concentrated in
    the top-k. See README.md's "Top-K Logprob Entropy" section for a worked
    numeric example of the gap size.
    """
    probs = np.exp(np.asarray(top_logprobs, dtype=float))
    probs = probs / (probs.sum() + EPS)
    return float(-np.sum(probs * np.log(probs + EPS)))


def topk_mass(top_logprobs: List[float]) -> float:
    """
    Sum of exp(logprob) over the top-k — a concentration proxy:
        mass^(k) = sum_{i=1}^{k} p_i
    Close to 1 means confident/concentrated; low values mean much mass lies
    outside the top-k (itself evidence of high true entropy, independent of
    the lower-bound entropy estimate above — a low top-k mass is informative
    even without knowing the shape of the unseen tail).
    """
    return float(np.exp(np.asarray(top_logprobs, dtype=float)).sum())


def top1_top2_margin(top_logprobs_sorted: List[float]) -> float:
    """logprob[0] - logprob[1] — confidence margin. 0 if fewer than 2 entries
    are available (maximally ambiguous by convention)."""
    if len(top_logprobs_sorted) < 2:
        return 0.0
    return float(top_logprobs_sorted[0] - top_logprobs_sorted[1])


def extract_blackbox_features(sequence: List[TokenTopK]) -> np.ndarray:
    """
    Aggregate per-token top-K features over an answer's token span into a
    fixed 7-dim vector, regardless of K.
    """
    if len(sequence) == 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    entropies = np.array([topk_entropy_lower_bound(t.top_logprobs) for t in sequence])
    masses = np.array([topk_mass(t.top_logprobs) for t in sequence])
    margins = np.array([top1_top2_margin(t.top_logprobs) for t in sequence])

    return np.array([
        float(entropies.mean()),
        float(entropies.max()),
        float(entropies.std()),
        float(masses.mean()),
        float(masses.min()),
        float(margins.mean()),
        float(margins.min()),
    ])


# =========================================================================
# Offline simulation path (for CI / tests / synthetic+HaluEval pipeline)
# =========================================================================

def simulate_topk_from_full_logits(
    logits: np.ndarray,
    chosen_token_ids: np.ndarray,
    top_k: int = 5,
) -> List[TokenTopK]:
    """
    Derive what a commercial API's `top_logprobs=k` response would have
    looked like, from full local logits — lets blackbox_detector's feature
    extraction and tests run fully offline/CPU-only, with no API key.

    Parameters
    ----------
    logits : np.ndarray, shape (T, V)
        Full-vocabulary logits (e.g. from pipeline.extract_logits_from_model).
    chosen_token_ids : np.ndarray, shape (T,)
        The actually-realized token id at each position.
    top_k : int
        K, matching the real API's top_logprobs parameter.

    Returns
    -------
    List[TokenTopK]
    """
    from entropy_baselines import softmax

    probs = softmax(logits, axis=-1)
    logprobs = np.log(probs + EPS)

    sequence: List[TokenTopK] = []
    for t in range(logits.shape[0]):
        row = logprobs[t]
        k = min(top_k, len(row))
        top_idx = np.argsort(row)[-k:][::-1]  # descending
        chosen_id = int(chosen_token_ids[t])

        top_ids = list(top_idx)
        top_vals = [float(row[i]) for i in top_idx]

        # Real APIs (e.g. OpenAI) always include the sampled token's own
        # logprob even if it falls outside the top-k; mirror that here.
        if chosen_id not in top_ids:
            top_ids.append(chosen_id)
            top_vals.append(float(row[chosen_id]))
            order = np.argsort(top_vals)[::-1]
            top_ids = [top_ids[i] for i in order]
            top_vals = [top_vals[i] for i in order]

        sequence.append(TokenTopK(
            token=str(chosen_id),
            logprob=float(row[chosen_id]),
            top_logprobs=top_vals,
        ))
    return sequence


# =========================================================================
# Real-API path (lazy-optional, OpenAI)
# =========================================================================

def fetch_topk_logprobs_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    top_k: int = 5,
    max_tokens: int = 100,
    api_key: Optional[str] = None,
) -> List[TokenTopK]:
    """
    Call OpenAI's chat.completions with logprobs=True, top_logprobs=top_k to
    fetch real black-box top-K logprobs for a live demo.

    Requires the `openai` package and an API key from the OPENAI_API_KEY
    environment variable (or the explicit `api_key` argument) — never
    hardcode a key. Opt-in only: not required for any test in this repo to
    pass (see simulate_topk_from_full_logits for the offline path CI uses).
    """
    if not _HAS_OPENAI:
        raise ImportError("Black-box API path requires: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it in your environment or "
            "pass api_key= explicitly — never hardcode an API key in source."
        )

    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_k,
    )

    sequence: List[TokenTopK] = []
    content = response.choices[0].logprobs.content or []
    for token_logprob in content:
        top_vals = [entry.logprob for entry in token_logprob.top_logprobs]
        sequence.append(TokenTopK(
            token=token_logprob.token,
            logprob=token_logprob.logprob,
            top_logprobs=top_vals,
        ))
    return sequence


# =========================================================================
# Detector
# =========================================================================

class BlackBoxEntropyDetector:
    """
    Same fit/predict_proba/evaluate contract as HallucinationDetector,
    operating on the 7-dim top-K feature vector. Composes
    HallucinationDetector(classifier_type="logistic") internally — the
    novel part is the feature extraction under the top-K constraint, not a
    new classifier architecture.
    """

    def __init__(self, feature_names: Optional[List[str]] = None, **logreg_kwargs) -> None:
        self._inner = HallucinationDetector(
            classifier_type="logistic",
            feature_names=feature_names or list(FEATURE_NAMES),
            **logreg_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlackBoxEntropyDetector":
        self._inner.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._inner.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self._inner.predict(X, threshold)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> DetectorMetrics:
        return self._inner.evaluate(X, y, threshold)

    def feature_importance(self):
        return self._inner.feature_importance()

    def save(self, path: str) -> None:
        self._inner.save(path)

    @classmethod
    def load(cls, path: str) -> "BlackBoxEntropyDetector":
        det = cls()
        det._inner = HallucinationDetector.load(path)
        return det


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BLACKBOX DETECTOR — STANDALONE VALIDATION")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Test 1: simulated top-k entropy is a lower bound on true entropy
    print("\n--- Test 1: Top-k entropy is a lower bound ---")
    from entropy_baselines import softmax, token_entropy

    V = 50
    logits = rng.standard_normal((20, V))
    chosen_ids = rng.integers(0, V, 20)
    full_entropy = token_entropy(softmax(logits)).mean()

    sequence = simulate_topk_from_full_logits(logits, chosen_ids, top_k=5)
    feats = extract_blackbox_features(sequence)
    topk_entropy_mean = feats[FEATURE_NAMES.index("topk_entropy_mean")]
    assert topk_entropy_mean <= full_entropy + 1e-9, "top-k entropy should not exceed full entropy"
    print(f"  Full entropy: {full_entropy:.4f}   Top-k(5) entropy: {topk_entropy_mean:.4f} ✅")

    # Test 2: fixed feature dim regardless of k
    print("\n--- Test 2: Fixed feature dim regardless of k ---")
    seq_k3 = simulate_topk_from_full_logits(logits, chosen_ids, top_k=3)
    seq_k10 = simulate_topk_from_full_logits(logits, chosen_ids, top_k=10)
    assert extract_blackbox_features(seq_k3).shape == (len(FEATURE_NAMES),)
    assert extract_blackbox_features(seq_k10).shape == (len(FEATURE_NAMES),)
    print("  Feature dim is fixed at", len(FEATURE_NAMES), "regardless of k ✅")

    # Test 3: end-to-end offline detector fit on separable synthetic data
    print("\n--- Test 3: End-to-end offline fit ---")
    N = 200
    X_list, y_list = [], []
    for i in range(N):
        is_halluc = rng.random() > 0.5
        base_logits = rng.standard_normal((15, V))
        if is_halluc:
            base_logits *= 0.3  # flatter distribution -> higher entropy
        ids = rng.integers(0, V, 15)
        seq = simulate_topk_from_full_logits(base_logits, ids, top_k=5)
        X_list.append(extract_blackbox_features(seq))
        y_list.append(1.0 if is_halluc else 0.0)

    X = np.array(X_list)
    y = np.array(y_list)
    split = int(0.7 * N)

    det = BlackBoxEntropyDetector()
    det.fit(X[:split], y[:split])
    metrics = det.evaluate(X[split:], y[split:])
    print(f"  AUROC: {metrics.auroc:.4f}")
    assert metrics.auroc > 0.6, "Detector should separate flatter (hallucinated) distributions"
    print("  End-to-end offline fit works ✅")

    # Test 4: missing package/key paths raise clear errors
    print("\n--- Test 4: Real-API path error handling ---")
    if not _HAS_OPENAI:
        try:
            fetch_topk_logprobs_openai("test prompt")
            raise AssertionError("Expected ImportError")
        except ImportError as e:
            assert "pip install openai" in str(e)
            print("  Missing openai package raises clear ImportError ✅")
    else:
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            fetch_topk_logprobs_openai("test prompt")
            raise AssertionError("Expected ValueError for missing API key")
        except ValueError as e:
            assert "OPENAI_API_KEY" in str(e)
            print("  Missing API key raises clear ValueError ✅")

    print(f"\n{'=' * 60}")
    print("Blackbox detector — ALL CHECKS PASS ✅")
    print(f"{'=' * 60}")
