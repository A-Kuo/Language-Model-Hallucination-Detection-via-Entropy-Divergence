"""Tests for entropy_baselines.py"""

import numpy as np
import pytest

from entropy_baselines import (
    softmax,
    token_entropy,
    compute_entropy_baseline_features,
    EntropyFeatureExtractor,
    FEATURE_NAMES,
)


def test_softmax_sums_to_one():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((10, 30))
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)


def test_uniform_distribution_max_entropy():
    V = 40
    logits = np.zeros((5, V))
    probs = softmax(logits)
    ent = token_entropy(probs)
    assert np.allclose(ent, np.log(V), atol=1e-4)


def test_one_hot_zero_entropy():
    V = 40
    logits = np.full((5, V), -30.0)
    logits[:, 3] = 30.0
    probs = softmax(logits)
    ent = token_entropy(probs)
    assert np.all(ent < 1e-4)


def test_feature_dim_matches_names():
    extractor = EntropyFeatureExtractor()
    assert extractor.feature_dim == len(FEATURE_NAMES)
    assert extractor.feature_names == FEATURE_NAMES


def test_perplexity_matches_known_formula():
    # 2 positions, vocab size 2, deterministic logits so probs are exact.
    logits = np.array([[0.0, 0.0], [0.0, 0.0]])  # uniform -> p=0.5 each
    target_ids = np.array([0, 1])
    feats = compute_entropy_baseline_features(logits, target_ids=target_ids)
    idx = FEATURE_NAMES.index("perplexity")
    # NLL = -log(0.5) for both -> mean NLL = -log(0.5) -> perplexity = 2.0
    assert feats[idx] == pytest.approx(2.0, rel=1e-3)


def test_perplexity_nan_without_target_ids():
    logits = np.random.default_rng(1).standard_normal((5, 20))
    feats = compute_entropy_baseline_features(logits)
    idx = FEATURE_NAMES.index("perplexity")
    assert np.isnan(feats[idx])


def test_topk_entropy_leq_full_entropy():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((30, 100))
    probs = softmax(logits)
    full_ent = token_entropy(probs)
    feats = compute_entropy_baseline_features(logits, top_k=5)
    # topk_entropy_mean should not exceed the mean full entropy (top-k is a
    # lower-bound / truncated estimate of the full distribution's entropy).
    idx = FEATURE_NAMES.index("topk_entropy_mean")
    assert feats[idx] <= full_ent.mean() + 1e-9


def test_extractor_slices_to_answer_span():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((10, 20))
    target_ids = rng.integers(0, 20, 10)
    extractor = EntropyFeatureExtractor()

    full = extractor.extract(logits, answer_start=0, target_ids=target_ids)
    sliced = extractor.extract(logits, answer_start=5, target_ids=target_ids)
    manual = compute_entropy_baseline_features(logits[5:], target_ids=target_ids[5:])

    assert np.allclose(sliced, manual, equal_nan=True)
    assert not np.allclose(full, sliced, equal_nan=True)


def test_mismatched_target_ids_length_raises():
    logits = np.zeros((5, 10))
    with pytest.raises(ValueError):
        compute_entropy_baseline_features(logits, target_ids=np.zeros(3))


def test_non_2d_logits_raises():
    with pytest.raises(ValueError):
        compute_entropy_baseline_features(np.zeros((5,)))
