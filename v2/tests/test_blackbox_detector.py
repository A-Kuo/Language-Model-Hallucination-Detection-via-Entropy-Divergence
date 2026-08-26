"""Tests for v2/blackbox_detector.py"""

import os

import numpy as np
import pytest

from v2.blackbox_detector import (
    TokenTopK,
    topk_entropy_lower_bound,
    topk_mass,
    top1_top2_margin,
    extract_blackbox_features,
    simulate_topk_from_full_logits,
    fetch_topk_logprobs_openai,
    BlackBoxEntropyDetector,
    FEATURE_NAMES,
    _HAS_OPENAI,
)
from v2.entropy_baselines import softmax, token_entropy


def test_simulate_topk_entropy_is_lower_bound_of_full_entropy():
    rng = np.random.default_rng(0)
    V = 60
    logits = rng.standard_normal((25, V))
    chosen_ids = rng.integers(0, V, 25)

    full_entropy = token_entropy(softmax(logits))
    seq = simulate_topk_from_full_logits(logits, chosen_ids, top_k=5)
    per_token_topk_entropy = np.array([topk_entropy_lower_bound(t.top_logprobs) for t in seq])

    assert np.all(per_token_topk_entropy <= full_entropy + 1e-9)


def test_feature_extraction_fixed_dim_regardless_of_k():
    rng = np.random.default_rng(1)
    V = 40
    logits = rng.standard_normal((10, V))
    ids = rng.integers(0, V, 10)

    for k in (2, 5, 20):
        seq = simulate_topk_from_full_logits(logits, ids, top_k=k)
        feats = extract_blackbox_features(seq)
        assert feats.shape == (len(FEATURE_NAMES),)


def test_topk_mass_bounded_in_zero_one():
    # logprobs must come from a valid softmax over some vocabulary, else
    # their exponentials aren't guaranteed to sum to <= 1.
    rng = np.random.default_rng(5)
    logits = rng.standard_normal(50)
    probs = softmax(logits[None, :])[0]
    logprobs = sorted(np.log(probs), reverse=True)[:5]
    mass = topk_mass(logprobs)
    assert 0.0 <= mass <= 1.0 + 1e-9


def test_top1_top2_margin_ordering():
    assert top1_top2_margin([-0.1, -2.0, -3.0]) == pytest.approx(1.9)
    assert top1_top2_margin([-0.1]) == 0.0
    assert top1_top2_margin([]) == 0.0


def test_empty_sequence_returns_nan_vector():
    feats = extract_blackbox_features([])
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isnan(feats))


def test_chosen_token_included_even_if_outside_topk():
    rng = np.random.default_rng(2)
    V = 100
    logits = rng.standard_normal((5, V))
    # Force chosen token to be the least-likely one (outside any small top-k).
    chosen_ids = np.argmin(logits, axis=-1)
    seq = simulate_topk_from_full_logits(logits, chosen_ids, top_k=3)
    for t in seq:
        assert t.logprob in t.top_logprobs
        assert len(t.top_logprobs) >= 3  # top-3 plus the forced-in chosen token


def test_end_to_end_offline_detector_fit_separates_flat_vs_peaked():
    rng = np.random.default_rng(3)
    V = 50
    N = 150
    X_list, y_list = [], []
    for _ in range(N):
        is_halluc = rng.random() > 0.5
        logits = rng.standard_normal((10, V))
        if is_halluc:
            logits *= 0.2  # flatter -> higher entropy -> "hallucinated"
        ids = rng.integers(0, V, 10)
        seq = simulate_topk_from_full_logits(logits, ids, top_k=5)
        X_list.append(extract_blackbox_features(seq))
        y_list.append(1.0 if is_halluc else 0.0)

    X, y = np.array(X_list), np.array(y_list)
    split = int(0.7 * N)
    det = BlackBoxEntropyDetector()
    det.fit(X[:split], y[:split])
    metrics = det.evaluate(X[split:], y[split:])
    assert metrics.auroc > 0.6


def test_openai_path_skipped_without_key_or_package(monkeypatch):
    if not _HAS_OPENAI:
        with pytest.raises(ImportError, match="pip install openai"):
            fetch_topk_logprobs_openai("test prompt")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            fetch_topk_logprobs_openai("test prompt")


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, 50).astype(float)

    det = BlackBoxEntropyDetector()
    det.fit(X, y)

    path = tmp_path / "blackbox.pkl"
    det.save(str(path))
    loaded = BlackBoxEntropyDetector.load(str(path))

    assert np.allclose(det.predict_proba(X), loaded.predict_proba(X))
