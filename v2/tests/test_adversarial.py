"""
Tests for v2/adversarial.py

Covers the sequence-mode scoring regression: AdversarialEvaluator._score_text()
must call predict_proba_sequence() (not the flat predict_proba()) when
use_sequence=True and the detector exposes a sequence-specific method, since
a HallucinationDetector(classifier_type="bilstm") has no working flat
predict_proba() (see v2/tests/test_detector.py for that guard).
"""

import numpy as np
import pytest

import v2.pipeline as pipeline_mod
from v2.adversarial import AdversarialEvaluator, obfuscate_text, paraphrase_text, multilingual_prefix


class _RecordingSequenceDetector:
    """Exposes both predict_proba and predict_proba_sequence; records which
    one was called, like a HallucinationDetector(classifier_type='bilstm')."""

    def __init__(self):
        self.flat_calls = 0
        self.sequence_calls = 0

    def predict_proba(self, X):
        self.flat_calls += 1
        return np.array([0.5] * len(X))

    def predict_proba_sequence(self, X_seq):
        self.sequence_calls += 1
        return np.array([0.7] * len(X_seq))


class _FlatOnlyDetector:
    """Mimics a raw BiLSTMDetector: only has predict_proba, but it already
    accepts sequence input natively."""

    def __init__(self):
        self.flat_calls = 0

    def predict_proba(self, X_seq):
        self.flat_calls += 1
        return np.array([0.3] * len(X_seq))


class _FakeEngineer:
    def extract_layer_sequence(self, attentions):
        return np.zeros((4, 6))

    def extract(self, attentions, ctx_len):
        return np.zeros(18)


def _patch_attention_extraction(monkeypatch):
    monkeypatch.setattr(
        pipeline_mod,
        "extract_attention_from_model",
        lambda text, model, tokenizer, device: (np.zeros((2, 2, 3, 3)), 1),
    )


def test_score_text_uses_predict_proba_sequence_when_available(monkeypatch):
    _patch_attention_extraction(monkeypatch)
    detector = _RecordingSequenceDetector()
    evaluator = AdversarialEvaluator(
        detector=detector,
        engineer=_FakeEngineer(),
        model=object(),
        tokenizer=object(),
        use_sequence=True,
    )

    score = evaluator._score_text("Question: Q\nAnswer: A")

    assert detector.sequence_calls == 1
    assert detector.flat_calls == 0
    assert score == pytest.approx(0.7)


def test_score_text_falls_back_to_predict_proba_for_raw_bilstm(monkeypatch):
    _patch_attention_extraction(monkeypatch)
    detector = _FlatOnlyDetector()
    evaluator = AdversarialEvaluator(
        detector=detector,
        engineer=_FakeEngineer(),
        model=object(),
        tokenizer=object(),
        use_sequence=True,
    )

    score = evaluator._score_text("Question: Q\nAnswer: A")

    assert detector.flat_calls == 1
    assert score == pytest.approx(0.3)


def test_score_text_flat_mode_uses_predict_proba(monkeypatch):
    _patch_attention_extraction(monkeypatch)
    detector = _RecordingSequenceDetector()
    evaluator = AdversarialEvaluator(
        detector=detector,
        engineer=_FakeEngineer(),
        model=object(),
        tokenizer=object(),
        use_sequence=False,
    )

    score = evaluator._score_text("Question: Q\nAnswer: A")

    assert detector.flat_calls == 1
    assert detector.sequence_calls == 0
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Transform sanity checks (promoted from __main__ self-test block)
# ---------------------------------------------------------------------------

def test_obfuscate_text_modifies_text():
    sample = "The capital of France is Paris, located along the Seine river."
    assert obfuscate_text(sample, rate=0.2, seed=0) != sample


def test_multilingual_prefix_prepends_language_instruction():
    sample = "The capital of France is Paris."
    assert multilingual_prefix(sample, "french").startswith("Répondez")


def test_paraphrase_preserves_word_count_or_fewer_changes():
    sample = "The city is large and known for its high towers."
    result = paraphrase_text(sample, seed=0)
    assert len(result.split()) == len(sample.split())
