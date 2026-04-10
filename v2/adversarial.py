"""
Adversarial Robustness Evaluation
===================================

Evaluates hallucination detector robustness against three attack vectors:

    1. OBFUSCATION  — character-level substitutions and spacing tricks
    2. PARAPHRASE   — synonym replacement while preserving meaning
    3. MULTILINGUAL — prefix-based language shift (cross-lingual transfer)

These correspond to realistic adversarial scenarios where a malicious actor
(or a model operating on noisy/translated input) tries to evade detection.

Design rationale:
    Obfuscation tests whether the detector relies on surface token patterns
    rather than semantic attention structure. A robust detector should score
    nearly identically on "Paris" and "P4r1s" because attention entropy and
    KL divergence are computed over token distributions, not token identity.

    Paraphrase tests whether rephrasing correct/hallucinated content shifts
    the detector's score. A robust detector should be paraphrase-invariant.

    Multilingual tests cross-lingual transfer: does the detector trained on
    English QA generalise to equivalent Spanish/French/German prompts?

Usage:
    from v2.adversarial import AdversarialEvaluator
    evaluator = AdversarialEvaluator(detector, feature_engineer, model, tokenizer)
    results = evaluator.evaluate_all(samples)
    evaluator.print_report(results)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# =========================================================================
# Attack functions
# =========================================================================

# Character-level obfuscation map (visually similar substitutions)
_HOMOGLYPHS = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$',
    'l': '1', 'g': '9', 't': '7', 'A': '4', 'E': '3',
}

_SYNONYMS: Dict[str, List[str]] = {
    "large":   ["big", "substantial", "considerable"],
    "small":   ["tiny", "little", "minor"],
    "found":   ["discovered", "identified", "located"],
    "first":   ["initial", "primary", "earliest"],
    "known":   ["recognized", "identified", "established"],
    "made":    ["produced", "created", "constructed"],
    "called":  ["named", "termed", "designated"],
    "used":    ["employed", "utilized", "applied"],
    "located": ["situated", "positioned", "found"],
    "built":   ["constructed", "erected", "created"],
    "born":    ["originating", "native"],
    "wrote":   ["authored", "composed", "penned"],
    "said":    ["stated", "claimed", "noted"],
    "high":    ["elevated", "tall", "lofty"],
    "low":     ["minimal", "reduced", "diminished"],
    "show":    ["demonstrate", "indicate", "reveal"],
    "shows":   ["demonstrates", "indicates", "reveals"],
    "include": ["encompass", "comprise", "contain"],
    "includes":["encompasses", "comprises", "contains"],
}

_LANG_PREFIXES = {
    "spanish":  "Responde en español: ",
    "french":   "Répondez en français: ",
    "german":   "Antworten Sie auf Deutsch: ",
    "japanese": "日本語で答えてください: ",
    "chinese":  "请用中文回答: ",
}


def obfuscate_text(text: str, rate: float = 0.15, seed: int = 42) -> str:
    """
    Character-level obfuscation: replace ~rate fraction of eligible chars
    with visually similar substitutes.

    Rate 0.15 is sufficient to break naive token-matching while remaining
    readable by humans and most tokenisers.
    """
    rng = random.Random(seed)
    result = []
    for ch in text:
        if ch in _HOMOGLYPHS and rng.random() < rate:
            result.append(_HOMOGLYPHS[ch])
        else:
            result.append(ch)
    return "".join(result)


def paraphrase_text(text: str, seed: int = 42) -> str:
    """
    Word-level synonym substitution.

    Replaces known words with synonyms from a curated map. Preserves
    sentence structure and factual content — only surface form changes.
    """
    rng = random.Random(seed)
    words = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(".,;:!?\"'")
        if clean in _SYNONYMS and rng.random() < 0.4:
            synonym = rng.choice(_SYNONYMS[clean])
            # Preserve capitalisation
            if word[0].isupper():
                synonym = synonym.capitalize()
            result.append(synonym)
        else:
            result.append(word)
    return " ".join(result)


def multilingual_prefix(text: str, language: str = "spanish") -> str:
    """
    Prepend a language instruction prefix to the text.

    Tests cross-lingual robustness: does the detector score change when
    the model is prompted in a different language?
    """
    prefix = _LANG_PREFIXES.get(language, _LANG_PREFIXES["spanish"])
    return prefix + text


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class AttackResult:
    """Result of one attack type on one sample."""
    attack_type: str
    original_score: float
    attacked_score: float
    score_delta: float        # attacked - original
    label: str                # true label
    original_text: str = ""
    attacked_text:  str = ""


@dataclass
class RobustnessReport:
    """Aggregated robustness results across all attacks."""
    results_by_attack: Dict[str, List[AttackResult]] = field(default_factory=dict)

    def mean_delta(self, attack: str) -> float:
        return float(np.mean([r.score_delta for r in self.results_by_attack[attack]]))

    def std_delta(self, attack: str) -> float:
        return float(np.std([r.score_delta for r in self.results_by_attack[attack]]))

    def rank_stability(self, attack: str, threshold: float = 0.05) -> float:
        """Fraction of samples where attack shifts score by less than threshold."""
        stable = sum(
            1 for r in self.results_by_attack[attack]
            if abs(r.score_delta) < threshold
        )
        return stable / max(len(self.results_by_attack[attack]), 1)


# =========================================================================
# Evaluator
# =========================================================================

class AdversarialEvaluator:
    """
    Evaluates hallucination detector robustness under three attack types.

    Parameters
    ----------
    detector : HallucinationDetector or BiLSTMDetector
        Trained detector. Must have predict_proba() or predict_proba_sequence().
    engineer : AttentionFeatureEngineer
        For flat feature extraction.
    model : transformers model
        HuggingFace causal LM (already loaded, on correct device).
    tokenizer : transformers tokenizer
    use_sequence : bool
        If True, uses BiLSTM sequence features. Otherwise uses 18D flat vector.
    device : str
    seed : int
    """

    def __init__(
        self,
        detector,
        engineer,
        model,
        tokenizer,
        use_sequence: bool = False,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.detector     = detector
        self.engineer     = engineer
        self.model        = model
        self.tokenizer    = tokenizer
        self.use_sequence = use_sequence
        self.device       = device
        self.seed         = seed

    def _score_text(self, text: str) -> float:
        """Extract features from text and return hallucination probability."""
        from v2.pipeline import extract_attention_from_model
        try:
            attentions, ctx_len = extract_attention_from_model(
                text, self.model, self.tokenizer, self.device
            )
            if self.use_sequence:
                seq = self.engineer.extract_layer_sequence(attentions)
                return float(self.detector.predict_proba(seq[None])[0])
            else:
                feats = self.engineer.extract(attentions, ctx_len)
                return float(self.detector.predict_proba(feats[None])[0])
        except Exception:
            return float("nan")

    def _run_attack(
        self,
        samples,
        transform_fn: Callable[[str], str],
        attack_name: str,
        max_samples: int = 100,
    ) -> List[AttackResult]:
        results = []
        eval_samples = samples[:max_samples]

        for sample in eval_samples:
            original_text = f"Question: {sample.question}\nAnswer: {sample.model_answer}"
            attacked_text = f"Question: {sample.question}\nAnswer: {transform_fn(sample.model_answer)}"

            orig_score = self._score_text(original_text)
            atk_score  = self._score_text(attacked_text)

            if not (np.isnan(orig_score) or np.isnan(atk_score)):
                results.append(AttackResult(
                    attack_type=attack_name,
                    original_score=orig_score,
                    attacked_score=atk_score,
                    score_delta=atk_score - orig_score,
                    label=sample.label,
                    original_text=original_text[:120],
                    attacked_text=attacked_text[:120],
                ))
        return results

    def evaluate_all(
        self,
        samples,
        max_samples: int = 100,
    ) -> RobustnessReport:
        """
        Run all three attack types and return a RobustnessReport.

        Parameters
        ----------
        samples : List[LabeledSample]
            Non-ambiguous labeled samples.
        max_samples : int
            Cap on samples per attack (keeps runtime manageable).
        """
        rng = random.Random(self.seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        clean = [s for s in shuffled if s.label != "ambiguous"]

        report = RobustnessReport()

        print("Running adversarial robustness evaluation...")

        print(f"  [1/3] Obfuscation attack (n={min(max_samples, len(clean))})...")
        report.results_by_attack["obfuscation"] = self._run_attack(
            clean,
            lambda t: obfuscate_text(t, rate=0.15, seed=self.seed),
            "obfuscation",
            max_samples,
        )

        print(f"  [2/3] Paraphrase attack (n={min(max_samples, len(clean))})...")
        report.results_by_attack["paraphrase"] = self._run_attack(
            clean,
            lambda t: paraphrase_text(t, seed=self.seed),
            "paraphrase",
            max_samples,
        )

        print(f"  [3/3] Multilingual attack — Spanish prefix (n={min(max_samples, len(clean))})...")
        report.results_by_attack["multilingual"] = self._run_attack(
            clean,
            lambda t: multilingual_prefix(t, "spanish"),
            "multilingual",
            max_samples,
        )

        return report

    @staticmethod
    def print_report(report: RobustnessReport) -> None:
        """Print a formatted robustness summary."""
        print(f"\n{'═' * 60}")
        print(f"  ADVERSARIAL ROBUSTNESS REPORT")
        print(f"{'═' * 60}")
        print(f"  {'Attack':<18} {'N':>5}  {'Mean Δ':>8}  {'Std Δ':>7}  {'Stable(±0.05)':>13}")
        print(f"  {'─' * 56}")
        for attack, results in report.results_by_attack.items():
            if not results:
                continue
            n = len(results)
            mu  = report.mean_delta(attack)
            std = report.std_delta(attack)
            stab = report.rank_stability(attack, threshold=0.05)
            status = "✅" if stab > 0.80 else ("⚠️ " if stab > 0.60 else "❌")
            print(f"  {attack:<18} {n:>5}  {mu:>+8.4f}  {std:>7.4f}  {stab:>12.1%}  {status}")
        print(f"{'═' * 60}")
        print(f"  Stability > 80% = robust  |  > 60% = acceptable  |  < 60% = fragile")
        print(f"{'═' * 60}")


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("Adversarial — Transform Validation")
    print("=" * 50)

    sample_text = "The capital of France is Paris, located along the Seine river."

    obf = obfuscate_text(sample_text, rate=0.2, seed=0)
    par = paraphrase_text(sample_text, seed=0)
    mul = multilingual_prefix(sample_text, "french")

    print(f"Original:    {sample_text}")
    print(f"Obfuscated:  {obf}")
    print(f"Paraphrase:  {par}")
    print(f"Multilingual:{mul}")

    # Verify transforms change text
    assert obf != sample_text, "Obfuscation should modify text"
    assert mul.startswith("Répondez"), "Multilingual prefix wrong"
    print("\nAll transform checks pass ✅")
