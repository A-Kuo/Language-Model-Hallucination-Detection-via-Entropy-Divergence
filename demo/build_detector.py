"""
Prerequisite build step for the Streamlit demo (demo/app.py).

Fits a CalibratedEntropyDetector on real, paired HaluEval data + real Pythia
attention/entropy features, and saves it to demo/detector.pkl so the demo
can load a pre-trained detector at startup instead of retraining on every
launch (matching how pipeline.py's `--save` already persists LogReg/MLP —
this is the same idea, just for the calibrated detector specifically, since
pipeline.py's CLI only ever persists LogReg/MLP/BiLSTM, never
CalibratedEntropyDetector, which is fit-and-discarded inside CV folds there).

Also writes demo/sample_pairs.json — a small, static set of real HaluEval
question/correct/hallucinated triples the deployed app reads directly,
instead of calling DataGenerator.from_halueval() (and therefore needing
`datasets` plus a live HuggingFace Hub download) at demo runtime. That live
call used to run unconditionally on every cold start (Streamlit re-executes
the whole script, including both tabs' bodies, on every run) and was a real
contributor to the deployed demo taking a very long time to show anything.

Usage:
    python demo/build_detector.py
    python demo/build_detector.py --num_samples 400 --model EleutherAI/pythia-160m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_generator import DataGenerator
from feature_engineer import AttentionFeatureEngineer
from entropy_baselines import EntropyFeatureExtractor
from pipeline import extract_attention_from_model, extract_logits_from_model, build_prompt_and_text
from calibrated_entropy_detector import CalibratedEntropyDetector


def build_sample_pairs(samples, out_path: str, n_pairs: int = 30) -> None:
    """Write up to n_pairs matched question/correct/hallucinated triples."""
    by_question = {}
    for s in samples:
        by_question.setdefault(s.question, {})[s.label] = s.model_answer
    pairs = [
        {"question": q, "correct": d["correct"], "hallucinated": d["hallucinated"]}
        for q, d in by_question.items()
        if "correct" in d and "hallucinated" in d
    ][:n_pairs]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} sample pairs to {out}")


def build(num_samples: int, model_name: str, out_path: str, seed: int = 42) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {num_samples} paired HaluEval samples...")
    samples = DataGenerator.from_halueval(num_samples=num_samples, seed=seed)
    clean = [s for s in samples if s.label != "ambiguous"]

    build_sample_pairs(samples, str(Path(__file__).parent / "sample_pairs.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    engineer = AttentionFeatureEngineer(context_length=32)
    entropy_extractor = EntropyFeatureExtractor()

    X_list, y_list = [], []
    failed = 0
    print(f"Extracting features for {len(clean)} samples...")
    for i, sample in enumerate(clean):
        try:
            prompt, text = build_prompt_and_text(tokenizer, sample.question, sample.model_answer)
            attentions, context_len = extract_attention_from_model(text, model, tokenizer, device, prompt=prompt)
            attn_feats = engineer.extract(attentions, context_len)
            logits, token_ids, answer_start = extract_logits_from_model(text, model, tokenizer, device, prompt=prompt)
            entropy_feats = entropy_extractor.extract(logits, answer_start=answer_start, target_ids=token_ids)
            if not (np.all(np.isfinite(attn_feats)) and np.all(np.isfinite(entropy_feats))):
                raise ValueError("non-finite feature value")
            X_list.append(np.concatenate([attn_feats, entropy_feats]))
            y_list.append(1.0 if sample.label == "hallucinated" else 0.0)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Warning: sample {i} failed — {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clean)} processed (failed: {failed})")

    X = np.array(X_list)
    y = np.array(y_list)
    feature_names = engineer.feature_names + entropy_extractor.feature_names
    print(f"Feature matrix: {X.shape}  failed: {failed}")
    print(f"Labels: {int(y.sum())} hallucinated / {int((y == 0).sum())} correct")

    print("Fitting CalibratedEntropyDetector (score_index='auto')...")
    det = CalibratedEntropyDetector(score_index="auto", feature_names=feature_names)
    det.fit(X, y)
    metrics = det.evaluate(X, y)
    print(f"In-sample AUROC: {metrics.auroc:.4f}  (in-sample only — not a held-out estimate; "
          f"see results/halueval_pythia160m_n400_paired.json for the real CV number)")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    det.save(str(out))
    print(f"Saved detector to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "detector.pkl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.num_samples, args.model, args.out, args.seed)
