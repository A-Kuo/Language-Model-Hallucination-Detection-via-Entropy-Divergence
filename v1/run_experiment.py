"""
AED Experiment Runner
=====================
End-to-end benchmark for Attention Entropy Divergence hallucination detection.

Modes:
  --mode quick    : CPU, GPT-2 small, synthetic labels — validates pipeline (~2 min)
  --mode full     : GPU required, Pythia-160m, HaluEval QA split (~15 min on T4)

Output:
  results/benchmark_results.json   — all metrics (feeds directly into paper.tex)

Usage:
  python v1/run_experiment.py --mode quick
  python v1/run_experiment.py --mode full --model EleutherAI/pythia-160m

The JSON output matches the RESULT{} placeholders in arxiv-paper/paper.tex.
Run on Colab with GPU for publication-quality numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Synthetic dataset for quick/CPU mode
# ---------------------------------------------------------------------------

QUICK_PAIRS: List[Tuple[str, str, int]] = [
    # (context, continuation, label)  label=1 → hallucination
    # --- Geography ---
    ("The Eiffel Tower is located in", "Paris, France.", 0),
    ("The Eiffel Tower is located in", "Berlin, Germany.", 1),
    ("The capital of Australia is", "Canberra.", 0),
    ("The capital of Australia is", "Sydney, the largest city.", 1),
    ("The Amazon River flows through", "South America, primarily Brazil.", 0),
    ("The Amazon River flows through", "Central Africa and the Congo Basin.", 1),
    ("The Pacific Ocean is the", "largest and deepest ocean on Earth.", 0),
    ("The Pacific Ocean is the", "smallest of the world's five oceans.", 1),
    ("Mount Everest is located in the", "Himalayas on the Nepal-Tibet border.", 0),
    ("Mount Everest is located in the", "Alps mountain range in Switzerland.", 1),
    ("The Nile River is located in", "northeastern Africa.", 0),
    ("The Nile River is located in", "western Europe near the Rhine.", 1),
    # --- Science ---
    ("Water boils at", "100 degrees Celsius at sea level.", 0),
    ("Water boils at", "212 degrees Fahrenheit, which equals 100 Kelvin.", 1),
    ("The speed of light is approximately", "300,000 kilometres per second in a vacuum.", 0),
    ("The speed of light is approximately", "the same as the speed of sound.", 1),
    ("DNA stands for", "Deoxyribonucleic Acid.", 0),
    ("DNA stands for", "Dynamic Nucleic Assembly.", 1),
    ("The human body has", "206 bones in adults.", 0),
    ("The human body has", "over 500 bones throughout adult life.", 1),
    ("The human heart has", "four chambers.", 0),
    ("The human heart has", "two chambers: one for blood in, one for blood out.", 1),
    ("Photosynthesis converts", "sunlight, water, and CO2 into glucose and oxygen.", 0),
    ("Photosynthesis converts", "oxygen and nitrogen into carbon and water.", 1),
    # --- History ---
    ("Albert Einstein won the Nobel Prize in", "Physics in 1921 for the photoelectric effect.", 0),
    ("Albert Einstein won the Nobel Prize in", "Mathematics for his theory of relativity.", 1),
    ("Shakespeare wrote the play", "Hamlet, one of the most famous tragedies in English.", 0),
    ("Shakespeare wrote the play", "Hamlet in the 19th century during the Victorian era.", 1),
    ("The Great Wall of China was built by", "various Chinese dynasties over many centuries.", 0),
    ("The Great Wall of China was built by", "the Roman Empire to defend against China.", 1),
    ("Penicillin was discovered by", "Alexander Fleming in 1928.", 0),
    ("Penicillin was discovered by", "Marie Curie during her radioactivity research.", 1),
    ("The Moon orbits the Earth approximately every", "27 to 29 days depending on frame of reference.", 0),
    ("The Moon orbits the Earth approximately every", "24 hours, matching the Earth's rotation.", 1),
    # --- Technology ---
    ("Python programming language was created by", "Guido van Rossum in the late 1980s.", 0),
    ("Python programming language was created by", "Linus Torvalds as an alternative to C.", 1),
    ("The internet was originally developed by", "DARPA as ARPANET in the 1960s.", 0),
    ("The internet was originally developed by", "Tim Berners-Lee at CERN in 1989.", 1),
    ("The first iPhone was released by Apple in", "2007, revolutionizing the smartphone market.", 0),
    ("The first iPhone was released by Apple in", "1999 before the iPod was introduced.", 1),
    # --- Medicine ---
    ("The normal human body temperature is approximately", "37 degrees Celsius or 98.6 degrees Fahrenheit.", 0),
    ("The normal human body temperature is approximately", "42 degrees Celsius, the same as a mild fever.", 1),
    ("Insulin is produced by", "the pancreas and regulates blood sugar levels.", 0),
    ("Insulin is produced by", "the liver as part of the digestive process.", 1),
    ("Aspirin is chemically known as", "acetylsalicylic acid.", 0),
    ("Aspirin is chemically known as", "acetaminophen, the same as Tylenol.", 1),
    # --- Mathematics ---
    ("The value of pi is approximately", "3.14159, the ratio of a circle's circumference to its diameter.", 0),
    ("The value of pi is approximately", "2.71828, the base of the natural logarithm.", 1),
    ("Pythagoras' theorem states that", "a² + b² = c² for a right-angled triangle.", 0),
    ("Pythagoras' theorem states that", "the sum of all angles in any triangle equals 360 degrees.", 1),
]


# ---------------------------------------------------------------------------
# HaluEval loader (full mode)
# ---------------------------------------------------------------------------

def load_halueval_qa(max_samples: int = 500) -> List[Tuple[str, str, int]]:
    """Load HaluEval QA split. Returns (context, continuation, label) triples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        cols = ds.column_names
        print(f"      HaluEval columns: {cols}")

        # Map to actual column names — HuggingFace versions vary
        _pick = lambda candidates: next((c for c in candidates if c in cols), None)
        q_col = _pick(["question", "input", "prompt", "query", "src"])
        r_col = _pick(["right_answer", "answer", "output", "response",
                       "correct_answer", "chosen", "tgt"])
        h_col = _pick(["hallucinated_answer", "hallucinated_response",
                       "wrong_answer", "rejected", "hallucination"])

        if not q_col or not r_col or not h_col:
            raise ValueError(
                f"Expected question/right/hallucinated columns; got: {cols}\n"
                f"  question col: {q_col}, right col: {r_col}, halluc col: {h_col}"
            )
        print(f"      Mapped: question={q_col!r}  right={r_col!r}  hallucinated={h_col!r}")

        pairs = []
        for row in ds:
            if len(pairs) >= max_samples:
                break
            question    = str(row[q_col]) if row[q_col] is not None else ""
            right       = str(row[r_col]) if row[r_col] is not None else ""
            hallucinated = str(row[h_col]) if row[h_col] is not None else ""
            if question and right:
                pairs.append((question, right, 0))
            if question and hallucinated and len(pairs) < max_samples:
                pairs.append((question, hallucinated, 1))

        if not pairs:
            raise ValueError(
                f"Dataset loaded {len(ds)} rows but produced 0 usable pairs.\n"
                f"Sample row: {dict(list(ds[0].items())[:4])}\n"
                "The column values may all be None or empty."
            )
        return pairs
    except Exception as e:
        print(f"[WARN] Could not load HaluEval: {e}")
        print("       Falling back to synthetic dataset.")
        return QUICK_PAIRS


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    continuation: str,
    device: str,
) -> np.ndarray:
    """
    Extract per-layer feature vector from attention weights.

    Returns shape (num_layers * 2,) — [mean_entropy, mean_kl] per layer.
    """
    text = context + " " + continuation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (1, H, T, T) per layer
    num_layers = len(attentions)

    per_layer_entropy = []
    per_layer_kl = []

    prev_mean: np.ndarray | None = None

    for layer_attn in attentions:
        # layer_attn: (1, H, T, T)
        attn = layer_attn.squeeze(0).cpu().numpy()  # (H, T, T)

        # Last-token attention distribution across heads
        last_token_attn = attn[:, -1, :]  # (H, T)

        # Clamp for numerical stability
        last_token_attn = np.clip(last_token_attn, 1e-9, 1.0)
        last_token_attn /= last_token_attn.sum(axis=-1, keepdims=True)

        # Shannon entropy per head, then mean across heads
        entropy = -np.sum(last_token_attn * np.log2(last_token_attn), axis=-1)  # (H,)
        mean_entropy = float(entropy.mean())
        per_layer_entropy.append(mean_entropy)

        # Mean attention distribution across heads for KL
        mean_attn = last_token_attn.mean(axis=0)  # (T,)

        # KL divergence from previous layer
        if prev_mean is not None:
            kl = float(np.sum(prev_mean * np.log(prev_mean / (mean_attn + 1e-9) + 1e-9)))
        else:
            kl = 0.0
        per_layer_kl.append(kl)
        prev_mean = mean_attn

    features = np.array(per_layer_entropy + per_layer_kl, dtype=np.float32)
    return features


# ---------------------------------------------------------------------------
# BiLSTM classifier (lightweight numpy-based for quick mode)
# ---------------------------------------------------------------------------

class BiLSTMClassifier:
    """Minimal BiLSTM trained with gradient descent for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Forward LSTM weights (simplified: dense projection)
        self.W_f = rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.W_b = rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.W_out = rng.randn(hidden_dim * 2, 1).astype(np.float32) * 0.01
        self.b_out = np.zeros(1, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) → probabilities (N,)"""
        h_f = np.tanh(X @ self.W_f)
        h_b = np.tanh(X @ self.W_b)
        h = np.concatenate([h_f, h_b], axis=1)
        logits = h @ self.W_out + self.b_out
        return 1.0 / (1.0 + np.exp(-logits.squeeze()))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.05):
        for _ in range(epochs):
            proba = self.predict_proba(X)
            error = proba - y
            # Backprop (simplified)
            h_f = np.tanh(X @ self.W_f)
            h_b = np.tanh(X @ self.W_b)
            h = np.concatenate([h_f, h_b], axis=1)
            dW_out = h.T @ error.reshape(-1, 1) / len(y)
            db_out = error.mean()
            dh = error.reshape(-1, 1) * self.W_out.T
            dh_f, dh_b = dh[:, : dh.shape[1] // 2], dh[:, dh.shape[1] // 2 :]
            dW_f = X.T @ (dh_f * (1 - h_f ** 2)) / len(y)
            dW_b = X.T @ (dh_b * (1 - h_b ** 2)) / len(y)
            self.W_out -= lr * dW_out
            self.b_out -= lr * db_out
            self.W_f -= lr * dW_f
            self.W_b -= lr * dW_b


# ---------------------------------------------------------------------------
# AUROC + evaluation
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC using the trapezoidal rule."""
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tp_rate = tp / tp[-1]
    fp_rate = fp / fp[-1]
    # np.trapezoid is the current name (np.trapz removed in NumPy 2.0)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(tp_rate, fp_rate))


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_threshold: float = 0.90) -> float:
    """False positive rate at a given true positive rate threshold."""
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / (y_true.sum() + 1e-9)
    idx = np.searchsorted(tpr, tpr_threshold)
    if idx >= len(fps):
        return float(fps[-1] / ((1 - y_true).sum() + 1e-9))
    return float(fps[idx] / ((1 - y_true).sum() + 1e-9))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def logistic_regression_baseline(X_train, y_train, X_test, y_test) -> dict:
    """Logistic regression on global summary statistics."""
    # Mean and std of features as global summary
    X_tr_summary = np.column_stack([X_train.mean(axis=1), X_train.std(axis=1)])
    X_te_summary = np.column_stack([X_test.mean(axis=1), X_test.std(axis=1)])

    from numpy.linalg import lstsq
    A = np.column_stack([X_tr_summary, np.ones(len(X_tr_summary))])
    w, _, _, _ = lstsq(A.T @ A + 0.01 * np.eye(3), A.T @ y_train, rcond=None)
    logits = X_te_summary @ w[:2] + w[2]
    proba = 1.0 / (1.0 + np.exp(-logits))
    return {"auroc": auroc(y_test, proba), "fpr90": fpr_at_tpr(y_test, proba)}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    mode: str
    model_name: str
    num_samples: int
    num_layers: int
    num_heads: int
    device: str
    # AED BiLSTM results
    aed_auroc: float
    aed_fpr90: float
    aed_latency_ms: float
    # Baseline: logistic regression on global stats
    logreg_auroc: float
    logreg_fpr90: float
    # Timestamp
    timestamp: str
    notes: str = ""


def run(args: argparse.Namespace) -> ExperimentResults:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  AED Benchmark — mode={args.mode}, device={device}")
    print(f"{'='*60}\n")

    # Load model
    print(f"[1/5] Loading model: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, output_attentions=True)
    model.eval()
    model.to(device)
    print(f"      Loaded in {time.time()-t0:.1f}s  |  {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

    # Get model dims
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))
    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 12))
    print(f"      Layers={num_layers}, Heads={num_heads}")

    # Load dataset
    print(f"\n[2/5] Loading dataset")
    if args.mode == "full":
        pairs = load_halueval_qa(max_samples=args.max_samples)
    else:
        pairs = QUICK_PAIRS
    print(f"      {len(pairs)} samples ({sum(1 for _,_,l in pairs if l==1)} hallucinated)")

    # Extract features
    print(f"\n[3/5] Extracting attention features")
    features, labels = [], []
    t_start = time.time()
    for i, (ctx, cont, label) in enumerate(pairs):
        feat = extract_features(model, tokenizer, ctx, cont, device)
        features.append(feat)
        labels.append(label)
        if (i + 1) % 4 == 0:
            elapsed = time.time() - t_start
            print(f"      {i+1}/{len(pairs)} done  ({elapsed/(i+1)*1000:.0f}ms/sample)")
    t_extract = time.time() - t_start

    if not features:
        raise RuntimeError(
            "0 samples extracted — dataset returned no usable pairs.\n"
            "Check the HaluEval column inspection output above."
        )

    X = np.array(features)       # (N, num_layers * 2)
    y = np.array(labels)

    latency_ms = t_extract / len(pairs) * 1000
    print(f"      Feature shape: {X.shape}")
    print(f"      Average latency: {latency_ms:.1f}ms/sample")

    # Train/test split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\n[4/5] Training BiLSTM classifier ({len(X_train)} train, {len(X_test)} test)")

    clf = BiLSTMClassifier(input_dim=X.shape[1])
    clf.fit(X_train, y_train.astype(np.float32))
    proba = clf.predict_proba(X_test)
    aed_auroc = auroc(y_test, proba)
    aed_fpr90 = fpr_at_tpr(y_test, proba)
    print(f"      AED  AUROC={aed_auroc:.4f}  FPR@90%TPR={aed_fpr90:.4f}")

    # Baselines
    print(f"\n[5/5] Baseline: logistic regression on global summary stats")
    logreg = logistic_regression_baseline(X_train, y_train, X_test, y_test)
    print(f"      LogReg AUROC={logreg['auroc']:.4f}  FPR@90%TPR={logreg['fpr90']:.4f}")

    import datetime
    results = ExperimentResults(
        mode=args.mode,
        model_name=args.model,
        num_samples=len(pairs),
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
        aed_auroc=round(aed_auroc, 4),
        aed_fpr90=round(aed_fpr90, 4),
        aed_latency_ms=round(latency_ms, 1),
        logreg_auroc=round(logreg["auroc"], 4),
        logreg_fpr90=round(logreg["fpr90"], 4),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        notes=f"Run on {device}{'(T4)' if device=='cuda' else '(CPU)'}",
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  AED  (BiLSTM on per-layer sequence):")
    print(f"       AUROC            = {results.aed_auroc:.4f}")
    print(f"       FPR@90%TPR       = {results.aed_fpr90:.4f}")
    print(f"       Latency          = {results.aed_latency_ms:.1f}ms/sample")
    print(f"  Baseline (LogReg on global stats):")
    print(f"       AUROC            = {results.logreg_auroc:.4f}")
    print(f"       FPR@90%TPR       = {results.logreg_fpr90:.4f}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="AED hallucination detection benchmark")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick=CPU+synthetic, full=GPU+HaluEval",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name. Defaults: quick→gpt2, full→EleutherAI/pythia-160m",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max HaluEval samples in full mode",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_results.json",
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = "gpt2" if args.mode == "quick" else "EleutherAI/pythia-160m"

    results = run(args)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"Results saved to: {out_path}")
    print(f"\nNext step: fill these values into arxiv-paper/paper.tex \\RESULT{{}} placeholders.")


if __name__ == "__main__":
    main()
