# LLM Hallucination Detection via Attention Entropy

**Real-time detection of unreliable LLM outputs using information-theoretic analysis of transformer attention patterns.**

Built for safety-critical systems — robotics, medical AI, autonomous agents — where trusting a hallucinated output has real consequences.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Large language models hallucinate. They generate confident-sounding text that is factually wrong, internally inconsistent, or completely fabricated. In conversational settings this is annoying; in safety-critical systems it's dangerous.

**This project detects hallucinations in real-time** by analyzing what's happening inside the model's attention mechanism during generation — before the output reaches downstream systems.

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| AUROC | > 0.70 | **0.999** |
| False Positive Rate | < 5% | **0.0%** |
| Latency (p95) | < 5ms | **0.098ms** |
| Throughput | — | **12,242 samples/sec** |

*Benchmarked on 500 synthetic samples with the hypothesis test + calibrator pipeline. Real-world performance with GPT-2 attention extraction will have higher latency due to the forward pass.*

---

## How It Works

### Core Insight

When a transformer is confident about its prediction, attention heads converge on specific tokens with low entropy. When it's uncertain or fabricating, attention becomes diffuse (high entropy) and layers disagree with each other (high KL divergence).

We measure both signals and apply a statistical hypothesis test to decide: **is this output reliable?**

### Mathematical Foundation

#### 1. Attention Entropy (per-head uncertainty)

For each attention head, we extract the last token's attention distribution and compute Shannon entropy:

```
H(head) = -Σᵢ p(i) · log₂(p(i))
```

- **Low entropy** → peaked attention → model knows what to attend to → likely reliable
- **High entropy** → diffuse attention → model is "looking everywhere" → possibly hallucinating

#### 2. Cross-Layer KL Divergence (internal consistency)

We measure how much consecutive layers disagree about where to attend:

```
D_KL(layer_l || layer_{l+1}) = Σᵢ p_l(i) · log(p_l(i) / p_{l+1}(i))
```

- **Low KL** → layers agree → stable internal representation → reliable
- **High KL** → layers contradict each other → unstable → hallucination risk

#### 3. Hypothesis Test (statistical decision)

We formulate detection as a one-tailed Z-test:

```
H₀: output is RELIABLE
H₁: output is UNRELIABLE (hallucination)

Z_combined = 0.40·Z_entropy + 0.45·Z_kl + 0.15·Z_spread

where Z_x = (observed_x - μ_baseline) / σ_baseline
```

The KL divergence signal gets the highest weight (0.45) because cross-layer disagreement is the strongest hallucination indicator.

#### 4. Confidence Calibration (isotonic regression)

Raw confidence scores are post-hoc calibrated using the Pool Adjacent Violators (PAV) algorithm so that "80% confident" actually means 80% correct:

```
f: raw_score → calibrated_score  (monotonic, via isotonic regression)
```

#### 5. Three-Tier Routing

```
confidence > 0.75  →  ✅ RELIABLE     →  output_to_user
0.50 ≤ conf ≤ 0.75 →  ⚠️  UNCERTAIN   →  escalate_to_human
confidence < 0.50  →  ❌ UNRELIABLE   →  reject_use_fallback
```

Conservative thresholds: the wide uncertain zone biases toward safety.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Prompt                          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  AttentionAnalyzer                                       │
│  ┌────────────────┐  ┌──────────────────────────────┐   │
│  │ GPT-2 Forward  │→ │ Entropy + KL Computation     │   │
│  │ Pass           │  │ H = -Σ p·log₂(p)             │   │
│  │ (output_       │  │ D_KL = Σ p·log(p/q)          │   │
│  │  attentions)   │  └──────────────┬───────────────┘   │
│  └────────────────┘                 │                    │
│                    AttentionAnalysisResult                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  HallucinationHypothesisTest                             │
│  ┌────────────────┐  ┌──────────────────────────────┐   │
│  │ Z-score        │→ │ P-value + Decision            │   │
│  │ Standardize    │  │ p = 1 - Φ(Z_combined)        │   │
│  │ vs baseline    │  │ reject H₀ if Z > z_critical  │   │
│  └────────────────┘  └──────────────┬───────────────┘   │
│                     HypothesisTestResult                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  ConfidenceCalibrator                                    │
│  ┌────────────────┐  ┌──────────────────────────────┐   │
│  │ Isotonic       │→ │ Three-Tier Routing            │   │
│  │ Calibration    │  │ RELIABLE / UNCERTAIN /        │   │
│  │ (PAV)          │  │ UNRELIABLE                    │   │
│  └────────────────┘  └──────────────┬───────────────┘   │
│                      CalibrationDecision                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  Output Action  │
              │  ✅ / ⚠️ / ❌    │
              └─────────────────┘
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/hallucination-detection.git
cd hallucination-detection
pip install -r requirements.txt
```

### 2. Download GPT-2

```bash
python download_model.py
```

### 3. Run Demo

```bash
# Full mode (requires GPT-2)
python run_demo.py

# Custom prompt
python run_demo.py --prompt "The capital of Mars is"

# Synthetic mode (no model needed — tests the statistical pipeline)
python run_demo.py --synthetic
```

### 4. Run Evaluation

```bash
python evaluate.py --num_samples 1000
```

### 5. Run Tests

```bash
# All tests (skips torch-dependent tests if torch unavailable)
pytest tests/ -v

# Fast: numpy-only tests
pytest tests/ -v -k "not torch and not slow"
```

---

## Project Structure

```
hallucination-detection/
├── src/
│   ├── __init__.py
│   ├── attention_analyzer.py    # Core: entropy + KL from attention tensors
│   ├── hypothesis_test.py       # Statistical Z-test framework
│   ├── confidence_calibrator.py # Isotonic calibration + 3-tier routing
│   └── utils.py                 # Logging, timing, serialization
├── tests/
│   ├── __init__.py
│   ├── test_attention_analyzer.py  # 32 tests: entropy, KL, stability
│   ├── test_hypothesis_test.py     # 34 tests: Z-scores, p-values, boundaries
│   └── test_integration.py         # 20 tests: full pipeline, serialization
├── download_model.py            # One-time model download
├── run_demo.py                  # Interactive demo
├── evaluate.py                  # Benchmarking + metrics
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Module Deep Dives

### `attention_analyzer.py`

Loads GPT-2, runs a forward pass with `output_attentions=True`, and computes:

- **Layer-head entropy matrix** `(L, H)`: Shannon entropy of the last token's attention row per head per layer
- **Cross-layer KL divergence**: pairwise KL between consecutive layers' averaged attention distributions
- **Summary statistics**: mean/max/std entropy, total/max KL

Standalone functions `compute_entropy_from_weights()` and `compute_kl_divergence()` work with raw numpy arrays — no model needed.

### `hypothesis_test.py`

Frames hallucination detection as a statistical test:

- **Null hypothesis H₀**: the output is reliable
- **Test statistic**: weighted combination of standardized entropy, KL, and entropy spread
- **Decision**: reject H₀ (flag as unreliable) if Z > z_critical at significance level α

Supports custom calibration baselines via `calibrate_from_corpus()` for domain adaptation.

### `confidence_calibrator.py`

Two components:

1. **IsotonicCalibrator**: Pure-numpy implementation of the PAV algorithm (Zadrozny & Elkan, 2002). Learns a monotonic mapping from raw scores to calibrated probabilities.
2. **ConfidenceCalibrator**: Applies calibration and routes to one of three actions based on configurable thresholds.

### `utils.py`

Production helpers: structured logging, `Timer` context manager (perf_counter precision), tokenization with lazy loading, batch iteration, and JSON serialization with full numpy type support.

---

## Test Suite

**86 total tests** (73 pass in CI without torch, all 86 pass locally with torch):

| File | Tests | What's Covered |
|------|-------|----------------|
| `test_attention_analyzer.py` | 32 | Entropy bounds, KL properties, Gibbs' inequality, numerical stability, batch shapes |
| `test_hypothesis_test.py` | 34 | Z-score computation, p-value properties, decision boundaries, weight sensitivity, edge cases |
| `test_integration.py` | 20 | Full pipeline end-to-end, serialization roundtrip, ECE measurement, decision distribution |

Key mathematical properties verified:
- Entropy is non-negative and bounded by log₂(T)
- Entropy is monotonically increasing with uniformity
- KL divergence satisfies Gibbs' inequality (D_KL ≥ 0)
- KL divergence is asymmetric
- P-values are in [0, 1] with p = 0.5 at baseline
- Confidence monotonically decreases with Z-score
- All three routing labels are reachable
- Isotonic calibration produces monotonic outputs

---

## Design Decisions

**Why attention entropy?**
Interpretable, fast (computed from existing forward pass), and theoretically grounded in information theory. The tradeoff: won't catch high-confidence hallucinations where the model is "confidently wrong."

**Why KL divergence across layers?**
Captures internal inconsistency: when layer 5's attention contradicts layer 6, the model is wavering between different representations. This is a stronger hallucination signal than entropy alone, which is why it gets the highest weight (0.45).

**Why conservative thresholds?**
For safety-critical deployment: it's better to escalate a reliable output to a human (false positive) than to trust an unreliable one (false negative). The wide UNCERTAIN zone reflects this bias.

**Why isotonic regression over Platt scaling?**
Isotonic regression makes no parametric assumptions about the score distribution. Platt scaling assumes a sigmoid relationship, which doesn't hold for Z-score-based confidence. Isotonic also has a clean O(n) PAV implementation.

**Why GPT-2?**
Publicly available, well-understood architecture, and small enough to run on a CPU. The technique generalizes to any transformer with accessible attention weights.

---

## Extending This Work

- **Domain calibration**: Use `calibrate_from_corpus()` to adapt baselines to your specific model and domain
- **Multi-model**: Swap GPT-2 for any HuggingFace model that exposes attention weights
- **Streaming**: The `<5ms` decision latency supports real-time token-by-token monitoring
- **Ensemble**: Combine attention entropy with output probability, embedding similarity, or retrieval-augmented verification

---

## References

- Shannon, C.E. (1948). *A Mathematical Theory of Communication*
- Kullback, S. & Leibler, R.A. (1951). *On Information and Sufficiency*
- Zadrozny, B. & Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*
- Vaswani, A. et al. (2017). *Attention Is All You Need*
- Xiao, Y. & Wang, W.Y. (2021). *On Hallucination and Predictive Uncertainty in Conditional Language Generation*

---

## License

MIT
