# Natural Hallucination Analysis

Two implementations of LLM hallucination detection via attention analysis.

**Default local model:** [Google Gemma 2 (2B)](https://huggingface.co/google/gemma-2-2b) — public, compact, supports `output_attentions`.
**LLM-as-judge:** Claude (Anthropic API) for QA generation and answer labeling.

| | v1 | v2 |
|---|---|---|
| **Approach** | Statistical hypothesis test | Trained classifier |
| **Features** | Entropy + KL divergence | 5 feature families (18D) |
| **Labels** | Hand-tuned baseline | LLM-as-judge (Claude) |
| **Local Model** | Gemma 2 (2B) | Any HuggingFace model |
| **Folder** | [`v1/`](v1/README.md) | [`v2/`](v2/README.md) |

---

## Key Results

### v1 — Hypothesis Test Pipeline

| Metric | Target | Achieved |
|--------|--------|----------|
| AUROC | > 0.70 | **0.999** |
| False Positive Rate | < 5% | **0.0%** |
| Latency (p95) | < 5ms | **0.098ms** |
| Throughput | -- | **12,242 samples/sec** |

*Benchmarked on 500 synthetic samples with the hypothesis test + calibrator pipeline.*

### v2 — Multi-Family Classifier

| Metric | Target | Achieved |
|--------|--------|----------|
| AUROC | > 0.85 | **0.999** |
| F1 | > 0.70 | **0.987** |
| FPR | < 10% | **0.0%** |

*Benchmarked on 500 synthetic samples (150 test) with logistic regression on 18D feature vector.*

---

## Model Migration: GPT-2 -> Gemma 2

All code previously defaulting to GPT-2 (124M) now uses **Google Gemma 2 (2B)** as the default local model. Key changes:

- Model config access is now generic (`num_hidden_layers` / `num_attention_heads` with fallback)
- All defaults, docstrings, CLI arguments, and documentation updated
- The technique is model-agnostic: pass any HuggingFace model ID via `model_name=`

---

## v1 — Attention Entropy + Hypothesis Test

[`v1/README.md`](v1/README.md) · [`v1/claude1.md`](v1/claude1.md)

Real-time detection using Shannon entropy and cross-layer KL divergence, with a Z-test decision framework and isotonic calibration.

## v2 — Multi-Family Features + Classifier

[`v2/README.md`](v2/README.md) · [`v2/claude2.md`](v2/claude2.md)

Research-grade detection using five feature families from 2024–2026 papers (Lookback Lens, frequency-domain, spectral Laplacian) with a self-labeled training pipeline.
