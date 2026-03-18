# Natural Hallucination Analysis

Two implementations of LLM hallucination detection via attention analysis.

| | v1 | v2 |
|---|---|---|
| **Approach** | Statistical hypothesis test | Trained classifier |
| **Features** | Entropy + KL divergence | 5 feature families (18D) |
| **Labels** | Hand-tuned baseline | LLM-as-judge |
| **Folder** | [`v1/`](v1/README.md) | [`v2/`](v2/README.md) |

---

## v1 — Attention Entropy + Hypothesis Test

[`v1/README.md`](v1/README.md) · [`v1/claude1.md`](v1/claude1.md)

Real-time detection using Shannon entropy and cross-layer KL divergence, with a Z-test decision framework and isotonic calibration.

## v2 — Multi-Family Features + Classifier

[`v2/README.md`](v2/README.md) · [`v2/claude2.md`](v2/claude2.md)

Research-grade detection using five feature families from 2024–2026 papers (Lookback Lens, frequency-domain, spectral Laplacian) with a self-labeled training pipeline.
