# Natural Hallucination Analysis

[![Tests](https://github.com/A-Kuo/Natural-Hallucination-Analysis/actions/workflows/test.yml/badge.svg)](https://github.com/A-Kuo/Natural-Hallucination-Analysis/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![v1 Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Natural-Hallucination-Analysis/blob/main/colab/v1_benchmark.ipynb)
[![v2 Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Natural-Hallucination-Analysis/blob/main/colab/v2_full_pipeline.ipynb)

**LLM hallucination detection via attention pattern analysis.** Two implementations — a lightweight statistical approach (v1) and a research-grade trained classifier (v2) — built for safety-critical applications.

Uses **open-source models only** (EleutherAI Pythia, Llama, Mistral) and Anthropic Claude for labeling. No OpenAI models or APIs.

---

## Quick Start

```bash
git clone https://github.com/A-Kuo/Natural-Hallucination-Analysis.git
cd Natural-Hallucination-Analysis

# v1 — statistical detection
cd v1 && pip install -r requirements.txt
pytest test_attention_analyzer.py -v

# v2 — trained classifier
cd ../v2 && pip install -r requirements.txt
python pipeline.py --synthetic --num_samples 1000
```

---

## Implementations

| | **v1** — Entropy + Hypothesis Test | **v2** — BiLSTM + Multi-Family Features |
|---|---|---|
| **Approach** | Statistical Z-test on entropy + KL | BiLSTM on per-layer attention sequences |
| **Features** | 2 families (entropy, KL divergence) | 5 families: 18D flat + L×6 sequence |
| **Classifier** | Hypothesis test + isotonic calibration | BiLSTM / LogReg / MLP |
| **AUROC** | 0.999 (synthetic) | **0.96** (BiLSTM, HaluEval) |
| **Default model** | EleutherAI/pythia-160m | EleutherAI/pythia-160m |
| **Labels** | Hand-tuned baseline thresholds | Claude as LLM-as-judge |
| **Robustness** | — | Obfuscation · Paraphrase · Multilingual |
| **Anomaly detection** | — | ChromaDB + centroid/Mahalanobis |
| **Deployment** | Docker | GCP Vertex AI (online + batch) |
| **Docs** | [`v1/README.md`](v1/README.md) | [`v2/README.md`](v2/README.md) |
| **Agent** | [`v1/AGENT.md`](v1/AGENT.md) | [`v2/AGENT.md`](v2/AGENT.md) |

---

## v1 Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| AUROC | > 0.70 | **0.999** |
| False Positive Rate | < 5% | **0.0%** |
| Latency (p95) | < 5ms | **0.098ms** |
| Throughput | — | **12,242 samples/sec** |

*500 synthetic samples, hypothesis test + calibrator pipeline.*

---

## Architecture Overview

**v1** applies information-theoretic measures directly to attention tensors:

```
Input → AttentionAnalyzer (entropy + KL) → HypothesisTest → ConfidenceCalibrator → RELIABLE / UNCERTAIN / UNRELIABLE
```

**v2** extracts a richer feature vector and trains a classifier:

```
Text → Model (attentions) → FeatureEngineer (18D) → Detector (LogReg / MLP) → hallucination probability
```

---

## Feature Families (v2)

| Family | Source | Dim |
|--------|--------|-----|
| Shannon Entropy | v1 baseline | 3 |
| Lookback Ratio | Chuang et al., EMNLP 2024 | 4 |
| Frequency Domain | Qi et al., 2026 | 4 |
| Spectral / Laplacian | Barbero et al., 2025 | 4 |
| Cross-Layer KL | v1 baseline | 3 |

---

## Repository Structure

```
├── v1/                     # Statistical detection
│   ├── attention_analyzer.py
│   ├── hypothesis_test.py
│   ├── confidence_calibrator.py
│   ├── utils.py
│   ├── test_attention_analyzer.py
│   ├── Dockerfile
│   ├── README.md
│   └── AGENT.md
├── v2/                     # Trained classifier
│   ├── data_generator.py
│   ├── feature_engineer.py
│   ├── detector.py
│   ├── pipeline.py
│   ├── README.md
│   └── AGENT.md
├── .github/workflows/      # CI
│   └── test.yml
├── pyproject.toml
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{kuo2025hallucination,
  title  = {Natural Hallucination Analysis},
  author = {Kuo, A},
  year   = {2025},
  url    = {https://github.com/A-Kuo/Natural-Hallucination-Analysis}
}
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
