# LLM Hallucination Detection via Attention Entropy

**Real-time detection of unreliable LLM outputs using information-theoretic analysis of transformer attention patterns.**

Built for safety-critical systems — robotics, medical AI, autonomous agents — where trusting a hallucinated output has real consequences.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
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

*Benchmarked on 500 synthetic samples with the hypothesis test + calibrator pipeline.*

---

## How It Works

When a transformer is confident about its prediction, attention heads converge on specific tokens with low entropy. When it's uncertain or fabricating, attention becomes diffuse (high entropy) and layers disagree with each other (high KL divergence).

We measure both signals and apply a statistical hypothesis test to decide: **is this output reliable?**

- **Attention entropy** — Shannon entropy of the last token's attention distribution per head
- **Cross-layer KL divergence** — disagreement between consecutive layers
- **Hypothesis test** — Z-score combination with configurable baseline
- **Confidence calibration** — isotonic regression (PAV) for calibrated probabilities
- **Three-tier routing** — RELIABLE / UNCERTAIN / UNRELIABLE

---

## Architecture

```
Input → AttentionAnalyzer (entropy + KL) → HypothesisTest → ConfidenceCalibrator → Output (✅/⚠️/❌)
```

---

## Quickstart

```bash
cd v1/
pip install -r requirements.txt
pytest test_attention_analyzer.py -v
```

*For full demo (Gemma 2 forward pass), see `v1/claude1.md` for module details and extension points.*

---

## Project Structure

```
v1/
├── attention_analyzer.py    # Entropy + KL from attention tensors
├── hypothesis_test.py       # Statistical Z-test framework
├── confidence_calibrator.py # Isotonic calibration + 3-tier routing
├── utils.py
├── test_attention_analyzer.py
├── requirements.txt
├── Dockerfile
├── README.md
└── claude1.md               # Agent instructions
```

---

## License

MIT
