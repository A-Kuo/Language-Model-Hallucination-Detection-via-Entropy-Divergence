# Hallucination Detection v2 — Multi-Family Attention Features

**Research-grade hallucination detection using self-generated labeled data and five feature families from cutting-edge papers.**

Scaled successor to v1. v2 eliminates hand-tuned baselines by training a lightweight classifier on features from any open-weight model, labeled by an LLM-as-judge (Claude).

---

## What Changed from v1

| | v1 | v2 |
|---|---|---|
| **Model** | Pythia only | Any HuggingFace model (Llama, Mistral, Phi, Pythia) |
| **Features** | 2 families (entropy, KL) | 5 families (+ lookback, frequency, spectral) |
| **Labels** | Hand-tuned Z-test | LLM-as-judge (Claude) on self-generated QA |
| **Classifier** | Hypothesis test | Logistic regression / MLP |
| **Data** | None | Self-generated, scales with API budget |

---

## Feature Families

1. **Shannon Entropy** (v1) — attention diffuseness per head
2. **Lookback Ratio** (Chuang et al., EMNLP 2024) — context vs. generation attention
3. **Frequency Domain** (Qi et al., 2026) — DFT high-frequency energy
4. **Spectral / Laplacian** (Barbero et al.) — Fiedler value, graph connectivity
5. **Cross-Layer KL** (v1) — layer disagreement

---

## Quickstart

```bash
cd v2/
pip install -r requirements.txt

# Synthetic demo (no model/API)
python pipeline.py --synthetic --num_samples 1000

# Full pipeline (requires ANTHROPIC_API_KEY)
python pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m --save detector.pkl
```

Default local model is [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m). For better hallucination rates, use larger models like Llama or Mistral.

*See [`v2/AGENT.md`](AGENT.md) for implementation details, known limitations, and research foundations.*

---

## Project Structure

```
v2/
├── data_generator.py   # Self-data via Anthropic API
├── feature_engineer.py # 5 families → 18D vector
├── detector.py         # LogReg / MLP classifier
├── pipeline.py         # End-to-end orchestration
├── README.md
├── AGENT.md            # Agent instructions
└── requirements.txt
```

---

## Feature Vector (18D)

| Family | Features | Dim |
|--------|----------|-----|
| Entropy | mean, max, std | 3 |
| Lookback | ratio_mean, min, std, entropy | 4 |
| Frequency | high_freq_mean, max, centroid, spectral_entropy | 4 |
| Spectral | fiedler_mean, std, spectral_gap, laplacian_energy | 4 |
| Cross-Layer KL | total, max, std | 3 |

---

## License

[MIT](../LICENSE)
