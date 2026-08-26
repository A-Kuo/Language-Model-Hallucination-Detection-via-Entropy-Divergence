# Hallucination Detection v2 — Entropy Statistics & Calibrated Divergence

**Detecting hallucinations in LLM outputs using single-pass entropy statistics and divergence from calibrated entropy distributions**, alongside multi-family attention features, embedding anomaly detection, and adversarial robustness evaluation.

Scaled successor to v1. v2 trains classifiers on multi-family attention + token-entropy features extracted from any open-weight model, labeled by Claude as LLM-as-judge — plus a calibrated entropy-divergence detector (this repo's main original contribution) and a black-box detector that works with only the top-K logprobs a commercial completion API exposes.

| Metric | Value |
|--------|-------|
| **AUROC (Logistic Regression, HaluEval)** | **0.91** |
| Classifier | Logistic regression (primary baseline) |
| Feature input | 18D flat attention-family vector |
| Sequence baseline | BiLSTM on per-layer sequences (L × 6) — AUROC 0.78, underperforms LogReg |
| Deployment | GCP Vertex AI online + batch endpoints |

---

## Detectors

| Detector | File | Access needed | Notes |
|---|---|---|---|
| Logistic Regression | `detector.py` | White-box (attention) | Primary/strongest baseline, ~0.91 AUROC |
| MLP | `detector.py` | White-box (attention) | Captures nonlinear feature interactions |
| BiLSTM | `detector.py` | White-box (attention) | Per-layer sequence model; currently ~0.78 AUROC, underperforms LogReg |
| **CalibratedEntropyDetector** | `calibrated_entropy_detector.py` | White-box (entropy and/or attention features) | **Main original contribution.** Calibrates raw entropy features against the distribution they take on correct answers (isotonic regression) and scores new examples by Mahalanobis divergence from that reference, instead of thresholding raw entropy directly |
| **BlackBoxEntropyDetector** | `blackbox_detector.py` | **Top-K logprobs only** | The only detector usable against a commercial completions API (e.g. OpenAI's `top_logprobs`) — no attention weights or full-vocab logits required |

Established single-pass token-entropy baselines (mean/max/std entropy, perplexity, top-k entropy, confidence margin) live in `entropy_baselines.py` and feed both the flat classifiers above and `CalibratedEntropyDetector`.

An abstention/selective-prediction experiment (`abstention.py`) shows the practical payoff of calibration: refusing to answer the highest-risk examples raises accuracy on the examples that are answered, reported as a risk-coverage curve.

---

## What Changed from v1

| | v1 | v2 |
|---|---|---|
| **Model** | Pythia only | Any HuggingFace model (Llama, Mistral, Phi, Pythia) |
| **Features** | 2 families (entropy, KL) | 5 attention families + single-pass token-entropy baselines |
| **Labels** | Hand-tuned Z-test | LLM-as-judge (Claude) on self-generated QA |
| **Classifier** | Hypothesis test | Logistic regression / MLP / BiLSTM / CalibratedEntropyDetector / BlackBoxEntropyDetector |
| **Data** | None | Self-generated, scales with API budget |
| **Black-box support** | None | `BlackBoxEntropyDetector` — top-K logprobs only |
| **Calibration** | None | `CalibratedEntropyDetector` — isotonic + Mahalanobis divergence |
| **Selective prediction** | None | `abstention.py` — risk-coverage curve |

---

## Feature Families

**Attention-based (white-box, full model access):**
1. **Shannon Entropy** (v1) — attention diffuseness per head
2. **Lookback Ratio** (Chuang et al., EMNLP 2024) — context vs. generation attention
3. **Frequency Domain** (Qi et al., 2026) — DFT high-frequency energy
4. **Spectral / Laplacian** (Barbero et al.) — Fiedler value, graph connectivity
5. **Cross-Layer KL** (v1) — layer disagreement

**Token-entropy-based (white-box, logits only — no attention needed):**
6. **Single-pass predictive entropy** (`entropy_baselines.py`) — mean/max/std Shannon entropy of the next-token distribution, perplexity, top-k-renormalized entropy, top1/top2 confidence margin

**Top-K-logprob-based (black-box, works with commercial APIs):**
7. **Top-K entropy lower bound** (`blackbox_detector.py`) — the same estimators as above, computed only from the small top-K logprob list a completions API actually returns

---

## Quickstart

```bash
cd v2/
pip install -r requirements.txt

# Synthetic demo (no model/API)
python pipeline.py --synthetic --num_samples 1000

# HaluEval benchmark (no API — pip install datasets), trains all detectors
# and prints AUROC/CI for LogReg, MLP, BiLSTM, CalibratedEntropyDetector,
# and BlackBoxEntropyDetector side by side
python pipeline.py --halueval --num_samples 500 --model EleutherAI/pythia-160m

# Abstention / selective-prediction experiment
python abstention.py --synthetic --num_samples 1000
python pipeline.py --synthetic --num_samples 1000 --abstention

# Full pipeline (requires ANTHROPIC_API_KEY)
python pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m --save detector.pkl
```

Default local model is [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m). For better hallucination rates, use larger models like Llama or Mistral.

**Black-box demo (real API):** `blackbox_detector.py::fetch_topk_logprobs_openai()` calls OpenAI's `chat.completions` with `logprobs=True, top_logprobs=5` for a live demo (requires `pip install openai` and `OPENAI_API_KEY`). Everything else — including all tests and the `--synthetic`/`--halueval` pipeline modes — uses `simulate_topk_from_full_logits()`, an offline path that derives simulated top-K logprobs from local model logits, so no network access or API key is required to reproduce results.

**Adversarial robustness:** Tested against obfuscation (character substitution), paraphrase (synonym replacement), and multilingual (Spanish/French/German/Japanese prefix) attacks. Stability > 80% across all attack types.

**Embedding anomaly detection:** ChromaDB vector store + sentence-transformers; centroid distance and Mahalanobis distance from correct-answer embedding distribution. Ensembled with attention score: `0.6 × attn + 0.4 × embedding`.

**Deployment:** Vertex AI online endpoint (REST, autoscaling) and batch prediction (JSONL → GCS). See `vertex_deploy.py`.

**Tests:** `pytest v2/tests` (also wired into the repo-root `pyproject.toml` testpaths). Every detector and feature-extraction module has unit coverage; torch/openai-dependent tests skip cleanly when those optional dependencies aren't installed.

*See [`v2/AGENT.md`](AGENT.md) for implementation details, known limitations, and research foundations.*

---

## Project Structure

```
v2/
├── data_generator.py               # Self-data via Anthropic API + HaluEval loader
├── feature_engineer.py             # 5 attention families → 18D vector + per-layer sequence (L×6)
├── entropy_baselines.py            # Single-pass token-entropy features (6D) from output logits
├── detector.py                     # LogReg / MLP / BiLSTM classifiers + shared metrics helper
├── calibrated_entropy_detector.py  # Main contribution: isotonic calibration + Mahalanobis divergence
├── blackbox_detector.py            # Top-K logprob detector (real API + offline simulation)
├── abstention.py                   # Risk-coverage / selective-prediction experiment
├── pipeline.py                     # End-to-end: stratified k-fold, bootstrap CIs, ablation
├── adversarial.py                  # Robustness: obfuscation, paraphrase, multilingual
├── embedding_anomaly.py            # ChromaDB + centroid/Mahalanobis anomaly detection
├── vertex_deploy.py                # GCP Vertex AI deployment (online + batch)
├── tests/                          # pytest suite for every module above
├── README.md
├── AGENT.md
└── requirements.txt
```

---

## Feature Vector Reference

**Attention families (18D):**

| Family | Features | Dim |
|--------|----------|-----|
| Entropy | mean, max, std | 3 |
| Lookback | ratio_mean, min, std, entropy | 4 |
| Frequency | high_freq_mean, max, centroid, spectral_entropy | 4 |
| Spectral | fiedler_mean, std, spectral_gap, laplacian_energy | 4 |
| Cross-Layer KL | total, max, std | 3 |

**Token-entropy baseline (6D, `entropy_baselines.py`):** `entropy_mean`, `entropy_max`, `entropy_std`, `perplexity`, `topk_entropy_mean`, `margin_mean`

**Black-box top-K (7D, `blackbox_detector.py`):** `topk_entropy_mean`, `topk_entropy_max`, `topk_entropy_std`, `topk_mass_mean`, `topk_mass_min`, `margin_mean`, `margin_min`

---

## Known Discrepancy

The repo-root `README.md` describes an aspirational, purely token-probability-based API (`EntropyScorer`, `KLDivergenceScorer`, `LayerEntropyAnalyzer`, `IsotonicCalibrator`) that predates v2 and does not match this package's actual classes. `entropy_baselines.py` and `calibrated_entropy_detector.py` now provide real, tested implementations of that same underlying idea (single-pass token entropy + isotonic calibration) under different names — reconciling the root README's class names/checklist against what actually exists here is a good follow-up, tracked separately from this module's own docs.

---

## License

[MIT](../LICENSE)
