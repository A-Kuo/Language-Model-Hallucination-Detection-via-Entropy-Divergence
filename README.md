# Language Model Hallucination Detection via Entropy Divergence

**Detecting hallucinations in LLM outputs using single-pass entropy statistics and divergence from calibrated entropy distributions.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![Status](https://img.shields.io/badge/Status-Beta-green.svg)]()

> *"A model that is confidently wrong is more dangerous than a model that admits uncertainty. The question is not whether LLMs hallucinate — they do. The question is whether we can measure the moment it happens."*

---

## The Core Problem

Large language models hallucinate. They generate fluent, confident text that is factually wrong, internally inconsistent, or fabricated. Detecting this at inference time — without ground-truth labels — is one of the central unsolved problems in deploying LLMs in high-stakes environments.

Existing detection approaches fall into two camps:

| Approach | Mechanism | Problem |
|----------|-----------|---------|
| **Output-text heuristics** | Keyword detection, hedging phrases, self-consistency checks | Model can be confidently wrong; hedging language is not a reliable hallucination signal |
| **Retrieval augmentation** | Ground every claim against a knowledge base | Doesn't work for reasoning tasks, synthesis, or domains without clean retrieval targets |
| **Human-in-the-loop** | Flag outputs for human review | Doesn't scale; doesn't give you a signal for *which* outputs to flag |

This project takes a different approach: **use the model's internal uncertainty — its output probability distribution and attention patterns — as a hallucination signal**, calibrated against a labeled reference set rather than thresholded ad hoc.

---

## Why Entropy Is a Better Signal

When a language model generates text, it doesn't just produce the most likely token — it maintains a probability distribution over its entire vocabulary at each step. The *shape* of that distribution is deeply informative.

**Low entropy** = the model's probability mass is concentrated. It has a strong, consistent prediction. Whether that prediction is correct is a separate question, but at minimum the model is not uncertain.

**High entropy** = probability mass is spread across many tokens. The model does not have a strong prediction. In factual domains, this correlates strongly with the model being outside its training distribution — i.e., hallucinating or confabulating.

This relationship is grounded in information theory: Shannon entropy H(p) = -Σ p(x) log p(x) is the expected surprise of a distribution. A model that "knows" the answer produces low-surprise next tokens. A model that is generating plausible-sounding text without grounded knowledge produces high-surprise distributions.

But raw entropy thresholds don't transfer across models or domains — a score of 0.4 might indicate hallucination in historical facts and be normal in creative writing. **This project's central contribution is calibration**: instead of thresholding raw entropy, it fits the distribution entropy actually takes on a labeled reference set of correct answers, then scores new examples by how far they diverge from that calibrated reference (see [`CalibratedEntropyDetector`](#detectors) below).

---

## Quick Start — Run in Colab

The fastest way to get paper-quality results (no local setup required):

| Notebook | Purpose | Runtime | Badge |
|----------|---------|---------|-------|
| **[GPU Benchmark](COLABS.md)** | Generate paper numbers (AUROC, FPR, latency) | ~15 min T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/gpu_benchmark.ipynb) |
| **[Ablation Study](COLABS.md)** | Fill Table 2 (entropy-only vs KL-only vs both) | ~8 min T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/ablation_study.ipynb) |
| **[Quick Validation](COLABS.md)** | Test pipeline on CPU (synthetic data) | ~3 min CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/quick_cpu_validation.ipynb) |

**Auto-commit:** Add `GH_TOKEN` to Colab Secrets → results push directly to repo
**See all:** [COLABS.md](COLABS.md) — full index with detailed instructions

---

## Detectors

| Detector | File | Access needed | Notes |
|---|---|---|---|
| Logistic Regression | `detector.py` | White-box (attention) | Primary/strongest baseline, ~0.91 AUROC on HaluEval |
| MLP | `detector.py` | White-box (attention) | Captures nonlinear feature interactions |
| BiLSTM | `detector.py` | White-box (attention) | Per-layer sequence model; currently ~0.78 AUROC, underperforms logistic regression |
| **CalibratedEntropyDetector** | `calibrated_entropy_detector.py` | White-box (entropy and/or attention features) | **Main original contribution.** Calibrates raw entropy features against the distribution they take on correct answers (isotonic regression) and scores new examples by Mahalanobis divergence from that reference, instead of thresholding raw entropy directly |
| **BlackBoxEntropyDetector** | `blackbox_detector.py` | **Top-K logprobs only** | The only detector usable against a commercial completions API (e.g. OpenAI's `top_logprobs`) — no attention weights or full-vocab logits required |

Established single-pass token-entropy baselines (mean/max/std entropy, perplexity, top-k entropy, confidence margin) live in `entropy_baselines.py` and feed both the flat classifiers above and `CalibratedEntropyDetector`.

An abstention/selective-prediction experiment (`abstention.py`) shows the practical payoff of calibration: refusing to answer the highest-risk examples raises accuracy on the examples that are answered, reported as a risk-coverage curve.

---

## Mathematical Approach

### Token-Level Entropy

For each generated token position *t*, the model produces a probability distribution p_t over the vocabulary V:

```
H_t = -Σ_{v ∈ V} p_t(v) · log p_t(v)
```

A sequence hallucination score is the mean or max token entropy over the generated span:

```
H_seq = mean(H_t for t in generated_tokens)
H_seq_max = max(H_t for t in generated_tokens)
```

High `H_seq` indicates the model was uncertain throughout the generation. High `H_seq_max` indicates at least one token was highly uncertain — useful for detecting *insertion* of a fabricated entity into an otherwise confident generation. Implemented in `entropy_baselines.py::compute_entropy_baseline_features`.

### Attention Entropy and Cross-Layer KL Divergence

A complementary white-box signal measures entropy and divergence over the model's *attention* patterns rather than its output distribution — implemented across five feature families in `feature_engineer.py` (Shannon entropy of attention, lookback ratio grounding context vs. generation, frequency-domain instability, spectral/Laplacian graph structure, and cross-layer KL divergence):

```
D_KL(p_layer_i || p_layer_j) = Σ_{v} p_layer_i(v) · log(p_layer_i(v) / p_layer_j(v))
```

Layers disagreeing strongly (high cross-layer KL) is a signature of internal inconsistency — the model's early-layer syntactic representation and late-layer semantic representation are pulling in different directions.

### Calibrated Entropy Divergence (this repo's contribution)

Raw entropy scores are not directly comparable across model sizes or domains. `calibrated_entropy_detector.py` fits an isotonic-regression calibration mapping from raw entropy to `P(hallucination)` on a labeled reference set, and separately fits a reference distribution of entropy signatures for *correct* answers (mean + shrinkage-regularized covariance). New examples are scored by:

```
p(x) = w · isotonic(u(x)) + (1 − w) · sigmoid(a · mahalanobis(x, μ_ref, Σ_ref) + b)
```

This directly implements the finding of arXiv:2603.21172 ("Entropy Alone is Insufficient for Safe Selective Prediction in LLMs") that pure entropy thresholds under-perform for selective prediction without calibration.

### Top-K Logprob Entropy (black-box)

Commercial completion APIs typically expose only a small top-K logprob list per token, not the full vocabulary distribution. `blackbox_detector.py` computes a documented *lower bound* on true entropy from only that top-K mass, still informative because a low top-K mass itself indicates most probability lies outside the visible top-K — i.e., high true entropy.

---

## Quickstart

```bash
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

# Full pipeline with self-generated data (requires ANTHROPIC_API_KEY)
python pipeline.py --data data/train.jsonl --model EleutherAI/pythia-160m --save detector.pkl
```

Default local model is [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m). For better hallucination rates, use larger models like Llama or Mistral.

### Usage Examples

```python
from detector import HallucinationDetector

detector = HallucinationDetector(classifier_type="logistic")
detector.fit(X_train, y_train)                 # X: (N, 18) attention features
metrics = detector.evaluate(X_test, y_test)
print(f"AUROC: {metrics.auroc:.4f}")
```

```python
from calibrated_entropy_detector import CalibratedEntropyDetector

detector = CalibratedEntropyDetector()
detector.fit(X_calibration, y_calibration)      # fits isotonic mapping + reference distribution
probs = detector.predict_proba(X_test)
divergence = detector.divergence_scores(X_test)  # raw Mahalanobis divergence, for diagnostics
```

```python
from blackbox_detector import simulate_topk_from_full_logits, extract_blackbox_features, BlackBoxEntropyDetector

# Offline (no API key) — derives simulated top-K logprobs from local model logits
sequence = simulate_topk_from_full_logits(logits, chosen_token_ids, top_k=5)
features = extract_blackbox_features(sequence)

detector = BlackBoxEntropyDetector()
detector.fit(X_train, y_train)
```

```python
from abstention import run_abstention_experiment

result = run_abstention_experiment(X, y, detector_factory=lambda: HallucinationDetector(classifier_type="logistic"))
print(f"Accuracy at {result['headline_coverage']:.0%} coverage: {result['headline_accuracy']:.4f}")
```

**Black-box demo (real API):** `blackbox_detector.py::fetch_topk_logprobs_openai()` calls OpenAI's `chat.completions` with `logprobs=True, top_logprobs=5` for a live demo (requires `pip install openai` and `OPENAI_API_KEY`). Everything else — including all tests and the `--synthetic`/`--halueval` pipeline modes — uses `simulate_topk_from_full_logits()`, an offline path that derives simulated top-K logprobs from local model logits, so no network access or API key is required to reproduce results.

**Adversarial robustness:** Tested against obfuscation (character substitution), paraphrase (synonym replacement), and multilingual (Spanish/French/German/Japanese prefix) attacks. Stability > 80% across all attack types. See `adversarial.py`.

**Embedding anomaly detection:** ChromaDB vector store + sentence-transformers; centroid distance and Mahalanobis distance from correct-answer embedding distribution. Ensembled with attention score: `0.6 × attn + 0.4 × embedding`. See `embedding_anomaly.py`.

**Deployment:** Vertex AI online endpoint (REST, autoscaling) and batch prediction (JSONL → GCS). See `vertex_deploy.py`.

**Tests:** `pytest tests` (wired into `pyproject.toml` testpaths). Every detector and feature-extraction module has unit coverage; torch/openai-dependent tests skip cleanly when those optional dependencies aren't installed.

*See [`AGENT.md`](AGENT.md) for implementation details, known limitations, and research foundations.*

---

## Project Structure

```
.
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
├── notebooks/                      # PyTorch experiment notebooks; results in notebooks/outputs/
├── colab/                          # Colab notebooks for GPU benchmark / ablation / validation
├── paper/                          # arXiv paper source
├── results/                        # Committed benchmark/ablation results (JSON)
├── AGENT.md
├── README.md
└── requirements.txt
```

---

## Feature Vector Reference

**Attention families (18D, `feature_engineer.py`):**

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

## Connection to Information Theory Literature

This work builds on a lineage of uncertainty quantification research in deep learning:

**Foundational uncertainty decomposition:**
- Malinin, A. & Gales, M. (2018). "Predictive Uncertainty Estimation via Prior Networks." *NeurIPS 2018.* — Introduced the decomposition of uncertainty into aleatoric (data uncertainty) and epistemic (model uncertainty) components for neural networks.
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML 2016.* — Established MC dropout as a practical tool for epistemic uncertainty estimation.

**LLM-specific calibration and hallucination detection:**
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). "Semantic Entropy: Detecting Hallucinations in Large Language Models." *ICML 2023.* — Related approach using semantic clustering across multiple generations; this repo uses single-pass distributional statistics plus calibration rather than multi-sample semantic clustering.
- Kadavath, S. et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv:2207.05221.* — Self-evaluation as a calibration signal.
- Chuang, Y. et al. (2024). "Lookback Lens: Detecting and Mitigating Contextual Hallucinations in LLMs Using Only Attention Maps." *EMNLP 2024.* — The lookback-ratio attention feature family.
- "Entropy Alone is Insufficient for Safe Selective Prediction in LLMs" (2026). *arXiv:2603.21172.* — Motivates the calibration layer in `calibrated_entropy_detector.py` directly.

**Entropy in information theory (foundational):**
- Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal, 27*(3):379–423.
- Kullback, S. & Leibler, R.A. (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics, 22*(1):79–86.

---

## How This Method Is Used Across the Portfolio

This repository is the methodological foundation for entropy-based uncertainty quantification in three other systems:

### 1. Cross-Cloud ML Orchestration
[`crosscloud-ml-orchestration`](https://github.com/A-Kuo/crosscloud-ml-orchestration) uses entropy-based routing to decide *which cloud provider* to route each inference request to. The core insight transfers directly: a model instance with lower output entropy on a given input type has more calibrated knowledge for that input type and should be preferred. Isotonic regression calibration aligns entropy scores across providers with different temperature scales.

### 2. Multi-Source Clinical Data Engineering
[`Multi-Source-Clinical-Data-Engineering-Platform`](https://github.com/A-Kuo/Multi-Source-Clinical-Data-Engineering-Platform) applies entropy-calibrated confidence to anomaly detection. The anomaly detector uses entropy over its own output distribution to distinguish genuine physiological anomalies (confident, low-entropy detection) from sensor noise or distribution-shifted inputs (high-entropy uncertainty that should trigger flagging rather than alerting). This avoids false positives from sensor malfunction triggering clinical alerts.

### 3. AI Safety & Red-Team Framework
[`AI-Safety-Benchmarking-RedTeam-Framework`](https://github.com/A-Kuo/AI-Safety-Benchmarking-RedTeam-Framework) uses entropy-based uncertainty as one component of vulnerability scoring. High-entropy regions in model output under adversarial prompts indicate semantic instability — a proxy for exploitability under that attack class.

---

## Research Limitations and Open Questions

### Current Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Calibration required per domain** | Entropy thresholds differ across knowledge domains; a score that indicates hallucination in historical facts may be normal in creative writing | Fit separate `CalibratedEntropyDetector` instances per use-case domain |
| **Confident confabulation** | A model can produce low-entropy hallucinations when it has seen similar (but incorrect) patterns many times in training | Pair with retrieval-augmented verification for high-stakes claims; attention-based features are a first-pass filter, not a complete solution (see `AGENT.md`) |
| **Layer/attention probing requires white-box access** | Attention-based and full-logit methods require model internals; not available for API-only models | `BlackBoxEntropyDetector` falls back to top-K logprobs for API-only access |
| **BiLSTM underperforms the flat baseline** | The per-layer sequence model has not yet outperformed logistic regression on HaluEval (~0.78 vs ~0.91 AUROC) — kept for sequence-model research, not as the recommended classifier | Use `classifier_type="logistic"` by default |
| **Does not localize the false claim** | The method scores *sequences*, not *propositions* — it cannot directly say "this specific entity name is hallucinated" | Use high-entropy token positions as candidate localization signals, then apply targeted retrieval |

### Open Research Questions

1. **Entropy under RLHF fine-tuning:** Models fine-tuned with RLHF learn to produce lower-entropy outputs as a reward-maximizing behavior. Does this compress the hallucination signal in the entropy distribution?
2. **Attention entropy vs. output entropy:** These are distinct signals with different layer-depth sensitivity; `ablation_study` in `pipeline.py` measures their relative contribution but a full head-to-head benchmark is still open.
3. **Multi-hop reasoning hallucinations:** Complex reasoning chains where each individual step has low entropy but the composition produces a false conclusion are not well captured by token-level entropy.
4. **Cross-lingual generalization:** Entropy calibration is likely language-specific.
5. **Hallucination vs. knowledge boundary:** Entropy reliably detects when the model is *outside* its training distribution, but conflates genuine uncertainty with confabulation on conflicting/incorrect training data.

---

## Current Status

**Beta**

- ✅ Attention-based multi-family feature extraction (5 families)
- ✅ Single-pass token-entropy scorer (`entropy_baselines.py`)
- ✅ Isotonic calibration + Mahalanobis divergence (`calibrated_entropy_detector.py`)
- ✅ Black-box top-K logprob detector (`blackbox_detector.py`)
- ✅ Abstention / selective-prediction evaluation (`abstention.py`)
- ✅ Stratified k-fold CV, bootstrap AUROC CIs, feature-family ablation
- ✅ Adversarial robustness evaluation, embedding anomaly detection
- 🔄 BiLSTM per-layer sequence model (experimental — underperforms the flat baseline)
- ⏸️ Proposition-level hallucination localization

---

## Related Work in This Portfolio

- [`crosscloud-ml-orchestration`](https://github.com/A-Kuo/crosscloud-ml-orchestration) — Applies entropy routing to multi-cloud inference provider selection
- [`Multi-Source-Clinical-Data-Engineering-Platform`](https://github.com/A-Kuo/Multi-Source-Clinical-Data-Engineering-Platform) — Uses entropy-calibrated anomaly detection for safety-critical sensor data
- [`AI-Safety-Benchmarking-RedTeam-Framework`](https://github.com/A-Kuo/AI-Safety-Benchmarking-RedTeam-Framework) — Entropy as a component of adversarial vulnerability scoring
- [`CIPHER`](https://github.com/A-Kuo/CIPHER) — Cryptographic integrity verification for AI-generated content; entropy-based authenticity analysis as complementary signal

---

## Citation

```bibtex
@software{hallucination_entropy_2026,
  author    = {A-Kuo},
  title     = {Language Model Hallucination Detection via Entropy Divergence},
  url       = {https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence},
  year      = {2026},
  note      = {Entropy-based uncertainty quantification and calibrated divergence for LLM hallucination detection at inference time}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable citation and [`paper/paper.tex`](paper/paper.tex) for the full writeup.

---

## License

[MIT](LICENSE)

---

*Uncertainty is not failure. Undetected uncertainty is.*
