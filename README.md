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

Each subsection below follows the same three-part structure: **Why** (the plain-English motivation — what problem this solves, and why this formulation rather than the obvious alternative), **The Math** (the formal definition, every symbol introduced before it's used), and **The Code** (where it's actually implemented, so you can read the formula and the code side by side).

### Notation

These symbols recur throughout every section below and are defined once here rather than re-introduced each time:

| Symbol | Meaning |
|---|---|
| $\mathcal{M}$ | The transformer language model under test |
| $L$ | Number of transformer layers |
| $H$ | Number of attention heads per layer |
| $T$ | Sequence length (number of tokens in the current input) |
| $V$ | The model's output vocabulary |
| $\mathbf{A}^{l,h} \in \mathbb{R}^{T \times T}$ | The attention weight matrix at layer $l$, head $h$ |
| $\mathbf{a}^{l,h} = \mathbf{A}^{l,h}_{T,:} \in \mathbb{R}^{T}$ | The **last token's** attention row — the distribution the model actually used to decide its next token |
| $p_t(v)$ | The model's predicted probability of vocabulary token $v$ at generation position $t$ (an **output**-distribution quantity, distinct from attention weights above) |
| $D_{\mathrm{KL}}(p \Vert q) = \sum_i p_i \log(p_i / q_i)$ | KL divergence between two discrete distributions, in nats |

Two distinct kinds of "distribution" appear throughout this project and are easy to conflate: **attention weights** ($\mathbf{a}^{l,h}$, over *input token positions*, one per layer/head) and the **output token distribution** ($p_t(v)$, over the *vocabulary*, one per generation step). The attention-based features below (entropy, lookback, frequency, spectral, cross-layer KL) all operate on the former; the token-entropy and black-box features operate on the latter. Both are legitimate, complementary white-box signals — the project's five attention families and its output-distribution features are two different axes of the same underlying question ("how uncertain is the model right now?"), not competing approaches.

### 1. Attention Entropy

**Why.** Entropy is not an arbitrary choice of uncertainty metric — it's the quantity Shannon's own axioms (continuity, monotonic increase with the number of equally-likely outcomes, and additivity over independent sub-choices) uniquely pin down as "the" measure of how much a distribution's mass is spread out. Compared to a simpler alternative like "the max attention weight" (which only looks at one point of the distribution), entropy accounts for the *entire shape* of where attention mass goes — it distinguishes "attention split evenly across 2 tokens" from "attention split evenly across 20," which a max-weight statistic alone cannot.

**The Math.** For attention row $\mathbf{a}^{l,h}$, the per-head Shannon entropy is

$$
H(\mathbf{a}^{l,h}) = -\sum_{j=1}^{T} a^{l,h}_j \log_2 a^{l,h}_j \;\in\; [0,\, \log_2 T]
$$

$H = 0$ is a delta distribution — the model attends entirely to one token (confident, focused). $H = \log_2 T$ is uniform — the model attends equally everywhere (maximally diffuse, uninformative). The per-layer mean across heads is $\bar{H}^l = \frac{1}{H}\sum_{h=1}^{H} H(\mathbf{a}^{l,h})$.

**The Code.** `feature_engineer.py::compute_entropy_features` (per-layer mean/max/std across heads); the analogous computation over the *output* distribution ($p_t(v)$ instead of attention weights) lives in `entropy_baselines.py` — see §3 below.

### 2. Cross-Layer KL Divergence

**Why.** Transformer layers are not independently redundant — they progressively refine a representation from surface-level syntactic patterns (early layers) toward abstract semantic content (late layers). When a model is generating a reliable continuation, this refinement should be *coherent*: consecutive layers should attend to roughly the same contextually relevant tokens, shifting incrementally rather than discontinuously. When a model is about to hallucinate, we hypothesize this coherence breaks down — the model can't find stable contextual support, so its attention pattern changes sharply from one layer to the next as it searches unsuccessfully for relevant information. This is a **structural** argument about how transformer representations evolve with depth, distinct from (and complementary to) any argument based on the output token distribution's entropy.

**The Math.** Divergence between consecutive layers' head-averaged attention:

$$
D_{\mathrm{KL}}^{l} = \frac{1}{H}\sum_{h=1}^{H} D_{\mathrm{KL}}\!\left(\mathbf{a}^{l,h} \,\big\Vert\, \mathbf{a}^{l+1,h}\right), \qquad l = 1, \ldots, L-1
$$

$D_{\mathrm{KL}}^l \approx 0$ means layers $l$ and $l+1$ attend to nearly the same tokens (stable, coherent). Large $D_{\mathrm{KL}}^l$ means they disagree about what's contextually relevant.

**The Code.** `feature_engineer.py::compute_kl_features` (global summary: total, max, std across layers) and `extract_layer_sequence` (the `kl_to_next` per-layer sequence feature, fed to the BiLSTM). A fuller worked derivation, including the full extraction algorithm, is in [`paper/paper.tex`](paper/paper.tex) §3–5.1.

### 3. Single-Pass Token Entropy

**Why.** Attention entropy (above) measures uncertainty in *how the model looked at its input*; token entropy measures uncertainty in *what the model actually said next* — the output distribution itself. It's the most direct operationalization of "the model doesn't know" and, unlike attention-based features, requires no attention weights at all — just logits, which is why this signal transfers most directly to the black-box case (§5).

A single scalar isn't enough to summarize a whole answer's uncertainty, so `entropy_baselines.py` computes six statistics, each catching something the others miss:

| Statistic | What it catches that the others don't |
|---|---|
| `entropy_mean` | Overall uncertainty across the answer — the primary aggregate signal |
| `entropy_max` | A single fabricated-entity spike hidden inside an otherwise confident sentence (the mean would dilute it away) |
| `entropy_std` | Whether uncertainty is spread evenly or concentrated in a few tokens — two answers with equal mean entropy can have very different risk profiles |
| `perplexity` | Realized surprise (of the tokens actually generated), vs. entropy's *potential* surprise (over the whole distribution) — these can diverge, e.g. a low-entropy distribution where the model still picked a low-probability token |
| `topk_entropy_mean` | A cheap, top-K-only approximation of entropy — this is the same estimator the black-box detector (§5) is forced to use everywhere; computing it here too lets it be validated against the true full-vocabulary entropy in tests |
| `margin_mean` | The (top-1 − top-2) logprob gap — a different functional form than entropy entirely (only looks at the top two mass points, ignores the tail), so it can disagree with entropy when a long tail of small probabilities inflates entropy without narrowing the top-1/top-2 gap |

**The Math.** For each generated token position $t$, over the answer span:

$$
H_t = -\sum_{v \in V} p_t(v) \log p_t(v)
$$

$$
H_{\mathrm{mean}} = \operatorname{mean}_t(H_t), \qquad H_{\max} = \max_t(H_t), \qquad \mathrm{perplexity} = \exp\!\Big(\operatorname{mean}_t\big(-\log p_t(v_t^{\ast})\big)\Big)
$$

where $v_t^{\ast}$ is the token actually realized at position $t$ (so perplexity depends on what was said, while $H_t$ depends only on the distribution the model *could have* said).

**The Code.** `entropy_baselines.py::compute_entropy_baseline_features`, fed by `pipeline.py::extract_logits_from_model`.

### 4. Calibrated Entropy Divergence — this repo's main contribution

**Why.** Raw entropy thresholds don't transfer across models or domains — a score that flags hallucination in historical facts might be normal in creative writing (arXiv:2603.21172, "Entropy Alone is Insufficient for Safe Selective Prediction in LLMs"). Instead of picking a fixed cutoff, `calibrated_entropy_detector.py` learns, from a labeled reference set, both (a) what raw-entropy values actually correspond to which hallucination probability, and (b) what a "normal, correct-answer" entropy signature looks like multivariately — then scores new examples by how far they diverge from that reference. Two design choices deserve their own justification, since they're the ones a natural first attempt would get wrong:

- **Why isotonic regression instead of Platt scaling** (fitting a logistic sigmoid to the raw score)? Platt scaling assumes the true calibration curve *is* a sigmoid — a specific parametric shape. Isotonic regression assumes only that the curve is monotonic (higher raw entropy never corresponds to *lower* hallucination probability) and fits the least-squares-optimal monotonic step function via pool-adjacent-violators, so it can represent calibration curves a fixed sigmoid cannot — e.g. one that's flat through a "confidently correct" low-entropy region and only turns sharply upward past some threshold. The tradeoff: isotonic regression has effectively as many degrees of freedom as calibration points before merging, so it can overfit on small calibration sets more readily than Platt's 2-parameter sigmoid — which is part of why the final score *blends* the isotonic term with a second, more constrained parametric term rather than relying on isotonic regression alone.
- **Why Mahalanobis distance instead of Euclidean distance** for the divergence term? Euclidean distance in raw feature space implicitly treats every feature as equally scaled and mutually independent — but `entropy_mean`, `perplexity`, and the rest have very different natural scales and are correlated with each other. Mahalanobis distance whitens by the reference covariance $\Sigma_{\mathrm{ref}}$, so a point that's far from the mean *along a direction where correct answers naturally vary a lot* is treated as less anomalous than the same raw distance along a direction where correct answers are tightly clustered.

**The Math.** Fit $\mu_{\mathrm{ref}}, \Sigma_{\mathrm{ref}}$ from the calibration set's correct-answer ($y=0$) examples (with diagonal shrinkage — see `calibrated_entropy_detector.py::fit_reference_distribution` — so $\Sigma_{\mathrm{ref}}$ stays invertible even when the feature count approaches the calibration-set size). The Mahalanobis divergence of a new example is

$$
d_{\mathrm{M}}(x) = \sqrt{(x-\mu_{\mathrm{ref}})^{\top}\, \Sigma_{\mathrm{ref}}^{-1}\, (x-\mu_{\mathrm{ref}})}
$$

Under the null hypothesis that $x$ is drawn from the same Gaussian as the reference set, $d_{\mathrm{M}}(x)^2 \sim \chi^2_{d}$ where $d$ is the feature dimension — since $\mathbb{E}[\chi^2_d] = d$, the *distance itself* scales roughly as $\sqrt{d}$, which is exactly why `embedding_anomaly.py`'s independent Mahalanobis implementation normalizes by $\sqrt{\dim}$ (previously justified there only by an empirical comment; this is the actual statistical reason). The final calibrated probability blends the two stages:

$$
p(x) = w \cdot \mathrm{isotonic}(u(x)) + (1-w) \cdot \sigma\big(a \cdot d_{\mathrm{M}}(x) + b\big)
$$

where $u(x)$ is a scalar raw score (default: `entropy_mean`), $\sigma$ is the logistic sigmoid, $(a, b)$ are fit by a 1-D logistic regression of $d_{\mathrm{M}}(x)$ against labels on the calibration set, and $w$ is a fixed blend weight.

**The Code.** `calibrated_entropy_detector.py` — `isotonic_regression` (Stage A), `fit_reference_distribution`/`mahalanobis_distance` (Stage B), `CalibratedEntropyDetector.predict_proba` (the blend).

### 5. Top-K Logprob Entropy (black-box)

**Why.** Everything above assumes access to attention weights or the full output distribution — commercial completion APIs give you neither, typically exposing only a small top-K logprob list per token (e.g. OpenAI's `top_logprobs=5`). `blackbox_detector.py` computes the same style of entropy estimator using only that truncated list. This is explicitly a **lower bound**, not an approximation: it ignores whatever probability mass lies outside the visible top-K, so it systematically *underestimates* true entropy — worst exactly for diffuse (high-entropy, hallucination-leaning) distributions, which is the case that matters most. It's still informative because a low top-K mass is itself evidence that a lot of probability lies outside the top-K, which independently implies high true entropy.

**The Math.** Given the top-$k$ logprobs at a position, renormalize them to sum to 1 and compute entropy over that truncated, renormalized distribution:

$$
\hat{p}^{(k)}_i = \frac{p_i}{\sum_{j=1}^{k} p_j} \quad (i = 1, \ldots, k), \qquad H^{(k)} = -\sum_{i=1}^{k} \hat{p}^{(k)}_i \log \hat{p}^{(k)}_i \;\le\; H^{(V)}
$$

**Worked example.** Suppose the true next-token distribution over a 10-token vocabulary is $p = (0.50, 0.20, 0.10, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01)$. Its true entropy is $H^{(V)} = 1.5711$ nats. Restricting to the top 5 ($0.50+0.20+0.10+0.06+0.05 = 0.91$ of the mass) and renormalizing gives $\hat{p}^{(5)} = (0.5495, 0.2198, 0.1099, 0.0659, 0.0549)$, with $H^{(5)} = 1.2434$ nats — an underestimate of $0.3277$ nats, entirely from the 9% of probability mass the top-5 view never sees. This gap only grows for flatter (higher-entropy, more hallucination-suggestive) distributions — exactly the regime where the estimate matters most and is least trustworthy in absolute terms, though it remains directionally useful (a lower top-K mass still reliably signals a more diffuse true distribution).

**The Code.** `blackbox_detector.py::topk_entropy_lower_bound` (the formula above), `topk_mass` (the $\sum_{j=1}^k p_j$ concentration proxy), `simulate_topk_from_full_logits` (derives what a real API's top-K response would look like, from full local logits, for offline testing).

### A note on AUROC: two equivalent estimators

This repo computes AUROC two different ways in two different places — `detector.py::compute_auroc` integrates the ROC curve via the trapezoidal rule (sweeping a threshold, plotting TPR vs. FPR), while `pipeline.py::stratified_kfold_cv` uses the Mann-Whitney U statistic, $\widehat{\mathrm{AUROC}} = \frac{1}{n_+ n_-}\sum_{i \in \mathrm{pos}}\sum_{j \in \mathrm{neg}} \mathbb{1}[s_i > s_j]$ — the fraction of positive/negative score pairs correctly ranked. These are the same quantity, computed two different ways (they differ only in tie-handling): AUROC *is* the probability that a randomly chosen hallucinated example gets a higher score than a randomly chosen correct one, and the trapezoidal ROC integral is a geometric restatement of exactly that probability. Neither implementation is more "correct" than the other; showing both is deliberate, since seeing the same result fall out of a geometric argument and a combinatorial/ranking argument is a good way to build confidence that both are right.

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
