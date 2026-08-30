# Language Model Hallucination Detection via Entropy Divergence

**Detecting hallucinations in LLM outputs using single-pass entropy statistics and divergence from calibrated entropy distributions.**


> *Just because you say it with confidence, doesn't make it true.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()
---

A model that is confidently wrong is more dangerous than a model that admits uncertainty. Given LLM uncertainty is a constant, the question is whether we can measure the moments before it happens.

Remember [Bard?](https://www.reuters.com/technology/google-ai-chatbot-bard-offers-inaccurate-information-company-ad-2023-02-08/) We had that before Google Gemini

<img width="518" height="50" alt="image" src="https://github.com/user-attachments/assets/ea6d7aa6-a87f-4ec2-8d78-ba2038a6d46c" />
---
## Contents

- [1. Abstract](#1-abstract)
- [2. Intuition](#2-intuition-what-entropy-is-and-why-it-matters-here)
- [3. Infrastructure](#3-infrastructure)
- [4. Method](#4-method)
  - [Notation](#notation)
  - [4.1 Attention Features](#41-attention-features-feature_engineerpy)
  - [4.2 Token-level Entropy Baselines](#42-token-level-entropy-baselines-entropy_baselinespy)
  - [4.3 Calibrated Entropy Divergence](#43-calibrated-entropy-divergence--the-main-contribution-calibrated_entropy_detectorpy)
  - [4.4 Black-box Top-K Detector (`blackbox_detector.py`)](#44-black-box-top-k-detector-blackbox_detectorpy)
  - [4.5 A note on AUROC](#45-a-note-on-auroc-two-estimators-one-quantity)
- [5. Results](#5-results)
  - [5.1 Headline](#51-headline-two-models-matched-pair-halueval)
  - [5.2 The Matched-Pair Fix](#52-the-matched-pair-fix--and-a-real-surprise)
  - [5.3 The Calibration Fix](#53-the-calibration-fix-from-a-negative-result-to-parity)
  - [5.4 3-Way Decision Routing](#54-3-way-decision-routing)
  - [5.5 Feature-Family Ablation Across Two Models](#55-feature-family-ablation-across-two-models)
  - [5.6 BiLSTM](#56-bilstm-a-second-data-point-undercuts-the-earlier-optimism)
  - [5.7 Black-box vs. White-box, Per Model](#57-black-box-vs-white-box-per-model)
  - [5.8 Latency](#58-latency)
  - [5.9 Historical benchmark — superseded, kept for provenance only](#59-historical-benchmark--superseded-kept-for-provenance-only)
  - [5.10 Caveats — read before citing any of this](#510-caveats--read-before-citing-any-of-this)
- [6. Running Experiments](#6-running-experiments)
  - [6.1 Local CLI](#61-local-cli)
  - [6.2 Kaggle GPU Runner](#62-kaggle-gpu-runner)
  - [6.3 Notebooks](#63-notebooks)
  - [6.4 Interactive Demo](#64-interactive-demo)
- [7. Repository Layout](#7-repository-layout)
- [8. Limitations](#8-limitations-and-open-questions)
  - [8.1 Known Limitations](#81-known-limitations)
  - [8.2 Open Questions](#82-open-questions)
- [9. Related Work](#9-related-works-(bibliograph))
- [10. Citation](#10-citation)
- [License](#license)

---

## 1. Abstract

Large language models produce confident text even when they are wrong. In a casual chat this is annoying; in a clinical, legal, or financial setting it is a real problem. [Dahl et al. (2024)](https://academic.oup.com/jla/article/16/1/64/7699227) tested public LLMs against real federal court cases and found hallucination rates from 58% (GPT-4) to 88% (Llama 2) on specific, verifiable legal questions — and models "struggle to predict their own hallucinations." [Singhal et al. (2023)](https://www.nature.com/articles/s41586-023-06291-2) found the same scaling problem in medicine: comprehension and reasoning on clinical QA improve with model size, but Med-PaLM remained "inferior to clinicians," with the gap concentrated exactly in the cases you would least want to get wrong. Confident and wrong is the failure mode that matters, because it is the one a user has no local reason to double-check.

Nor is this a bug that better training quietly fixes. [Kalai et al. (2025)](https://arxiv.org/abs/2509.04664) (OpenAI) give a structural reason it persists: generation is provably harder than verification: a model's generative error rate is bounded below by roughly twice its classification error rate on the same question, and standard training/eval procedures reward confident guessing over calibrated abstention. That is an argument for treating hallucination as a property to be *measured and scored* at inference time, not one to be trained away, which is this repo's premise.

You usually do not have ground truth available at inference time, so you cannot simply compare the output to a label and throw away the bad cases.

Existing detection approaches fall into three camps, and each has a clear gap:

| Approach | Mechanism | Where it breaks |
|----------|-----------|-----------------|
| **Output-text heuristics** | Keyword detection, hedging phrases, self-consistency checks | A model can be confidently wrong while sounding polished; hedging language is not a reliable signal |
| **Retrieval augmentation** | Ground every claim against a knowledge base | Hard to design for multi-step reasoning or synthesis; needs clean retrieval targets |
| **Human-in-the-loop** | Flag outputs for review | Does not scale, and does not tell you *which* outputs to flag |

This project takes a different route: **use the model's own internal uncertainty — its output probability distribution and its attention patterns — as the hallucination signal.** When a model leaves its comfort zone and starts fabricating, both change in measurable ways.

The core claim is not that entropy detects hallucination. That much is established ([Kadavath et al., 2022](https://arxiv.org/abs/2207.05221); [Kuhn et al., 2023](https://arxiv.org/abs/2302.09664); [Farquhar et al., 2024](https://www.nature.com/articles/s41586-024-07421-0)). The claim is that **a raw entropy threshold is the wrong way to use it.** Entropy scales differently across models, domains, and answer lengths, so a cutoff tuned on one distribution does not transfer. [Phillips et al. (2026)](https://arxiv.org/abs/2603.21172) show this directly: entropy-based selective prediction has a failure mode that persists until entropy is combined with a second, calibrated signal.

So instead of thresholding entropy, this repo **fits the distribution that entropy features actually take on known-correct answers, then scores new outputs by how far they diverge from that reference.** Two recent results converge on the same idea from different directions: [Villani et al. (2026)](https://arxiv.org/abs/2605.28264) find that the *shape and tail* of the token-entropy distribution carries signal independent of its mean, and build a calibrated reference distribution around exactly that; [Vathul et al. (2025)](https://arxiv.org/abs/2503.18242) classify sequence-level entropy *patterns* with a BiLSTM rather than reducing them to a scalar. This repo implements both readings — a calibrated divergence detector and a per-layer sequence model — and, importantly, reports where they do and do not beat a plain linear baseline.

Everything here runs in a single forward pass. That is a deliberate constraint. Semantic entropy ([Farquhar et al., 2024](https://www.nature.com/articles/s41586-024-07421-0)) and self-consistency methods ([Manakul et al., 2023](https://aclanthology.org/2023.emnlp-main.557/)) are strong but need 5–10 samples per query, and that cost is the most commonly cited barrier to deployment — the motivation behind cheap single-pass approximations like [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927). Single-pass is the regime this project targets.

**What the results actually show (§5), stated up front.** On matched-pair HaluEval QA, across two different model families (Pythia-160m and Qwen2.5-0.5B-Instruct), `CalibratedEntropyDetector` — the headline contribution — reaches 0.984–0.987 CV AUROC, edging out logistic regression (0.983–0.986) on both. That was not true in an earlier iteration of this project: the calibrated detector used to clearly underperform the linear baseline, and the fix (§5.3) was not tuning-until-it-looked-good but correcting one hardcoded hyperparameter that the project's own feature-family ablation had already shown was the wrong choice. The black-box detector, restricted to the top-5 logprobs an API would return, reaches 0.90–0.93 — a real practical result, though clearly behind the white-box detectors. Feature-family importance does not agree between the two models (§5.5): what looks like a dominant signal on one model is negligible on the other. Cross-layer KL divergence — the signal in the project's own name — remains the weakest family on both. These are reported as found, including where an earlier draft of this README was wrong about the direction a fix would move a number (§5.2).

---

## 2. Intuition: what entropy is, and why it matters here

When a language model generates a token it does not just pick one word. Internally it holds a probability for every word in its vocabulary. Conceptually:

```
"Paris":  0.7
"London": 0.2
"Berlin": 0.1
everything else: ~0
```

Those probabilities sum to 1 over the vocabulary $V$, and they are the model's belief about the next word. Entropy summarizes how spread out that belief is:

$$H(p) = -\sum_{v \in V} p(v) \log p(v)$$

- **Low entropy** — one word holds most of the probability. The model is focused. Whether it is *correct* is a separate question, but it is not uncertain.
- **High entropy** — probability is spread across many words. The model is undecided and effectively guessing.

This is grounded in information theory: Shannon entropy is the expected surprise of a distribution ([Shannon, 1948](https://ieeexplore.ieee.org/document/6773024)). A model that knows the answer produces low-surprise tokens. A model producing plausible-sounding text without grounded knowledge produces high-surprise ones.

**The catch, and the reason this repo exists.** Raw entropy values are not comparable across settings. An entropy of 2.0 nats might be alarming for a short factual lookup and completely normal for open-ended generation. That is why the pipeline calibrates against a reference distribution instead of thresholding — see §4.3.

---

## 3. Infrastructure

1. **Feature extraction** that summarizes internal uncertainty.
   - Token-level entropy from output logits (`entropy_baselines.py`)
   - Attention-based features across five families (`feature_engineer.py`)
   - Top-k logprob features for black-box APIs (`blackbox_detector.py`)

2. **Detectors** built on those features — logistic regression and MLP baselines, a BiLSTM over per-layer sequences, plus the two calibration-focused detectors that are the main contribution.

3. **Evaluation** on synthetic data and the HaluEval benchmark ([Li et al., 2023](https://aclanthology.org/2023.emnlp-main.397/)): stratified k-fold AUROC with bootstrap confidence intervals, feature-family ablation, and abstention / risk-coverage analysis.

4. **Experiment infrastructure** — a local CLI, a Kaggle GPU runner wired to GitHub Actions, and JSON result artifacts so runs can be compared over time.

---

## 4. Method

Each subsection below gives the **why** (what problem it solves, and why this choice over the obvious alternative), the **math**, and the **code** that implements it.

### Notation

| Symbol | Meaning |
|---|---|
| $L$ | Number of transformer layers |
| $H$ | Number of attention heads per layer |
| $T$ | Sequence length in tokens |
| $V$ | Output vocabulary |
| $\mathbf{a}^{l,h} \in \mathbb{R}^{T}$ | Last token's attention row at layer $l$, head $h$ — the distribution the model used to pick its next token |
| $p_t(v)$ | Model's predicted probability of token $v$ at generation step $t$ |
| $D_{\mathrm{KL}}(p \Vert q) = \sum_i p_i \log(p_i/q_i)$ | KL divergence, in nats |

Two different distributions appear throughout and are easy to conflate: **attention weights** ($\mathbf{a}^{l,h}$, over input positions, one per layer/head) and the **output token distribution** ($p_t(v)$, over the vocabulary, one per generation step). The five attention families operate on the former; token-entropy and black-box features on the latter. They are complementary axes on the same question, not competing methods.

### 4.1 Attention features (`feature_engineer.py`)

**Why.** Attention entropy measures *how the model looked at its input*, which is a different signal than *what it said*. Prior work found this signal is not just present but often stronger than logit confidence: [Chuang et al. (2024)](https://aclanthology.org/2024.emnlp-main.84/) show a linear probe on attention ratios alone matches detectors using full hidden states.

Five families, 18 dimensions total:

| Family | What it captures | Grounded in |
|---|---|---|
| **Entropy** (3D) | How scattered attention is per head | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024) |
| **Lookback ratio** (4D) | Attention on source context vs. the model's own generations | [Chuang et al. (2024)](https://aclanthology.org/2024.emnlp-main.84/) |
| **Frequency / DFT** (4D) | High-frequency energy — rapid, unstable attention shifts | [Qi et al. (2026)](https://arxiv.org/abs/2602.18145) |
| **Spectral / Laplacian** (4D) | Fiedler value and spectral gap of attention-as-graph | [Binkowski et al. (2025)](https://arxiv.org/abs/2502.17598) |
| **Cross-layer KL** (3D) | Disagreement between consecutive layers | This repo / [Bazarova et al. (2026)](https://arxiv.org/abs/2504.10063) |

**The math.** Per-head Shannon entropy of the last token's attention row:

$$H(\mathbf{a}^{l,h}) = -\sum_{j=1}^{T} a^{l,h}_j \log_2 a^{l,h}_j \;\in\; [0,\, \log_2 T]$$

$H = 0$ is a delta distribution (attends entirely to one token). $H = \log_2 T$ is uniform (attends everywhere equally). Cross-layer divergence between consecutive layers:

$$D_{\mathrm{KL}}^{l} = \frac{1}{H}\sum_{h=1}^{H} D_{\mathrm{KL}}\!\left(\mathbf{a}^{l,h} \,\big\Vert\, \mathbf{a}^{l+1,h}\right), \qquad l = 1, \ldots, L-1$$

**Why cross-layer KL specifically.** Transformer layers are not redundant copies — they progressively refine representations from syntactic to semantic ([Tenney et al., 2019](https://aclanthology.org/P19-1452/)). When generation is well-grounded, that refinement should be *coherent*: consecutive layers attend to roughly the same tokens, shifting incrementally. When the model cannot find stable contextual support, attention changes sharply layer to layer as it searches. This is a structural argument about representation dynamics, distinct from any claim about output entropy. [Bazarova et al. (2026)](https://arxiv.org/abs/2504.10063) independently arrive at a related conclusion using topological divergence on attention graphs.

### 4.2 Token-level entropy baselines (`entropy_baselines.py`)

**Why six statistics instead of one.** A single scalar cannot summarize a whole answer's uncertainty, and [Villani et al. (2026)](https://arxiv.org/abs/2605.28264) show empirically that distributional shape beyond the mean carries independent signal. Each statistic catches something the others miss:

| Statistic | What it catches that the others don't |
|---|---|
| `entropy_mean` | Overall uncertainty — the primary aggregate |
| `entropy_max` | A single fabricated-entity spike inside an otherwise confident sentence, which the mean dilutes away |
| `entropy_std` | Whether uncertainty is evenly spread or concentrated in a few tokens |
| `perplexity` | Realized surprise (of tokens actually generated) vs. entropy's *potential* surprise over the whole distribution |
| `topk_entropy_mean` | The same truncated estimator the black-box detector is forced to use — computed here against known full-vocab entropy so the approximation can be validated |
| `margin_mean` | Top-1 minus top-2 logprob gap — a different functional form that ignores the tail, so it can disagree with entropy when a long tail inflates it |

$$H_t = -\sum_{v \in V} p_t(v) \log p_t(v), \qquad \mathrm{perplexity} = \exp\!\Big(\mathrm{mean}_t\big(-\log p_t(v_t^{\ast})\big)\Big)$$

where $v_t^{\ast}$ is the token actually realized at position $t$ — so perplexity depends on what was said, while $H_t$ depends only on what could have been said.

### 4.3 Calibrated entropy divergence — the main contribution (`calibrated_entropy_detector.py`)

**Why.** This is the part that addresses the transfer problem from §2. Rather than picking a cutoff, the detector learns from a labeled calibration set (a) what raw entropy values actually correspond to which hallucination probability, and (b) what a normal correct-answer entropy signature looks like *multivariately* — then scores by divergence from that reference.

Two design choices a first attempt would plausibly get wrong:

**Why isotonic regression, not Platt scaling.** Platt scaling assumes the calibration curve *is* a sigmoid ([Platt, 1999](https://www.researchgate.net/publication/2594015); [Guo et al., 2017](https://arxiv.org/abs/1706.04599)). Isotonic regression assumes only monotonicity — higher entropy never maps to *lower* hallucination probability — and fits the least-squares-optimal monotonic function by pool-adjacent-violators ([Zadrozny & Elkan, 2002](https://dl.acm.org/doi/10.1145/775047.775151)). It can represent curves a fixed sigmoid cannot, e.g. flat through a confidently-correct low-entropy region then sharply rising. The tradeoff is more degrees of freedom and easier overfitting on small calibration sets — which is why the final score blends it with a more constrained parametric term rather than using it alone.

**Why Mahalanobis distance, not Euclidean.** Euclidean distance implicitly treats every feature as equally scaled and mutually independent, which `entropy_mean`, `perplexity`, and the rest are not. Mahalanobis whitens by the reference covariance, so displacement along a direction where correct answers naturally vary counts for less than the same displacement along a tight one. This mirrors standard practice in out-of-distribution detection ([Lee et al., 2018](https://arxiv.org/abs/1807.03888)).

**The math.** Fit $\mu_{\mathrm{ref}}, \Sigma_{\mathrm{ref}}$ on correct-answer ($y=0$) calibration examples, with diagonal shrinkage so $\Sigma_{\mathrm{ref}}$ stays invertible when feature count approaches calibration-set size:

$$d_{\mathrm{M}}(x) = \sqrt{(x-\mu_{\mathrm{ref}})^{\top}\, \Sigma_{\mathrm{ref}}^{-1}\, (x-\mu_{\mathrm{ref}})}$$

Under the null hypothesis that $x$ comes from the reference distribution, $d_{\mathrm{M}}(x)^2 \sim \chi^2_{d}$ for feature dimension $d$. Since $\mathbb{E}[\chi^2_d] = d$, the distance scales as $\sqrt{d}$ — which is why `embedding_anomaly.py` normalizes by $\sqrt{\dim}$. The final score blends both stages:

$$p(x) = w \cdot \mathrm{isotonic}(u(x)) + (1-w) \cdot \sigma\big(a \cdot d_{\mathrm{M}}(x) + b\big)$$

with $u(x)$ a scalar raw score, $(a,b)$ fit by 1-D logistic regression of $d_\mathrm{M}$ against labels, and $w$ a fixed blend weight ($w=0.7$ by default — swept empirically, not guessed; see §5.3).

**A note on $u(x)$, since an earlier version of this got it wrong.** $u(x)$ was originally hardcoded to a fixed feature column (`entropy_mean`), on the assumption that raw entropy was the natural scalar to calibrate. This project's own feature-family ablation (§5.5) later showed that assumption was false — lookback-ratio features carry more single-feature signal than entropy on the model tested. `u(x)` now defaults to whichever feature column has the strongest single-feature separation on the training data, resolved once at `fit()` time (`score_index="auto"`, `calibrated_entropy_detector.py::_resolve_score_index`). §5.3 has the before/after numbers — this one change closed most of the gap to the linear baseline.

**Beyond a probability: 3-way routing.** `route()` maps $p(x)$ into RELIABLE / UNCERTAIN / UNRELIABLE using two thresholds fit from the calibration set's own score distribution (not hardcoded), reviving a decision-routing concept from an earlier version of this project rebuilt on the calibration machinery above. See §5.4 for the exact rule and the [Streamlit demo](demo/) for it in use.

### 4.4 Black-box top-K detector (`blackbox_detector.py`)

**Why.** Everything above needs attention weights or the full output distribution. Commercial APIs give neither — typically only a top-K logprob list per token (e.g. OpenAI's `top_logprobs=5`). This detector works under that constraint, which is the same practical motivation behind [SelfCheckGPT](https://aclanthology.org/2023.emnlp-main.557/), though that method spends multiple samples where this one spends one.

**The math.** Renormalize the visible top-$k$ and compute entropy over it:

$$\hat{p}^{(k)}_i = \frac{p_i}{\sum_{j=1}^{k} p_j}, \qquad H^{(k)} = -\sum_{i=1}^{k} \hat{p}^{(k)}_i \log \hat{p}^{(k)}_i \;\le\; H^{(V)}$$

This is a **lower bound, not an approximation** — and it is worth being precise about the bias. Take a true distribution over 10 tokens: $p = (0.50, 0.20, 0.10, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01)$, with $H^{(V)} = 1.571$ nats. The top-5 hold 0.91 of the mass; renormalized they give $H^{(5)} = 1.243$ nats — an underestimate of 0.328 nats, entirely from the 9% the API never showed you.

**Interpretation:** the gap grows for flatter, higher-entropy distributions — exactly the hallucination-leaning regime where the estimate matters most and is least trustworthy in absolute terms. It stays *directionally* useful because low top-K mass is itself evidence that most probability sits outside the visible window.

### 4.5 A note on AUROC: two estimators, one quantity

`detector.py::compute_auroc` integrates the ROC curve trapezoidally; `pipeline.py::stratified_kfold_cv` uses the Mann-Whitney U statistic:

$$\mathrm{AUROC} = \frac{1}{n_{+} n_{-}} \sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbb{1}[s_i > s_j]$$

These are the same quantity up to tie handling ([Hanley & McNeil, 1982](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747)) — AUROC *is* the probability a random hallucinated example outranks a random correct one, and the ROC integral is a geometric restatement of that probability. Both are implemented deliberately; agreement between a geometric and a ranking derivation is a cheap correctness check.

---

## 5. Results

Four result generations live in `results/`. §5.1–§5.6 below are the current, trustworthy numbers — matched-pair HaluEval data, two independent model families, a tuned calibrated detector. §5.7 keeps the old, since-superseded numbers as a provenance record, not a citable result. Read §5.9 for what's still not measured.

### 5.1 Headline: two models, matched-pair HaluEval

400 samples (200 matched question pairs — see §5.2), stratified 5-fold CV, bootstrap 95% CIs. Reproduce with:

```bash
python pipeline.py --halueval --num_samples 400 --model EleutherAI/pythia-160m \
    --results results/halueval_pythia160m_n400_paired.json
python pipeline.py --halueval --num_samples 400 --model Qwen/Qwen2.5-0.5B-Instruct \
    --results results/halueval_qwen25_0.5b_n400.json
```

| Detector | Pythia-160m (base) | Qwen2.5-0.5B-Instruct |
|---|---|---|
| **CalibratedEntropyDetector** | **0.9837** [0.9723, 0.9926] | **0.9869** [0.9767, 0.9951] |
| Logistic regression | 0.9828 [0.9681, 0.9937] | 0.9860 [0.9714, 0.9963] |
| MLP | 0.9799 [0.9639, 0.9922] | 0.9844 [0.9677, 0.9956] |
| `BlackBoxEntropyDetector` | 0.9285 [0.9030, 0.9501] | 0.9005 [0.8668, 0.9288] |

**Two real findings here.** First, the method is not tied to one weak base model: results hold — and are marginally *better* — on a completely different model family (Qwen2.5, instruction-tuned, different tokenizer, different attention head/layer geometry) using the same 24D feature pipeline, chat-template-aware prompting (§5.1's "why" is in `pipeline.py::build_prompt_and_text`), and no per-model tuning. Second, `CalibratedEntropyDetector` — this repo's headline contribution — now edges out logistic regression on *both* models. That was not true in an earlier iteration of this table; §5.3 explains what changed and why, honestly, including the version of this detector that did not work.

### 5.2 The matched-pair fix — and a real surprise

An earlier version of `DataGenerator.from_halueval()` loaded a HaluEval mirror (`pminervini/HaluEval`, `qa_samples`) with **zero matched question pairs** — 10,000 rows, 10,000 unique questions, meaning "correct" and "hallucinated" were disjoint question populations rather than paired variants of the same question. That is a real confound: a detector could score well by picking up on differences between two unrelated question sets rather than hallucination itself. `from_halueval()` now loads `shunk031/HaluEval` (`qa` config), which preserves the original `knowledge`/`question`/`right_answer`/`hallucinated_answer` schema — verified directly to contain 10,000 genuinely matched rows (`tests/test_data_generator.py` protects this invariant offline, no network required).

**The honest surprise: fixing the confound did not shrink the numbers.** The working hypothesis going in was that the disjoint-question setup inflated AUROC — that hypothesis does not survive contact with the data. Pythia-160m's logistic-regression CV AUROC went from 0.9728 (disjoint, `results/halueval_pythia160m_n400.json`) to 0.9828 (matched-pair, §5.1) — *up*, not down. The likely explanation: disjoint questions introduce topic-level noise a model has to generalize across, while matched pairs isolate the one thing that actually differs (the answer's entropy/attention signature) and remove everything else as a nuisance variable. This is a genuine update against a prior stated with too much confidence in an earlier draft of this README — worth leaving visible rather than quietly editing away.

### 5.3 The calibration fix: from a negative result to parity

The previous draft of this README reported `CalibratedEntropyDetector` clearly behind logistic regression (0.9149 vs 0.9728) and treated it as an open negative result. Digging into *why* found a specific, fixable cause: `score_index` — the single feature column fed to the detector's isotonic-calibration stage — was hardcoded to column 0 (`entropy_mean`), despite this repo's own ablation (§5.5) showing lookback-ratio features, not entropy, carry the strongest single-feature signal. `score_index` now defaults to `"auto"`, resolving at `fit()` time to whichever column has the highest single-feature `|AUROC-0.5|` on the training data (`calibrated_entropy_detector.py::_resolve_score_index`).

Swept on real paired-HaluEval features (Pythia-160m, 400 samples, 5-fold CV):

| Configuration | CV AUROC |
|---|---|
| Old defaults (`score_index=0`, `blend_weight=0.5`) | 0.9323 |
| `score_index="auto"`, `blend_weight=0.5` | 0.9818 |
| `score_index="auto"`, `blend_weight=0.7` (**new default**) | **0.9833–0.9837** |
| `score_index="auto"`, `blend_weight=0.0` (pure divergence term) | 0.8840 |
| `score_index="auto"`, `blend_weight=1.0` (pure isotonic term) | 0.9689 |

Auto-selecting the scalar score closes nearly the entire gap by itself (0.9323 → 0.9818); `blend_weight=0.7` — also swept, not guessed — adds a further small, consistent gain over the old default of 0.5. Neither pure term alone matches the blend (0.8840 and 0.9689 respectively), so the two-stage design is doing real, verifiable work, not just carrying dead weight. This is not a case of tuning until a number looks good: the fix is one hardcoded index that ablation had already shown was the wrong choice, corrected to select empirically rather than by a fixed guess.

### 5.4 3-way decision routing

Beyond a raw probability, `CalibratedEntropyDetector.route()` now maps its output into **RELIABLE / UNCERTAIN / UNRELIABLE** with an associated action (`accept` / `escalate` / `reject`), reviving a decision-routing concept from an earlier version of this project (`v1/confidence_calibrator.py`, dropped during a repo consolidation) rebuilt on the current calibration machinery rather than the old code. The two thresholds are fit from the calibration set, not hardcoded — see `calibrated_entropy_detector.py`'s module docstring for the exact quantile rule and the conservative-safety reasoning behind it (a false RELIABLE is worse than a false UNCERTAIN). This is the headline UI element in the [Streamlit demo](demo/) (§6.4).

### 5.5 Feature-family ablation across two models

Drop one family, retrain logistic regression on the rest (full 24D model: Pythia held-out AUROC 0.97, Qwen held-out AUROC 0.9714):

| Family removed | Pythia-160m Δ | Qwen2.5-0.5B-Instruct Δ |
|---|---|---|
| Lookback | **+0.0114** | −0.0017 |
| Entropy (attention + token) | +0.0075 / +0.0019 | −0.0039 / −0.0006 |
| Frequency | −0.0017 | +0.0022 |
| Spectral | −0.0022 | +0.0025 |
| Cross-layer KL | +0.0006 | +0.0039 |

(Δ = full-model AUROC minus AUROC-with-family-removed; positive means the family helps.)

**These two columns do not agree with each other, and that disagreement is itself the finding.** On Pythia, lookback dominates by an order of magnitude over every other family — consistent with [Chuang et al. (2024)](https://aclanthology.org/2024.emnlp-main.84/), whose entire method is the lookback ratio. On Qwen, no family shows a magnitude above 0.004 in either direction — the full 24D model barely beats any single-family-ablated subset, meaning the feature set is *redundant* on this model rather than *dependent* on one dominant signal. Family importance is not a fixed property of the method; it is model-dependent, and reporting only one model's ablation (as an earlier version of this README did) would have overstated how universal the lookback finding is. Cross-layer KL divergence — the signal in this project's name — is the weakest or near-weakest family on both models, which was already the case before this fix and remains an open, unresolved weak point (§8.1).

### 5.6 BiLSTM: a second data point undercuts the earlier optimism

| | Pythia-160m held-out | Qwen2.5-0.5B-Instruct held-out |
|---|---|---|
| BiLSTM | 0.9747 | 0.9539 |
| Logistic regression | 0.9700 | 0.9714 |
| MLP | 0.9728 | 0.9728 |

An earlier draft of this README, working from Pythia numbers alone, said the BiLSTM was "no longer obviously worse" than the linear baseline after two corrupting bugs were fixed (label/sequence misalignment; a `context_length` error). That reads differently with a second model in hand: on Qwen, BiLSTM is clearly *behind* both flat detectors (0.9539 vs ~0.972), the opposite pattern from Pythia. Neither held-out split is cross-validated, so neither number alone should be over-read — but two models pointing in opposite directions is stronger evidence of instability than either one pointing anywhere. The honest summary is now "inconsistent across models," not "recovering," and a proper CV comparison remains the actual open item.

### 5.7 Black-box vs. White-box, Per Model

`BlackBoxEntropyDetector` — top-5 logprobs only, no attention, no full-vocabulary distribution — scores 0.9285 on Pythia and 0.9005 on Qwen (§5.1), both clearly behind the white-box detectors but well above chance. The gap widens on Qwen, plausibly because chat-template-formatted generation concentrates more probability mass in the visible top-5 for a well-instruction-tuned model's more templated answers, making the truncated-entropy lower bound (§4.4) a worse approximation of the true distribution's shape. That is a plausible mechanism, not a confirmed one — it has not been tested directly (e.g. by measuring top-5 mass coverage per model) and should be read as a hypothesis for future work, not a finding.

### 5.8 Latency

38.49 ms/sample on a T4 GPU (from the historical benchmark, §5.9 — a comparable number has not yet been re-measured on the current pipeline). The comparison that matters: semantic-entropy methods ([Farquhar et al., 2024](https://www.nature.com/articles/s41586-024-07421-0)) need 5–10 generations per query. Single-pass detection is roughly an order of magnitude cheaper, which is the whole argument for accepting a weaker signal.

### 5.9 Historical Benchmark — superseded, kept for provenance only

`results/halueval_pythia160m_n400.json` (disjoint-question data, old detector defaults) and `results/benchmark_results.json` / `results/ablation_results.json` (T4 GPU, April 2026, an earlier two-signal "AED" configuration — per-head entropy + cross-layer KL only — run through notebooks since removed as dead wrappers around a deleted module, §6.3) are no longer the numbers to cite. They remain committed because §5.2 and §5.5 directly compare against them to show what changed and why; do not use them as current evidence on their own. `results/benchmark_results_cpu_quick.json` (GPT-2, 50 samples) exists only to prove the pipeline executes.

### 5.10 Caveats — read before citing any of this

1. **Sample sizes are still modest.** 400 samples per model, 120-sample held-out splits. The CIs in §5.1 are wide enough that some between-detector gaps are not individually significant — read the CV table's confidence intervals, not just the point estimates.
2. **Two models is not many models.** §5.1/§5.5/§5.6 now span a base model and an instruction-tuned model of similar scale (160M/500M parameters) — real progress over one model, but still nothing above ~0.5B parameters, and no cross-domain evaluation (everything is HaluEval QA).
3. **No committed abstention/risk-coverage curve on real data.** `abstention.py` runs, but no result artifact is committed.
4. **The adversarial-robustness claim is unmeasured.** `adversarial.py` implements obfuscation, paraphrase, and multilingual-prefix attacks; no committed results file records the outcome.
5. **§5.7's black-box explanation is a hypothesis, not a finding** — flagged there, repeated here because it's the kind of claim that's easy to skim past as settled.

`notebooks/real_pipeline_benchmark/` exists to run §5.1 at a scale a laptop CPU cannot reach; see §6.3.

---

## 6. Running experiments

### 6.1 Local CLI

```bash
pip install -r requirements.txt
```

Synthetic demo — no model download, no API, runs in seconds:

```bash
python pipeline.py --synthetic --num_samples 1000
```

HaluEval benchmark with a local model (needs `pip install torch transformers datasets`) — works with both base and instruction-tuned models, since `pipeline.py` detects a chat template and switches prompt format automatically (`pipeline.py::build_prompt_and_text`):

```bash
python pipeline.py --halueval --num_samples 400 --model EleutherAI/pythia-160m --results results/my_run.json
python pipeline.py --halueval --num_samples 400 --model Qwen/Qwen2.5-0.5B-Instruct --results results/my_run_qwen.json
```

Abstention / risk-coverage analysis:

```bash
python abstention.py --synthetic --num_samples 1000
python pipeline.py --synthetic --num_samples 1000 --abstention
```

Your own labeled data (JSONL, see `data_generator.py::LabeledSample` for the schema):

```bash
python pipeline.py --data path/to/train.jsonl --model EleutherAI/pythia-160m --save detector.pkl
```

Tests:

```bash
pytest tests
```

Passing `--results <path>` writes a structured JSON summary (per-detector CV AUROC with bootstrap CIs, held-out metrics, ablation deltas, feature importances) so runs stay comparable.

### 6.2 Kaggle GPU Runner

GPU work runs on Kaggle, triggered manually through GitHub Actions. **It never fires on push** — GPU quota is finite, so runs happen only on explicit command.

```bash
# From the Actions tab: "Kaggle GPU Run" → "Run workflow" → pick kernel_dir
# Or from the CLI:
gh workflow run kaggle_runner.yml -f kernel_dir=notebooks/real_pipeline_benchmark
```

The workflow pushes the chosen kernel to Kaggle, polls until it finishes, pulls the output back, and commits it to `notebooks/results/`. One-time setup (Kaggle phone verification, `KAGGLE_USERNAME` / `KAGGLE_KEY` repo secrets) is documented in [`notebooks/README.md`](notebooks/README.md).

Before trusting CI, push a kernel manually once to confirm auth works:

```bash
pip install kaggle
kaggle kernels push -p notebooks/real_pipeline_benchmark/
kaggle kernels status <your-username>/real-pipeline-benchmark
```

### 6.3 Notebooks

| Notebook | Status | Purpose |
|---|---|---|
| `notebooks/pytorch_dl_testbed.ipynb` | **Working** | Synthetic sandbox — a controllable pseudo-LM with tunable difficulty (`gamma`) and confident-confabulation fraction (`confab_frac`). Tests architecture hypotheses cheaply before touching real code. |
| `notebooks/real_pipeline_benchmark/` | **Working** | Clones the repo and calls the real `pipeline.py::run_real_pipeline()` on real HaluEval data — the local CLI (not yet this notebook) already produced §5.1's two-model results; this notebook exists to rerun the same thing at a scale/model size a laptop CPU cannot reach. |
| `notebooks/full_pipeline.ipynb` | Working | End-to-end Claude-labeled data generation → features → classifier (needs `ANTHROPIC_API_KEY`) |

Three other notebooks (`gpu_benchmark.ipynb`, `ablation_study.ipynb`, `quick_cpu_validation.ipynb`) generated §5.9's historical numbers back when the repo had a separate `run_experiment.py` module. That module was removed during consolidation, the notebooks were thin wrappers around it with no algorithmic content of their own, and the current 5-family pipeline strictly supersedes the two-signal feature set they exercised — so rather than porting dead wrappers, they were removed. `results/benchmark_results.json` and `results/ablation_results.json` remain committed as the historical record §5.9 already documents.

### 6.4 Interactive Demo

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

`demo/detector.pkl` is committed, so this runs immediately — `demo/build_detector.py` exists to rebuild it against a different model/sample size, not as a required first step.

A small local model (Pythia-160m by default) answers questions live — some correctly, some hallucinated — while a `CalibratedEntropyDetector` scores every answer in real time and shows the §5.4 RELIABLE/UNCERTAIN/UNRELIABLE routing as the headline result, with the raw probability and top contributing features underneath. A second mode browses real HaluEval question pairs (correct vs. hallucinated answer, same question) side by side. See [demo/README.md](demo/README.md) for details and the honest framing on what a 160M-parameter base model will and won't do on demand.

---

## 7. Repository Layout

```text
.
├── data_generator.py               # HaluEval loader + Claude-labeled synthetic data
├── feature_engineer.py             # 5 attention families → 18D vector + per-layer (L×6) sequence
├── entropy_baselines.py            # Single-pass token-entropy features (6D) from logits
├── detector.py                     # LogReg / MLP / BiLSTM + shared metrics
├── calibrated_entropy_detector.py  # Main contribution: isotonic calibration + Mahalanobis divergence
├── blackbox_detector.py            # Top-K logprob detector (real API + offline simulation)
├── abstention.py                   # Risk-coverage / selective prediction
├── pipeline.py                     # End-to-end runner: k-fold CV, bootstrap CIs, ablation
├── adversarial.py                  # Robustness: obfuscation, paraphrase, multilingual
├── embedding_anomaly.py            # ChromaDB + centroid/Mahalanobis anomaly detection
├── vertex_deploy.py                # GCP Vertex AI deployment scaffolding
├── tests/                          # pytest suite, one file per module
├── notebooks/                      # Kaggle/Colab experiment notebooks (see §6.3 for status)
├── demo/                           # Streamlit live demo (see §6.4)
│   └── requirements.txt            # self-contained: numpy/scipy + streamlit + torch (CPU wheel) + transformers + datasets
├── paper/                          # arXiv paper source (paper.tex, references.bib)
├── results/                        # Committed benchmark JSON (see §5)
├── AGENT.md                        # Design notes, math reference, known limitations
├── runtime.txt                     # pins Python 3.11 for Streamlit Cloud (wheel availability, see demo/README.md)
└── requirements.txt
```

Core dependencies are `numpy` and `scipy` only. `torch`/`transformers` (real models), `datasets` (HaluEval), `anthropic` (data generation), `openai` (black-box demo), `chromadb`/`sentence-transformers` (embedding anomaly), and `kaggle` (CI) are all optional and lazily imported — tests that need them skip cleanly when they are absent. `demo/`'s dependencies live in `demo/requirements.txt`, not the root file — deliberately, and not just for hygiene: Streamlit Community Cloud auto-detects a requirements file next to the app's main module before falling back to the repo root, so this placement is what makes a zero-config cloud deploy actually install torch (see [demo/README.md](demo/README.md#deploying-streamlit-community-cloud)).

---

## 8. Limitations and Open Questions

### 8.1 Known Limitations

| Limitation | Why it happens | Current mitigation |
|---|---|---|
| **Calibration is domain-specific by design** | Entropy scale shifts across domains; a threshold tuned on factual QA does not transfer to creative writing | Fit a separate `CalibratedEntropyDetector` per domain; §5.1 shows it transfers across *models*, not yet across *domains* |
| **Confident confabulation** | A model can produce low-entropy *and wrong* output when training data reinforced an incorrect pattern | Attention features are a first-pass filter, not a solution; pair with retrieval for high-stakes claims |
| **White-box access required for most features** | Attention and full-logit features need model internals, unavailable via API | `BlackBoxEntropyDetector` falls back to top-K logprobs, with strictly less information (§4.4), and is measurably weaker on the instruction-tuned model tested (§5.7) |
| **Cross-layer KL contributes nothing measurable** | Three independent runs now (§5.5's two models plus §5.9's historical run) show it near-zero or negative in ablation, despite naming the project | Either find the regime where it helps, or drop it and rename the framing around lookback + entropy |
| **Feature-family importance does not transfer across models** | §5.5: lookback dominates on Pythia, nothing dominates on Qwen | Don't tune a deployment around one model's ablation result — re-run it per target model |
| **BiLSTM is inconsistent across models** | §5.6: ahead of the linear baseline on Pythia, behind it on Qwen, on non-cross-validated splits both times | Default to `classifier_type="logistic"`; treat BiLSTM as unproven rather than either "fixed" or "broken" |
| **No claim localization** | The detectors score *sequences*, not propositions — "this answer is risky", not "this span is wrong" | Use high-entropy token positions as candidate spans, then targeted retrieval |

**The confident-confabulation limitation is the most fundamental one.** Anthropic's circuit-tracing work ([Batson et al., 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)) locates hallucination in a competition between a refusal-to-speculate circuit and a known-entity detector. Every feature in this repo measures downstream symptoms of that competition, not the circuit itself — so a model doing confident motivated reasoning produces focused, low-entropy attention and scores as reliable. Closing that gap needs activation-level probing, e.g. [Cross-Layer Attention Probing](https://arxiv.org/abs/2509.09700), not better attention statistics.

### 8.2 Open Questions

Ordered roughly by what would most change the picture:

1. **Does calibration pay off under distribution *shift*, not just across models?** §5.1 shows `CalibratedEntropyDetector` transfers cleanly across two model families on the same domain (HaluEval QA) — the model-transfer question is answered. The domain-transfer question is not: train the calibrator on one domain, test on another.
2. **Is cross-layer KL salvageable?** It has now failed to contribute in three independent ablations across two models and two pipeline generations. Either there is a regime where the layer-dynamics story holds (larger models, longer contexts, RAG settings like [Bazarova et al., 2026](https://arxiv.org/abs/2504.10063)), or the framing should change.
3. **Why does feature-family importance flip between models (§5.5)?** Lookback dominates on Pythia and is negligible on Qwen; nothing dominates on Qwen at all. Is this instruction-tuning specific, scale-specific, or architecture-specific? A third model would start to distinguish these.
4. **Entropy under RLHF.** Instruction-tuned models are rewarded for confident-sounding output, which compresses the entropy distribution. Qwen2.5-0.5B-Instruct's results (§5.1) don't show obvious compression relative to Pythia, but this wasn't measured directly — a controlled comparison (same model family, base vs. instruct checkpoint) would isolate it.
5. **Multi-hop reasoning and snowballing.** Chains where every individual step is low-entropy but the composition is false are not captured by token-level entropy at all. [Zhang et al. (2023)](https://arxiv.org/abs/2305.13534) document this concretely: once a model commits to an early false claim, it generates confident, low-entropy justifications for it — and can independently recognize 67–87% of those justifications as wrong when asked to check its own work later. That gap between what the model *can* verify and what it *did* generate confidently is exactly the ceiling described in §8.1.
6. **Cross-lingual transfer.** Calibration is almost certainly language-specific; untested.
7. **Uncertainty vs. knowledge boundary.** Entropy reliably flags "outside training distribution," but conflates honest uncertainty with confident confabulation on conflicting training data. These are different failure modes deserving different responses.

---

## 9. Related Works (Bibliography)

**Foundations — information theory and uncertainty**
1. Shannon, C.E. (1948). [A Mathematical Theory of Communication](https://ieeexplore.ieee.org/document/6773024). *Bell System Technical Journal* 27(3):379–423.
2. Kullback, S. & Leibler, R.A. (1951). [On Information and Sufficiency](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full). *Annals of Mathematical Statistics* 22(1):79–86.
3. Gal, Y. & Ghahramani, Z. (2016). [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142). *ICML 2016.*
4. Malinin, A. & Gales, M. (2018). [Predictive Uncertainty Estimation via Prior Networks](https://arxiv.org/abs/1802.10501). *NeurIPS 2018.*
5. Malinin, A. & Gales, M. (2021). [Uncertainty Estimation in Autoregressive Structured Prediction](https://arxiv.org/abs/2002.07650). *ICLR 2021.*

**Calibration and selective prediction**
6. Platt, J. (1999). [Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods). *Advances in Large Margin Classifiers*, MIT Press, 61–74. — the parametric sigmoid alternative rejected in §4.3.
7. Zadrozny, B. & Elkan, C. (2002). [Transforming Classifier Scores into Accurate Multiclass Probability Estimates](https://dl.acm.org/doi/10.1145/775047.775151). *KDD 2002.* — isotonic calibration.
8. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). *ICML 2017.*
9. Geifman, Y. & El-Yaniv, R. (2017). [Selective Classification for Deep Neural Networks](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks). *NeurIPS 2017.* — the risk-coverage framing used in `abstention.py`.
10. Hanley, J. & McNeil, B. (1982). [The Meaning and Use of the Area under a ROC Curve](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747). *Radiology* 143(1):29–36. — AUROC ≡ Mann-Whitney U (§4.5).
11. Lee, K., Lee, K., Lee, H. & Shin, J. (2018). [A Simple Unified Framework for Detecting Out-of-Distribution Samples](https://arxiv.org/abs/1807.03888). *NeurIPS 2018.* — Mahalanobis-distance OOD scoring.

**Why this matters — stakes and mechanism**
12. Dahl, M., Magesh, V., Suzgun, M. & Ho, D.E. (2024). [Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models](https://academic.oup.com/jla/article/16/1/64/7699227). *Journal of Legal Analysis* 16(1):64–93. — 58% (GPT-4) to 88% (Llama 2) hallucination rates on verifiable federal case questions; models "struggle to predict their own hallucinations."
13. Singhal, K. et al. (2023). [Large Language Models Encode Clinical Knowledge](https://www.nature.com/articles/s41586-023-06291-2). *Nature* 620:172–180. — Med-PaLM improves with scale but remains "inferior to clinicians," motivating this repo's §1 framing.
14. Kalai, A.T., Nachum, O., Vempala, S.S. & Zhang, E. (2025). [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664). *arXiv:2509.04664.* (OpenAI) — a generation-vs-classification error bound and a training/eval-incentive argument for why hallucination is structural, not incidental; the motivation for treating it as an inference-time measurement problem (§1).
15. Zhang, M., Press, O., Merrill, W., Liu, A. & Smith, N.A. (2023). [How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534). *arXiv:2305.13534.* — models justify early false claims with confident, low-entropy text they can later recognize as wrong 67–87% of the time; the basis for §8.2's snowballing question.

**LLM hallucination — surveys and benchmarks**
16. Ji, Z. et al. (2023). [Survey of Hallucination in Natural Language Generation](https://dl.acm.org/doi/10.1145/3571730). *ACM Computing Surveys* 55(12).
17. Huang, L. et al. (2025). [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/abs/2311.05232). *ACM Transactions on Information Systems* 43(2):1–55.
18. Li, J., Cheng, X., Zhao, X., Nie, J.-Y. & Wen, J.-R. (2023). [HaluEval: A Large-Scale Hallucination Evaluation Benchmark](https://aclanthology.org/2023.emnlp-main.397/). *EMNLP 2023.* — the benchmark used throughout (see §5.2 for the matched-pair loading fix).

**Uncertainty-based hallucination detection**
19. Kadavath, S. et al. (2022). [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221). *arXiv:2207.05221.*
20. Manakul, P., Liusie, A. & Gales, M. (2023). [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection](https://aclanthology.org/2023.emnlp-main.557/). *EMNLP 2023.* — the multi-sample black-box alternative.
21. Kuhn, L., Gal, Y. & Farquhar, S. (2023). [Semantic Uncertainty](https://arxiv.org/abs/2302.09664). *ICLR 2023.*
22. Farquhar, S., Kossen, J., Kuhn, L. & Gal, Y. (2024). [Detecting Hallucinations in Large Language Models Using Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0). *Nature* 630:625–630.
23. Kossen, J. et al. (2024). [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927). *arXiv:2406.15927.* — single-pass approximation of semantic entropy.
24. Chen, C. et al. (2024). [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744). *ICLR 2024.* — EigenScore, covariance eigenvalues of response embeddings.
25. Phillips, E., Gustafsson, F.K., Wu, S., Thakur, A. & Clifton, D.A. (2026). [Entropy Alone is Insufficient for Safe Selective Prediction in LLMs](https://arxiv.org/abs/2603.21172). *arXiv:2603.21172.* — the direct motivation for §4.3's calibration layer.
26. Villani, M.J., Deshpande, P., Seshadri, A., Yalovetzky, R. & Kumar, N. (2026). [Entropy Distribution as a Fingerprint for Hallucinations in Generative Models](https://arxiv.org/abs/2605.28264). *arXiv:2605.28264.* — independent convergence on calibrated entropy-distribution scoring.
27. Vathul, A., Lee, D., Chen, S. & Tasmia, A. (2025). [ShED-HD: A Shannon Entropy Distribution Framework for Lightweight Hallucination Detection](https://arxiv.org/abs/2503.18242). *arXiv:2503.18242.* — BiLSTM over sequence-level entropy patterns.

**Attention-based detection**
28. Chuang, Y.-S., Qiu, L., Hsieh, C.-Y., Krishna, R., Kim, Y. & Glass, J. (2024). [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/). *EMNLP 2024.* — lookback-ratio features; the dominant family on one of §5.5's two models, negligible on the other.
29. Binkowski, J., Janiak, D., Sawczyn, A., Gabrys, B. & Kajdanowicz, T. (2025). [Hallucination Detection in LLMs Using Spectral Features of Attention Maps](https://arxiv.org/abs/2502.17598). *EMNLP 2025.* — LapEigvals; the spectral family.
30. Qi, S. et al. (2026). [Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention](https://arxiv.org/abs/2602.18145). *arXiv:2602.18145.* — the frequency family.
31. Bazarova, A. et al. (2026). [Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](https://arxiv.org/abs/2504.10063). *ACL 2026.* — TOHA.

**Interpretability context**
32. Tenney, I., Das, D. & Pavlick, E. (2019). [BERT Rediscovers the Classical NLP Pipeline](https://aclanthology.org/P19-1452/). *ACL 2019.* — layer-wise syntactic→semantic progression.
33. Batson, J. et al. (2025). [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). *Anthropic.* — the circuit-level account of hallucination behind §8.1.
34. Biderman, S. et al. (2023). [Pythia: A Suite for Analyzing Large Language Models](https://arxiv.org/abs/2304.01373). *ICML 2023.* — the default model used here.



Special thanks to UW Madison faculty (present and former) for teaching:

[Generative Models](https://github.com/AdaptInfer/dgm-fall-2025)

[LLMs From Scratch](https://github.com/rasbt/LLMs-from-scratch)

---

## 10. Citation

```bibtex
@software{kuo2026hallucination,
  author = {Kuo, Austin},
  title  = {Language Model Hallucination Detection via Entropy Divergence},
  url    = {https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence},
  year   = {2026},
  note   = {Single-pass entropy statistics and calibrated divergence for
            LLM hallucination detection at inference time}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form and [`paper/paper.tex`](paper/paper.tex) for the extended writeup.

---

## License

[MIT](LICENSE)

---

*Uncertainty is not failure. Undetected uncertainty is.*
