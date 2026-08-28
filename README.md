# Language Model Hallucination Detection via Entropy Divergence

**Detecting hallucinations in LLM outputs using single-pass entropy statistics and divergence from calibrated entropy distributions.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

> *Just because you say it with confidence doesn't make it true. A model that is confidently wrong is more dangerous than a model that admits uncertainty. Given LLM uncertainty is a constant, the question is whether we can measure the moment it happens.*

---

## 1. Abstract

Large language models produce confident text even when they are wrong. In a casual chat this is annoying; in a clinical, legal, or financial setting it is a real problem. [Dahl et al. (2024)](https://academic.oup.com/jla/article/16/1/64/7699227) tested public LLMs against real federal court cases and found hallucination rates from 58% (GPT-4) to 88% (Llama 2) on specific, verifiable legal questions — and models "struggle to predict their own hallucinations." [Singhal et al. (2023)](https://www.nature.com/articles/s41586-023-06291-2) found the same scaling problem in medicine: comprehension and reasoning on clinical QA improve with model size, but Med-PaLM remained "inferior to clinicians," with the gap concentrated exactly in the cases you would least want to get wrong. Confident and wrong is the failure mode that matters, because it is the one a user has no local reason to double-check.

Nor is this a bug that better training quietly fixes. [Kalai et al. (2025)](https://arxiv.org/abs/2509.04664) (OpenAI) give a structural reason it persists: generation is provably harder than verification — a model's generative error rate is bounded below by roughly twice its classification error rate on the same question — and standard training/eval procedures reward confident guessing over calibrated abstention. That is an argument for treating hallucination as a property to be *measured and scored* at inference time, not one to be trained away, which is this repo's premise.

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

**What the results actually show (§5), stated up front.** On HaluEval QA with Pythia-160m, a plain logistic regression on 24 features reaches 0.973 AUROC, and the black-box detector — restricted to the top-5 logprobs an API would return — reaches 0.930. That black-box number is the strongest practical result here. The calibrated detector, which is the headline contribution, currently *underperforms* the linear baseline (0.915) on this in-distribution benchmark; the argument for why that is expected, and what would actually test it, is in §5.1. Two feature families are pulling their weight and three are not, including cross-layer KL divergence — the signal in the project's own name. These are honest negative results and they are reported as such.

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

## 3. What this repository implements

Four things:

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

$$H_t = -\sum_{v \in V} p_t(v) \log p_t(v), \qquad \mathrm{perplexity} = \exp\!\Big(\operatorname{mean}_t\big(-\log p_t(v_t^{\ast})\big)\Big)$$

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

with $u(x)$ a scalar raw score (default `entropy_mean`), $(a,b)$ fit by 1-D logistic regression of $d_\mathrm{M}$ against labels, and $w$ a fixed blend weight.

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

Two sets of numbers live in `results/`, from different pipeline generations. Both are real and reproducible; read §5.4 before drawing conclusions from either.

### 5.1 Current pipeline — `results/halueval_pythia160m_n400.json`

Pythia-160m, HaluEval QA, 400 balanced samples (200/200), CPU, seed 42. Stratified 5-fold cross-validation over all 400 samples, with bootstrap 95% CIs. Reproduce with:

```bash
python pipeline.py --halueval --num_samples 400 --model EleutherAI/pythia-160m \
    --results results/halueval_pythia160m_n400.json
```

| Detector | Features | CV AUROC | 95% CI |
|---|---|---|---|
| **Logistic regression** | 24D (18 attention + 6 token-entropy) | **0.9728** | [0.9545, 0.9870] |
| MLP | 24D | 0.9662 | [0.9452, 0.9826] |
| `BlackBoxEntropyDetector` | 7D top-K logprobs only | 0.9304 | [0.9038, 0.9529] |
| `CalibratedEntropyDetector` | 24D | 0.9149 | [0.8839, 0.9398] |

**Interpretation — the black-box result is the interesting one.** `BlackBoxEntropyDetector` reaches 0.9304 AUROC using *only* the top-5 logprobs an API would return: no attention weights, no full-vocabulary distribution. That is within 0.04 AUROC of the full white-box linear model while discarding almost all of its input. For a method meant to run against a commercial API, that gap is the number that matters, and it is small.

**The calibrated detector currently underperforms the plain linear baseline** (0.9149 vs 0.9728), and its CI does not overlap the linear model's. This is an honest negative result for the repo's headline contribution *in this configuration*. The most likely explanation is that the calibration layer is solving a problem this benchmark does not have: it exists to make scores transfer across domains and models (§4.3), and a single-domain, single-model, in-distribution evaluation gives it no opportunity to pay that off while still charging the variance cost of fitting an isotonic map and a covariance on limited calibration data. The claim it is built on — from [Phillips et al. (2026)](https://arxiv.org/abs/2603.21172) — is about *risk-coverage behavior under distribution shift*, which this table does not measure. Testing it properly needs a cross-domain or cross-model split; see §5.4.

Held-out split (120 samples, single train/test split rather than CV):

| Detector | AUROC | F1 | FPR |
|---|---|---|---|
| BiLSTM (per-layer sequences) | 0.9860 | 0.9630 | 0.0000 |
| MLP | 0.9797 | 0.9241 | 0.1600 |
| Logistic regression | 0.9794 | 0.9371 | 0.1200 |

**The BiLSTM number is a genuine reversal, but do not over-read it.** The historical result (§5.2) had the BiLSTM at 0.78 AUROC, well behind the linear baseline, and that gap was the repo's standing open question. Two bugs found and fixed since then account for much of it: a label/sequence misalignment in the BiLSTM path, and a `context_length` error that made the lookback-ratio feature degenerate (it measured attention to the whole sequence rather than to the prompt). After the fixes the BiLSTM leads on the held-out split. **However** — this is a single 120-sample split, not cross-validated, while the flat detectors above have 5-fold CV numbers. A 0.006 AUROC lead on 120 samples is not a result. The honest summary is that the BiLSTM is no longer obviously *worse*, and deserves a proper CV comparison it does not yet have.

**Feature-family ablation** (drop one family, retrain logistic regression; full 24D model = 0.9794 held-out AUROC):

| Family removed | AUROC | Δ |
|---|---|---|
| Lookback | 0.9554 | **+0.0240** |
| Entropy (attention) | 0.9720 | +0.0074 |
| Frequency | 0.9786 | +0.0009 |
| Cross-layer KL | 0.9806 | −0.0011 |
| Spectral | 0.9846 | −0.0051 |
| Token entropy | 0.9877 | −0.0083 |

**Interpretation:** only the lookback family clearly earns its place. Removing it costs 0.024 AUROC — an order of magnitude more than any other family. Three families (cross-layer KL, spectral, token entropy) have *negative* deltas, meaning the model scored slightly better without them; at this sample size those are noise, but they are certainly not evidence of contribution. This is corroborated by the logistic regression's own weights, where `lb_ratio_entropy` (0.89) and `lb_ratio_mean` (0.67) sit at the top alongside `ent_std` (0.83).

That lookback dominates is consistent with [Chuang et al. (2024)](https://aclanthology.org/2024.emnlp-main.84/), whose entire method is the lookback ratio. It is *awkward* for this repo specifically, since cross-layer KL divergence is in the project's name and is contributing nothing measurable here.

### 5.2 Historical benchmark — `results/benchmark_results.json`

T4 GPU, April 2026, Pythia-160m, HaluEval QA, 500 samples. This predates the current feature stack: it used an earlier two-signal "AED" configuration (per-head entropy + cross-layer KL only), run through notebooks that are now broken (§6.3).

| Detector | AUROC | F1 | FPR@90%TPR | Latency |
|---|---|---|---|---|
| Logistic regression | 0.9068 | 0.8148 | 0.30 | — |
| BiLSTM ("AED") | 0.7808 | 0.8264 | 0.38 | 38.49 ms/sample |

Companion ablation (`results/ablation_results.json`), on those two signals only:

| Configuration | AUROC | FPR@90%TPR |
|---|---|---|
| Entropy only | 0.8225 | 0.30 |
| KL only | 0.5400 | 0.60 |
| Both | 0.8300 | 0.30 |

**Interpretation:** KL divergence alone scored 0.54 — barely above the 0.50 chance line — and adding it to entropy bought +0.0075 AUROC. Combined with §5.1's ablation, where cross-layer KL again contributes nothing measurable, this is now two independent runs a year apart pointing the same direction. The signal the repo is named after is the weakest one in it. That is worth stating plainly rather than burying.

A third file, `results/benchmark_results_cpu_quick.json` (GPT-2, 50 samples, CPU), exists only to prove the pipeline executes; at that sample size the numbers are noise.

### 5.3 Latency

38.49 ms/sample on a T4 (§5.2), for one forward pass with `output_attentions=True` plus feature extraction. The comparison that matters: semantic-entropy methods ([Farquhar et al., 2024](https://www.nature.com/articles/s41586-024-07421-0)) need 5–10 generations per query. Single-pass detection is roughly an order of magnitude cheaper, which is the whole argument for accepting a weaker signal.

### 5.4 Caveats — read before citing any of this

1. **The HaluEval split is not matched-pair, and this likely inflates every AUROC above.** The original HaluEval design pairs each question with *both* a correct and a hallucinated answer, which controls for question content and isolates the hallucination signal. The mirror this repo loads (`pminervini/HaluEval`, `qa_samples`) has 10,000 rows with **10,000 unique questions and zero matched pairs** — verified directly. So the "correct" and "hallucinated" classes are entirely disjoint question sets, and a detector can score well by picking up surface differences between those two populations rather than hallucination as such. **This is the single biggest threat to the validity of §5.1**, and fixing it (loading the paired HaluEval format) should come before any further tuning.
2. **Sample sizes are small.** 400 samples with a 120-sample held-out split. The CIs in §5.1 are wide enough that most between-detector gaps there are not individually significant.
3. **One model, one domain.** Everything is Pythia-160m on HaluEval QA. Nothing here speaks to cross-model or cross-domain transfer — which, as noted in §5.1, is exactly the regime `CalibratedEntropyDetector` was designed for and has therefore not actually been tested in.
4. **No committed abstention/risk-coverage curve on real data.** `abstention.py` runs, but no result artifact is committed.
5. **The adversarial-robustness claim is unmeasured.** `adversarial.py` implements obfuscation, paraphrase, and multilingual-prefix attacks; no committed results file records the outcome.

`notebooks/real_pipeline_benchmark/` exists to run §5.1 at a scale and model size that a laptop CPU cannot reach; see §6.3.

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

HaluEval benchmark with a local model (needs `pip install torch transformers datasets`):

```bash
python pipeline.py --halueval --num_samples 500 --model EleutherAI/pythia-160m --results results/my_run.json
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

### 6.2 Kaggle GPU runner — the current workflow

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
| `notebooks/real_pipeline_benchmark/` | **Working, not yet run at scale** | Clones the repo and calls the real `pipeline.py::run_real_pipeline()` on real HaluEval data. This is what will fill the §5.4 gaps. |
| `notebooks/full_pipeline.ipynb` | Working | End-to-end Claude-labeled data generation → features → classifier (needs `ANTHROPIC_API_KEY`) |
| `notebooks/gpu_benchmark.ipynb` | **Broken** | Produced `results/benchmark_results.json`; imports the removed `run_experiment.py` |
| `notebooks/ablation_study.ipynb` | **Broken** | Produced `results/ablation_results.json`; same broken import |
| `notebooks/quick_cpu_validation.ipynb` | **Broken** | Same broken import |

The three broken notebooks are the ones that generated §5's numbers, back when the repo had a separate `run_experiment.py` module. That module was removed during consolidation and the notebooks were never ported. They need rewriting against the current `pipeline.py` API, or retiring in favor of `real_pipeline_benchmark`.

---

## 7. Repository layout

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
├── paper/                          # arXiv paper source (paper.tex, references.bib)
├── results/                        # Committed benchmark JSON (see §5)
├── AGENT.md                        # Design notes, math reference, known limitations
└── requirements.txt
```

Core dependencies are `numpy` and `scipy` only. `torch`/`transformers` (real models), `datasets` (HaluEval), `anthropic` (data generation), `openai` (black-box demo), `chromadb`/`sentence-transformers` (embedding anomaly), and `kaggle` (CI) are all optional and lazily imported — tests that need them skip cleanly when they are absent.

---

## 8. Limitations and open questions

### 8.1 Known limitations

| Limitation | Why it happens | Current mitigation |
|---|---|---|
| **Benchmark is not matched-pair** | The loaded HaluEval mirror has zero paired questions (§5.4), so correct/hallucinated classes are disjoint question sets and reported AUROC is likely inflated | Load the paired HaluEval format — this is the highest-priority fix |
| **The calibration layer is untested where it should help** | It underperforms the linear baseline in-distribution (§5.1); its purpose is cross-domain transfer, which nothing here measures | Build a cross-domain or cross-model evaluation split |
| **Calibration is domain-specific by design** | Entropy scale shifts across domains; a threshold tuned on factual QA does not transfer to creative writing | Fit a separate `CalibratedEntropyDetector` per domain |
| **Confident confabulation** | A model can produce low-entropy *and wrong* output when training data reinforced an incorrect pattern | Attention features are a first-pass filter, not a solution; pair with retrieval for high-stakes claims |
| **White-box access required for most features** | Attention and full-logit features need model internals, unavailable via API | `BlackBoxEntropyDetector` falls back to top-K logprobs, with strictly less information (§4.4) |
| **Cross-layer KL contributes nothing measurable** | Two independent runs (§5.1, §5.2) show it near-zero or negative in ablation, despite naming the project | Either find the regime where it helps, or drop it and rename the framing around lookback + entropy |
| **No claim localization** | The detectors score *sequences*, not propositions — "this answer is risky", not "this span is wrong" | Use high-entropy token positions as candidate spans, then targeted retrieval |

**The confident-confabulation limitation is the most fundamental one.** Anthropic's circuit-tracing work ([Batson et al., 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)) locates hallucination in a competition between a refusal-to-speculate circuit and a known-entity detector. Every feature in this repo measures downstream symptoms of that competition, not the circuit itself — so a model doing confident motivated reasoning produces focused, low-entropy attention and scores as reliable. Closing that gap needs activation-level probing, e.g. [Cross-Layer Attention Probing](https://arxiv.org/abs/2509.09700), not better attention statistics.

### 8.2 Open questions

Ordered roughly by what would most change the picture:

1. **Does the matched-pair split change everything?** §5.4's first caveat is the one that could move every number in §5.1. Re-running on properly paired HaluEval — same question, correct vs. hallucinated answer — removes the confound and gives a real measurement.
2. **Does calibration pay off under distribution shift?** The whole argument for §4.3 is cross-domain transfer, and the current evaluation cannot see it. Train the calibrator on one domain, test on another.
3. **Is cross-layer KL salvageable?** It has now failed to contribute in two ablations. Either there is a regime where the layer-dynamics story holds (larger models, longer contexts, RAG settings like [Bazarova et al., 2026](https://arxiv.org/abs/2504.10063)), or the framing should change.
4. **Entropy under RLHF.** Instruction-tuned models are rewarded for confident-sounding output, which compresses the entropy distribution. Does that compress the hallucination signal with it?
5. **Multi-hop reasoning and snowballing.** Chains where every individual step is low-entropy but the composition is false are not captured by token-level entropy at all. [Zhang et al. (2023)](https://arxiv.org/abs/2305.13534) document this concretely: once a model commits to an early false claim, it generates confident, low-entropy justifications for it — and can independently recognize 67–87% of those justifications as wrong when asked to check its own work later. That gap between what the model *can* verify and what it *did* generate confidently is exactly the ceiling described in §8.1.
6. **Cross-lingual transfer.** Calibration is almost certainly language-specific; untested.
7. **Uncertainty vs. knowledge boundary.** Entropy reliably flags "outside training distribution," but conflates honest uncertainty with confident confabulation on conflicting training data. These are different failure modes deserving different responses.

---

## 9. Related work

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
18. Li, J., Cheng, X., Zhao, X., Nie, J.-Y. & Wen, J.-R. (2023). [HaluEval: A Large-Scale Hallucination Evaluation Benchmark](https://aclanthology.org/2023.emnlp-main.397/). *EMNLP 2023.* — the benchmark used throughout (see §5.4 for a caveat on the mirror this repo loads).

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
28. Chuang, Y.-S., Qiu, L., Hsieh, C.-Y., Krishna, R., Kim, Y. & Glass, J. (2024). [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/). *EMNLP 2024.* — lookback-ratio features; the strongest-contributing family in §5.1's ablation.
29. Binkowski, J., Janiak, D., Sawczyn, A., Gabrys, B. & Kajdanowicz, T. (2025). [Hallucination Detection in LLMs Using Spectral Features of Attention Maps](https://arxiv.org/abs/2502.17598). *EMNLP 2025.* — LapEigvals; the spectral family.
30. Qi, S. et al. (2026). [Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention](https://arxiv.org/abs/2602.18145). *arXiv:2602.18145.* — the frequency family.
31. Bazarova, A. et al. (2026). [Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](https://arxiv.org/abs/2504.10063). *ACL 2026.* — TOHA.

**Interpretability context**
32. Tenney, I., Das, D. & Pavlick, E. (2019). [BERT Rediscovers the Classical NLP Pipeline](https://aclanthology.org/P19-1452/). *ACL 2019.* — layer-wise syntactic→semantic progression.
33. Batson, J. et al. (2025). [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). *Anthropic.* — the circuit-level account of hallucination behind §8.1.
34. Biderman, S. et al. (2023). [Pythia: A Suite for Analyzing Large Language Models](https://arxiv.org/abs/2304.01373). *ICML 2023.* — the default model used here.

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
