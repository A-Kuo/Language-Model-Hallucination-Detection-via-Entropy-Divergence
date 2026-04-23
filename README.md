# Language Model Hallucination Detection via Entropy Divergence

**Information-theoretic uncertainty signals for detecting when LLMs don't know what they're saying**

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

This work takes a different approach: **use the internal probability distribution of the model as a hallucination signal**, before the output is even decoded.

---

## Why Entropy Is a Better Signal

When a language model generates text, it doesn't just produce the most likely token — it maintains a probability distribution over its entire vocabulary at each step. The *shape* of that distribution is deeply informative.

**Low entropy** = the model's probability mass is concentrated. It has a strong, consistent prediction. Whether that prediction is correct is a separate question, but at minimum the model is not uncertain.

**High entropy** = probability mass is spread across many tokens. The model does not have a strong prediction. In factual domains, this correlates strongly with the model being outside its training distribution — i.e., hallucinating or confabulating.

This relationship is grounded in information theory: Shannon entropy H(p) = -Σ p(x) log p(x) is the expected surprise of a distribution. A model that "knows" the answer produces low-surprise next tokens. A model that is generating plausible-sounding text without grounded knowledge produces high-surprise distributions.

**Why this is better than output-text heuristics:**
- It operates on the model's internal state, not post-hoc text analysis
- It's architecture-agnostic — works on any autoregressive LLM
- It produces a continuous score, not a binary flag, enabling threshold tuning
- It's computed in a single forward pass with negligible overhead
- It generalizes across domains and languages without fine-tuning

---

## Quick Start — Run in Colab

The fastest way to get paper-quality results (no local setup required):

| Notebook | Purpose | Runtime | Badge |
|----------|---------|---------|-------|
| **[GPU Benchmark](COLABS.md)** | Generate paper numbers (AUROC, FPR, latency) | ~15 min T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/gpu_benchmark.ipynb) |
| **[Ablation Study](COLABS.md)** | Fill Table 2 (entropy-only vs KL-only vs both) | ~8 min T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/ablation_study.ipynb) |
| **[Quick Validation](COLABS.md)** | Test pipeline on CPU (GPT-2, synthetic data) | ~3 min CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/quick_cpu_validation.ipynb) |

**Auto-commit:** Add `GH_TOKEN` to Colab Secrets → results push directly to repo  
**See all:** [COLABS.md](COLABS.md) — full index with detailed instructions

---

## Professional Research Document

For a comprehensive academic treatment suitable for research statements, publications, and portfolio presentations, see **[RESEARCH.md](RESEARCH.md)**.

Includes:
- Detailed mathematical foundations (information theory, proof of bounds)
- Five-family feature engineering methodology
- Experimental results with confidence intervals and ablation studies
- Robustness evaluation against adversarial attacks
- Reproducibility instructions and API setup
- Comparison with related work

---

## Multi-Provider API Testing

Compare hallucination detection performance across different LLM providers (OpenAI, HuggingFace, Anthropic) using the new API testing framework:

```bash
cd v2/
python api_testing.py \
  --providers anthropic openai huggingface \
  --dataset halueval \
  --samples 100 \
  --output results/api_benchmark.json
```

Generates comparison table with AUROC, confidence intervals, and API costs per provider.

See [RESEARCH.md § 6.2](RESEARCH.md#62-running-benchmarks) for complete API setup instructions.

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

High `H_seq` indicates the model was uncertain throughout the generation. High `H_seq_max` indicates at least one token was highly uncertain — useful for detecting *insertion* of a fabricated entity into an otherwise confident generation.

### KL Divergence Between Forward Passes

A single entropy measurement can be noisy. A more robust signal uses **KL divergence between multiple stochastic forward passes** (using dropout at inference time or temperature sampling):

```
D_KL(p_pass1 || p_pass2) = Σ_{v} p_pass1(v) · log(p_pass1(v) / p_pass2(v))
```

If the model has genuine knowledge about a token, its distribution will be stable across passes (low KL). If the model is confabulating, small random perturbations will produce very different distributions (high KL). This is the **epistemic uncertainty** signal — it distinguishes "I know this confidently" from "I'm generating plausibly."

### Layer-Wise Entropy Divergence

An extended method measures entropy at multiple transformer layers. Early layers capture syntactic plausibility; later layers capture semantic and factual grounding. **Entropy that is high in late layers but low in early layers** is a signature of syntactic fluency without factual backing — the hallucination pattern in transformer architectures.

```
ΔH_layer = H_final_layer - H_intermediate_layer
```

Large positive `ΔH_layer` for a token: the model is syntactically confident but factually unanchored.

### Self-Referencing Score

A complementary method (used in the `self_reference` module): generate the claim, then ask the model to evaluate the claim's probability given its own context. The cross-entropy between the original generation and the self-referencing evaluation provides a calibration signal:

```
SR_score = H(p_original || p_self_eval)
```

Consistent generations will have low self-referencing entropy. Hallucinated claims will be inconsistently supported when the model examines them in context.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT PROMPT                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTOREGRESSIVE LLM (frozen)                       │
│                                                                     │
│  Token 1    Token 2    Token 3    ...    Token N                    │
│  ┌──────┐   ┌──────┐   ┌──────┐          ┌──────┐                  │
│  │p_t(v)│   │p_t(v)│   │p_t(v)│          │p_t(v)│                  │
│  └──┬───┘   └──┬───┘   └──┬───┘          └──┬───┘                  │
│     │          │           │                  │                      │
│     ▼          ▼           ▼                  ▼                      │
│  ┌──────┐   ┌──────┐   ┌──────┐          ┌──────┐                  │
│  │ H_1  │   │ H_2  │   │ H_3  │          │ H_N  │                  │
│  └──────┘   └──────┘   └──────┘          └──────┘                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTROPY AGGREGATION MODULE                        │
│                                                                     │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │  Sequence-Level  │   │  Token-Max        │   │  Layer-Wise    │  │
│  │  Mean Entropy    │   │  Spike Detection  │   │  Divergence    │  │
│  └────────┬─────────┘   └────────┬──────────┘   └──────┬─────────┘  │
│           └──────────────────────┴───────────────────────┘          │
│                                  │                                   │
│                                  ▼                                   │
│                     ┌────────────────────────┐                      │
│                     │  HALLUCINATION SCORE   │                      │
│                     │  [0.0 — 1.0 continuous]│                      │
│                     └────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      [ACCEPT output]  [FLAG for review]  [REJECT / retry]
```

---

## Connection to Information Theory Literature

This work builds on a lineage of uncertainty quantification research in deep learning:

**Foundational uncertainty decomposition:**
- Malinin, A. & Gales, M. (2018). "Predictive Uncertainty Estimation via Prior Networks." *NeurIPS 2018.* — Introduced the decomposition of uncertainty into aleatoric (data uncertainty) and epistemic (model uncertainty) components for neural networks. The KL divergence approach in this repo operationalizes the epistemic component for generative models.
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML 2016.* — Established MC dropout as a practical tool for epistemic uncertainty estimation. The multi-pass KL divergence method in this repo extends this to token-level uncertainty in autoregressive generation.

**LLM-specific calibration:**
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). "Semantic Entropy: Detecting Hallucinations in Large Language Models." *ICML 2023.* — Related approach using semantic clustering across multiple generations; this repo uses distributional divergence within single-model forward passes rather than cross-generation semantic clustering.
- Kadavath, S. et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv:2207.05221.* — Self-evaluation as calibration signal; related to the self-referencing module in this repo.

**Entropy in information theory (foundational):**
- Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal, 27*(3):379–423. — The entropy measure H(p) = -Σ p(x) log p(x) used throughout this work.
- Kullback, S. & Leibler, R.A. (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics, 22*(1):79–86. — KL divergence as a measure of distributional difference.

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

## Implementation

### Installation

```bash
git clone https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence.git
cd Language-Model-Hallucination-Detection-via-Entropy-Divergence
pip install -e .
```

Dependencies:
```
torch>=2.0
transformers>=4.35
numpy>=1.24
scipy>=1.10
```

### Usage Examples

#### Basic Token Entropy Scoring

```python
from hallucination_detection import EntropyScorer

scorer = EntropyScorer.from_pretrained("meta-llama/Llama-3-8b-hf")

prompt = "The capital of the Byzantine Empire was"
result = scorer.score(prompt, max_new_tokens=20)

print(f"Generation: {result.text}")
print(f"Mean entropy: {result.mean_entropy:.3f}")
print(f"Max token entropy: {result.max_entropy:.3f}")
print(f"Hallucination risk: {result.risk_level}")  # LOW / MEDIUM / HIGH
```

#### KL Divergence Across Passes

```python
from hallucination_detection import KLDivergenceScorer

scorer = KLDivergenceScorer(
    model_name="meta-llama/Llama-3-8b-hf",
    n_passes=5,           # number of stochastic forward passes
    dropout_rate=0.1      # inference-time dropout for epistemic uncertainty
)

result = scorer.score(
    prompt="Describe the mechanism by which mRNA vaccines produce immunity.",
    max_new_tokens=100
)

print(f"Mean KL divergence: {result.mean_kl:.4f}")
print(f"High-uncertainty tokens: {result.flagged_tokens}")
# e.g., ["binds", "spike", "ACE2"] — model is unstable on specific claims
```

#### Layer-Wise Entropy Analysis

```python
from hallucination_detection import LayerEntropyAnalyzer

analyzer = LayerEntropyAnalyzer(
    model_name="meta-llama/Llama-3-8b-hf",
    layers_to_probe=[8, 16, 24, 31]  # layer indices to instrument
)

result = analyzer.analyze(
    prompt="The GDP of Lichtenstein in 2019 was",
    max_new_tokens=10
)

# Plot entropy per layer per token
result.plot_entropy_surface("entropy_surface.png")
print(f"Late-layer entropy spike: {result.late_layer_spike}")
# High spike → syntactic fluency without factual anchoring
```

#### Batch Evaluation

```python
from hallucination_detection import HallucinationEvaluator
import pandas as pd

evaluator = HallucinationEvaluator(method="entropy_kl_combined")

prompts = df['question'].tolist()
scores = evaluator.evaluate_batch(prompts, batch_size=16, show_progress=True)

df['hallucination_risk'] = scores['risk_score']
df['entropy_mean'] = scores['mean_entropy']
df['flagged'] = scores['risk_level'].isin(['MEDIUM', 'HIGH'])
```

---

## Calibration

Raw entropy scores are not directly comparable across model sizes or domains. The calibration module fits a threshold function on a labeled held-out set:

```python
from hallucination_detection.calibration import IsotonicCalibrator

calibrator = IsotonicCalibrator()
calibrator.fit(
    entropy_scores=calibration_set['entropy'],
    labels=calibration_set['is_hallucination']  # binary labels from human annotation
)

# Apply calibrated threshold
calibrator.save("calibrators/llama3-8b-factual.pkl")

# Use in production
scorer = EntropyScorer.from_pretrained("meta-llama/Llama-3-8b-hf")
scorer.set_calibrator("calibrators/llama3-8b-factual.pkl")
```

---

## Research Limitations and Open Questions

### Current Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Calibration required per domain** | Entropy thresholds differ across knowledge domains; a score of 0.4 may indicate hallucination in historical facts but be normal in creative writing | Fit separate calibrators per use-case domain |
| **Confident confabulation** | A model can produce low-entropy hallucinations when it has seen similar (but incorrect) patterns many times in training | Pair with retrieval-augmented verification for high-stakes claims |
| **Layer probing requires white-box access** | Token-level and layer-wise methods require logit or hidden-state access; not available for API-only models | API-accessible models fall back to multi-generation KL divergence via sampling |
| **Computational overhead of multi-pass KL** | N forward passes is N× the compute cost of standard inference | Cache logits for hot prompts; use entropy scoring (single pass) as a pre-filter |
| **Does not localize the false claim** | The method scores *sequences* or *tokens*, not *propositions* — it cannot directly say "this specific entity name is hallucinated" | Use high-entropy token positions as candidate localization signals, then apply targeted retrieval |

### Open Research Questions

1. **Entropy under RLHF fine-tuning:** Models fine-tuned with RLHF learn to produce lower-entropy outputs as a reward-maximizing behavior. Does this compress the hallucination signal in the entropy distribution? Initial experiments suggest yes — recalibration after RLHF is necessary.

2. **Attention entropy vs. output entropy:** Output vocabulary entropy and attention pattern entropy are distinct signals. Attention entropy may provide earlier (lower-layer) detection of factual uncertainty. This has not been thoroughly benchmarked.

3. **Multi-hop reasoning hallucinations:** Complex reasoning chains where each individual step has low entropy but the composition produces a false conclusion are not well captured by token-level entropy. Sequence-level divergence across entire reasoning traces is an active research area.

4. **Cross-lingual generalization:** Entropy calibration is likely language-specific. A model may have different entropy profiles for English vs. lower-resource languages even when knowledge is equivalent.

5. **Hallucination vs. knowledge boundary:** Entropy reliably detects when the model is *outside* its training distribution. But this conflates genuine uncertainty (model was never trained on this) with confabulation (model was trained on conflicting or incorrect data). Distinguishing these requires additional signals.

---

## Current Status

**Beta (Q2 2026)**

- ✅ Token-level entropy scorer (single forward pass)
- ✅ KL divergence multi-pass scorer (MC dropout)
- ✅ Isotonic calibration for threshold fitting
- ✅ Batch evaluation pipeline
- ✅ Integration interfaces for crosscloud-ml and clinical-platform
- 🔄 Layer-wise entropy analyzer (experimental, requires transformer internals access)
- 🔄 Self-referencing score module
- ⏸️ Attention entropy probe (future work)
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
  note      = {Entropy-based uncertainty quantification for LLM hallucination detection at inference time}
}
```

**Related references:**
```bibtex
@inproceedings{malinin2018predictive,
  author    = {Malinin, Andrey and Gales, Mark},
  title     = {Predictive Uncertainty Estimation via Prior Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2018}
}

@inproceedings{gal2016dropout,
  author    = {Gal, Yarin and Ghahramani, Zoubin},
  title     = {Dropout as a {B}ayesian Approximation: Representing Model Uncertainty in Deep Learning},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  year      = {2016}
}

@inproceedings{kuhn2023semantic,
  author    = {Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  title     = {Semantic Entropy: Detecting Hallucinations in Large Language Models},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023}
}

@article{shannon1948mathematical,
  author  = {Shannon, Claude E.},
  title   = {A Mathematical Theory of Communication},
  journal = {Bell System Technical Journal},
  volume  = {27},
  number  = {3},
  pages   = {379--423},
  year    = {1948}
}
```

---

*Uncertainty is not failure. Undetected uncertainty is. April 2026.*
