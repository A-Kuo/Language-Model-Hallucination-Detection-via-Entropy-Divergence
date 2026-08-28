# Agent Instructions — Hallucination Detection

Instructions for AI agents extending or maintaining this project.

---

## Research Foundations (Mathematical)

Compressed reference — full derivations, "why this over alternatives" reasoning, and a defined notation table are in [README.md §4 "Method"](README.md#4-method). Notation here matches the README: `L` = number of layers, `H` = heads per layer, `T` = sequence length, `a` = an attention distribution (never the Laplacian — see item 4).

### 1. Shannon Entropy
```
H(a) = -Σ_i a_i · log₂(a_i)
```

### 2. Lookback Ratio — Chuang et al., EMNLP 2024
```
r = Σ_{context} a(i) / Σ_{all} a(i)     per head, per layer
```
Low lookback = not grounding in context = hallucination risk.

### 3. Frequency Domain — Qi et al., 2026
```
X_k = DFT(a)     →     E_high = Σ_{k>T/2} |X_k|²
```
Hallucinated tokens show high-frequency energy = fragmented grounding.

### 4. Spectral / Laplacian — Binkowski et al., EMNLP 2025 (LapEigvals)
```
Lap = D - W     →     eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λ_T
```
(`Lap`, not `L` — `L` is reserved for layer count everywhere in this repo; `feature_engineer.py`'s own variable is named `Lap` for the same reason.) Fiedler value (λ₂) indicates graph connectivity; low λ₂ = bottlenecks.

### 5. Cross-Layer KL
```
D_KL(layer_l || layer_{l+1}) = Σ p · log(p/q)
```

### 6. Single-Pass Token Entropy — `entropy_baselines.py`
```
H_t = -Σ_v p_t(v) · log p_t(v)          (mean/max/std over the answer span)
perplexity = exp(mean NLL of realized tokens)
```
Computed from output logits directly (white-box, no attention needed).

### 7. Calibrated Entropy Divergence — `calibrated_entropy_detector.py`
```
p(x) = w · isotonic(u(x)) + (1-w) · sigmoid(a·mahalanobis(x, μ_ref, Σ_ref) + b)
```
`μ_ref`/`Σ_ref` are fit on the calibration set's correct-answer (y=0) examples only. This is the repo's main original contribution — see README.md §4.3 for the full rationale.

### 8. Top-K Logprob Entropy (black-box) — `blackbox_detector.py`
Same entropy/margin/mass estimators as (6), computed only from the small top-K logprob list a commercial completions API returns — a documented lower bound on true entropy, not an approximation.

---

## Model Stack

- **Local model**: Default `EleutherAI/pythia-160m` (EleutherAI, Apache 2.0). Any HuggingFace causal LM works: Llama, Mistral, Phi, etc.
- **LLM-as-judge**: Claude (Anthropic API) for QA generation and answer labeling.
- **OpenAI**: optional, lazy-imported dependency used only by `blackbox_detector.py::fetch_topk_logprobs_openai()` for a live top-K logprob demo (requires `pip install openai` + `OPENAI_API_KEY`). Not required anywhere else — `simulate_topk_from_full_logits()` is the offline path everything else (including all tests) uses.

---

## Self-Data Pipeline

Claude generates QA → local model answers → Claude judges (correct/hallucinated) → feature engineer extracts 18D attention vector (+ 6D token-entropy vector via `entropy_baselines.py`) → train classifier. Scales linearly with API budget.

---

## Known Limitations: Anthropic Circuit Tracing

Batson et al. (2025) "On the Biology of a Large Language Model" identifies the causal mechanism of hallucination. Key implications:

### 1. We detect symptoms, not causes

Hallucination is governed by a binary competition: **refusal-to-speculate circuit** vs. **known-entity detector**. Hallucinations occur when the known-entity circuit misfires. Our feature families (attention-based and entropy-based alike) measure downstream signals, not the circuit itself. A future iteration should probe refusal and known-entity circuits via activation probing (CLAP, arXiv:2509.09700).

### 2. Confident hallucination evades detection

Models can produce **motivated reasoning** — fabricating internally consistent reasoning with focused, low-entropy attention. Our detector would score this "reliable." This is the fundamental ceiling of all attention-based methods. Addressing it requires hidden-state probing or external verification (RAG, entailment).

### 3. Discrete mechanism, continuous model

The refusal/known-entity competition is a discrete switch. Our logistic regression maps continuous features to probability — a reasonable approximation but misses the binary nature. Spectral features (Fiedler, spectral gap) may partially capture circuit switching; unvalidated.

### Implications

- Attention-based detection is a **first-pass filter**, not complete
- **Confident hallucination** requires complementary methods
- **Circuit-level probing** is the research frontier

---

## Extending This Work

- Add feature families from CHARM, Multi-View Attention papers
- Integrate activation probing for circuit-level signals
- Fit per-domain `CalibratedEntropyDetector` instances rather than one global calibration
- Get a proper cross-validated BiLSTM vs logistic regression comparison (see README §5.6 — the original comparison's corrupting bugs are fixed, but the held-out-only comparison now disagrees across two model families, which CV would help resolve)
- Scale past the two models now benchmarked (Pythia-160m, Qwen2.5-0.5B-Instruct — README §5.1): try something above ~0.5B params (Llama-3.2-1B+, Phi-3, Qwen2.5-1.5B+) to see whether the two models' *disagreeing* feature-family ablation (§5.5) starts converging with scale
- Semantic entropy / multi-sample UQ methods are deliberately out of scope here (this repo focuses on single-pass signals) — see README §9 for that literature

---

## References

1. Chuang et al. (2024). *Lookback Lens.* EMNLP.
2. Qi et al. (2026). *Frequency-Aware Attention.* arXiv:2602.18145.
3. Binkowski, J., Janiak, D., Sawczyn, A., Gabrys, B., Kajdanowicz, T. (2025). *Hallucination Detection in LLMs Using Spectral Features of Attention Maps.* EMNLP 2025. arXiv:2502.17598.
4. Multi-View Attention (2025). arXiv:2504.04335.
5. CHARM (2025). *Neural Message-Passing on Attention Graphs.* arXiv:2509.24770.
6. Shannon (1948). *A Mathematical Theory of Communication.*
7. Batson et al. (2025). *On the Biology of a Large Language Model.* Anthropic.
8. Templeton et al. (2025). *Circuit Tracing.* Anthropic.
9. Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models.* ICML.
