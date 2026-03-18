# Agent Instructions — Hallucination Detection v2

Instructions for AI agents extending or maintaining this project.

---

## Research Foundations (Mathematical)

### 1. Shannon Entropy (v1)
```
H(a) = -Σ p(i) · log₂(p(i))
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

### 4. Spectral / Laplacian — Barbero et al.
```
L = D - W     →     eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λ_T
```
Fiedler value (λ₂) indicates graph connectivity; low λ₂ = bottlenecks.

### 5. Cross-Layer KL (v1)
```
D_KL(layer_l || layer_{l+1}) = Σ p · log(p/q)
```

---

## Self-Data Pipeline

Claude generates QA → local model answers → Claude judges (correct/hallucinated) → feature engineer extracts 18D → train classifier. Scales linearly with API budget.

---

## Known Limitations: Anthropic Circuit Tracing

Batson et al. (2025) "On the Biology of a Large Language Model" identifies the causal mechanism of hallucination. Key implications:

### 1. We detect symptoms, not causes

Hallucination is governed by a binary competition: **refusal-to-speculate circuit** vs. **known-entity detector**. Hallucinations occur when the known-entity circuit misfires. Our 5 feature families measure downstream attention patterns, not the circuit. A future v3 should probe refusal and known-entity circuits via activation probing (CLAP, arXiv:2509.09700).

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

- Swap classifier (LogReg → MLP, or add ensemble)
- Add feature families from CHARM, Multi-View Attention papers
- Integrate activation probing for circuit-level signals
- Domain-specific calibration from corpus

---

## References

1. Chuang et al. (2024). *Lookback Lens.* EMNLP.
2. Qi et al. (2026). *Frequency-Aware Attention.* arXiv:2602.18145.
3. Barbero et al. (2025). *Spectral Features of Attention Maps.* arXiv:2502.17598.
4. Multi-View Attention (2025). arXiv:2504.04335.
5. CHARM (2025). *Neural Message-Passing on Attention Graphs.* arXiv:2509.24770.
6. Shannon (1948). *A Mathematical Theory of Communication.*
7. Batson et al. (2025). *On the Biology of a Large Language Model.* Anthropic.
8. Templeton et al. (2025). *Circuit Tracing.* Anthropic.
