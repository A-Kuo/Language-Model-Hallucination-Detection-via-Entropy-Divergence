# Agent Instructions — Hallucination Detection v1

Instructions for AI agents extending or maintaining this project.

---

## Mathematical Foundation

#### 1. Attention Entropy (per-head uncertainty)

```
H(head) = -Σᵢ p(i) · log₂(p(i))
```

- **Low entropy** → peaked attention → likely reliable
- **High entropy** → diffuse attention → possibly hallucinating

#### 2. Cross-Layer KL Divergence (internal consistency)

```
D_KL(layer_l || layer_{l+1}) = Σᵢ p_l(i) · log(p_l(i) / p_{l+1}(i))
```

#### 3. Hypothesis Test

```
H₀: output is RELIABLE
H₁: output is UNRELIABLE (hallucination)

Z_combined = 0.40·Z_entropy + 0.45·Z_kl + 0.15·Z_spread
where Z_x = (observed_x - μ_baseline) / σ_baseline
```

KL gets highest weight (0.45) — cross-layer disagreement is the strongest indicator.

#### 4. Confidence Calibration (isotonic regression)

```
f: raw_score → calibrated_score  (monotonic, via PAV)
```

#### 5. Three-Tier Routing

```
confidence > 0.75  →  RELIABLE
0.50 ≤ conf ≤ 0.75 →  UNCERTAIN (escalate)
confidence < 0.50  →  UNRELIABLE (reject)
```

---

## Module Deep Dives

### `attention_analyzer.py`

- Loads Gemma 2 (or any HuggingFace model), runs forward pass with `output_attentions=True`
- **Layer-head entropy matrix** `(L, H)`: Shannon entropy of last token's attention row per head
- **Cross-layer KL**: pairwise KL between consecutive layers' averaged attention
- Standalone `compute_entropy_from_weights()` and `compute_kl_divergence()` work with raw numpy — no model needed

### `hypothesis_test.py`

- **Null hypothesis H₀**: output is reliable
- **Test statistic**: weighted Z_entropy + Z_kl + Z_spread
- **Decision**: reject H₀ if Z > z_critical at α
- `calibrate_from_corpus()` for domain adaptation

### `confidence_calibrator.py`

1. **IsotonicCalibrator**: Pure-numpy PAV (Zadrozny & Elkan, 2002)
2. **ConfidenceCalibrator**: Applies calibration + 3-tier routing with configurable thresholds

### `utils.py`

TokenizationHelper, setup_logger, Timer, serialize_result/deserialize_result, batch_texts.

---

## Test Suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_attention_analyzer.py` | 32 | Entropy bounds, KL properties, Gibbs' inequality, numerical stability, torch tensor tests |

Key mathematical properties verified: entropy non-negative and bounded by log₂(T), entropy monotonically increasing with uniformity, KL ≥ 0 (Gibbs' inequality), KL asymmetric, numerical stability with near-zero weights.

Each `.py` module also contains a `__main__` self-test block for standalone validation of hypothesis test scenarios, calibrator routing, and isotonic regression.

---

## Design Decisions

- **Attention entropy**: Interpretable, fast, theoretically grounded. Tradeoff: won't catch "confidently wrong" hallucinations.
- **KL across layers**: Captures internal inconsistency; stronger signal than entropy alone.
- **Conservative thresholds**: Bias toward escalating (false positive) over trusting (false negative).
- **Isotonic over Platt**: No parametric assumptions; Z-score-based confidence isn't sigmoid.
- **Gemma 2**: Public, compact (2B), GPU/CPU-friendly. Technique generalizes to any transformer with `output_attentions`.

---

## Extending This Work

- **Domain calibration**: `calibrate_from_corpus()` for model/domain adaptation
- **Multi-model**: Swap Gemma 2 for any HuggingFace model with attention weights
- **Streaming**: <5ms latency supports token-by-token monitoring
- **Ensemble**: Combine with output probability, embedding similarity, RAG verification

---

## References

- Shannon (1948). *A Mathematical Theory of Communication*
- Kullback & Leibler (1951). *On Information and Sufficiency*
- Zadrozny & Elkan (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*
- Vaswani et al. (2017). *Attention Is All You Need*
- Xiao & Wang (2021). *On Hallucination and Predictive Uncertainty in Conditional Language Generation*
