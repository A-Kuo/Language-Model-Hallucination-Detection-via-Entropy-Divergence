# Paper Outline — [Working Title TBD]

Structure only. No prose content yet — each line is a one-sentence scope note
for what that section will argue or report, not a draft of it. This replaces
the old `paper.tex` skeleton, which was written around a two-signal
(entropy + cross-layer KL) system and predates the current 5-family feature
stack, the calibrated/black-box detectors, and the real HaluEval numbers in
`results/halueval_pythia160m_n400.json`.

**Working title candidates** (pick one before drafting):
- *Calibrated Entropy Divergence for Single-Pass Hallucination Detection*
- *What Attention and Entropy Actually Tell You About Hallucination: A Multi-Family Ablation*
- Something that doesn't center "entropy divergence" if §7 confirms cross-layer KL doesn't earn its place in the final framing.

---

## Abstract
- One paragraph, written last. Must state the two headline numbers (LogReg 0.973 CV AUROC; black-box top-K 0.930) and the one honest negative result (calibrated detector underperforms in-distribution) — abstracts that hide the negative result don't survive review.

## 1. Introduction
- 1.1 The stakes — why confident-and-wrong matters (legal/clinical hallucination rates, real citations already sourced: Dahl et al. 2024, Singhal et al. 2023).
- 1.2 Why hallucination detection at inference time, not just better training (Kalai et al. 2025's generation-vs-classification argument).
- 1.3 The single-pass constraint — cost of multi-sample methods (semantic entropy, SelfCheckGPT) as the reason this line of work exists.
- 1.4 What's actually novel here, stated plainly: (a) five attention-feature families evaluated together rather than in isolation, (b) a calibrated-divergence detector as an alternative to thresholding, (c) a black-box variant tested against the white-box ceiling, (d) an ablation that reports which families don't help — not just which do.
- 1.5 Contributions list (bulleted, 3–5 items, each mapped to a later section/table).

## 2. Related Work
- 2.1 Uncertainty estimation foundations (Shannon; Gal & Ghahramani; Malinin & Gales).
- 2.2 LLM hallucination — surveys and stakes (Ji et al.; Huang et al.; Dahl et al.; Kalai et al.).
- 2.3 Sampling-based detection (semantic entropy, SelfCheckGPT, SEPs) — positioned as the higher-cost alternative this work trades against.
- 2.4 Internal-state / white-box detection (INSIDE/EigenScore, Kadavath et al.).
- 2.5 Attention-based detection specifically (Lookback Lens, spectral/LapEigvals, frequency-aware, TOHA) — each family in §4 traces to one of these.
- 2.6 Calibration and selective prediction (Platt; isotonic/Zadrozny & Elkan; Guo et al.; Geifman & El-Yaniv) — the machinery §4's calibrated detector is built from.
- 2.7 Gap statement: closes with what none of the above do together (multi-family fusion + calibration + an honest ablation of what doesn't work), which is the paper's actual claim to novelty.

## 3. Problem Setup
- 3.1 Task definition — binary hallucination detection given model + prompt + generated answer, no ground truth at inference time.
- 3.2 Notation table (L, H, T, V, attention tensor, token distribution) — reused verbatim from README §4's notation table so the paper and repo stay in sync.

## 4. Method
- 4.1 Attention feature families (5, 18D total) — one paragraph per family: entropy, lookback ratio, frequency/DFT, spectral/Laplacian, cross-layer KL. Each gets a formula + one-sentence rationale, ported from README §4.1.
- 4.2 Token-level entropy baselines (6D) — why six statistics, not one (README §4.2's per-statistic table).
- 4.3 Calibrated Entropy Divergence (the headline method) — isotonic calibration + Mahalanobis divergence from a reference distribution; explicit "why not Platt / why not Euclidean" subsections, ported from README §4.3.
- 4.4 Black-box top-K detector — the API-realistic variant; the truncated-entropy lower-bound argument with the worked numeric example.
- 4.5 Classifiers — logistic regression / MLP / BiLSTM, stated briefly (these are standard; not a contribution in themselves).
- 4.6 AUROC estimator note — trapezoidal vs. Mann-Whitney U equivalence, one paragraph (matches README §4.5).

## 5. Experimental Setup
- 5.1 Dataset — HaluEval QA via the `pminervini/HaluEval` mirror. **Must explicitly disclose the no-matched-pairs finding here as a methods-section limitation, not bury it in discussion** — this is a threat to internal validity, and readers need it before they see the results table, not after.
- 5.2 Model — Pythia-160m; note single-model scope as a stated limitation, not an omission.
- 5.3 Evaluation protocol — stratified 5-fold CV with bootstrap 95% CIs; separate 120-sample held-out split for the BiLSTM/family-ablation comparisons; state plainly that these are two different evaluation regimes and why (CV needs many folds, BiLSTM sequence data was only prepared for the held-out split).
- 5.4 Baselines and what each comparison isolates (linear vs MLP vs BiLSTM; white-box vs black-box; calibrated vs uncalibrated).

## 6. Results
- 6.1 Main detection table — CV AUROC + CI for LogReg / MLP / CalibratedEntropy / BlackBoxTopK (the real numbers already in `results/halueval_pythia160m_n400.json`).
- 6.2 Held-out comparison — LogReg / MLP / BiLSTM, with the caveat that this is a single split, not CV.
- 6.3 Feature-family ablation table — full model vs. drop-one-family deltas.
- 6.4 Feature importance (top logistic-regression weights) as corroborating evidence for §6.3.
- 6.5 Historical benchmark comparison — briefly reproduce the April-2026 two-signal numbers (`results/benchmark_results.json`) as a "before" reference point, explicitly framed as measuring a different, since-fixed pipeline (label/sequence-alignment and context-length bugs), not as a stronger result being buried.
- 6.6 Latency — single-pass cost vs. the 5–10x cost of sampling-based methods.

## 7. Analysis and Discussion
- 7.1 Why lookback ratio dominates the ablation and the other four families mostly don't — connect to Chuang et al.'s original finding that a linear probe on lookback ratio alone rivals richer detectors.
- 7.2 The calibrated-divergence negative result — argue explicitly that an in-distribution, single-domain benchmark cannot show what this method is for (cross-domain transfer), and that a fair test requires the experiment in §9.1. Do not let this read as an excuse; state what evidence would change the conclusion.
- 7.3 Cross-layer KL — two independent negative ablations now (this paper's and the earlier two-signal one). State directly that the theoretical motivation (layer-wise representation refinement, Tenney et al.) has not yet found empirical support in this setup, and consider what regime might change that (longer contexts, larger models, RAG settings — per Bazarova et al.'s topological-divergence result, which found signal in a related but distinct formulation).
- 7.4 Black-box vs. white-box gap — the practical takeaway: how much performance survives when only top-K logprobs are visible, and what that implies for deployment against commercial APIs.

## 8. Limitations
- 8.1 Matched-pair confound (restated from §5.1, cross-referenced, not re-argued).
- 8.2 Single model, single domain — no cross-model or cross-domain evidence anywhere in this paper.
- 8.3 No claim-level localization — sequence-level scores only.
- 8.4 The confident-confabulation ceiling — attention/entropy features are downstream of the causal mechanism (Batson et al.'s circuit-tracing account), not the mechanism itself; a model executing confident motivated reasoning is invisible to every method in this paper.
- 8.5 No committed adversarial-robustness or abstention/risk-coverage results despite the code existing (`adversarial.py`, `abstention.py`) — state as future work, not as a hidden gap.

## 9. Future Work
- 9.1 Matched-pair HaluEval re-run — the highest-priority follow-up, since it could change every number in §6.
- 9.2 Cross-domain / cross-model calibration test — the experiment that would actually validate or falsify §4.3's premise.
- 9.3 Larger models (Pythia-1.4B+, Llama-3) — scale check.
- 9.4 Claim localization via high-entropy token spans + targeted retrieval.

## 10. Conclusion
- Restate the two honest headline results (black-box near-parity; calibration underperforming where it can't show its advantage) and the one clear methodological lesson (ablate everything, report what doesn't help) without resolving the open questions the paper raises.

## References
- Full `references.bib`, already at 30+ verified entries; no new work needed here beyond keeping it in sync with README §9.

## Appendix
- A.1 Hyperparameters (classifier configs, isotonic/Mahalanobis blend weight, shrinkage constant).
- A.2 Feature dimension reference table (which family produces which named features — mirrors `feature_engineer.py` docstrings).
- A.3 Compute/environment details (CPU vs T4 GPU runs, wall-clock per experiment).
- A.4 Full per-fold CV numbers, if space allows (supplementary rather than main-text).

---

## Open decisions before drafting begins

1. **Title** — depends on how §7.3's cross-layer KL discussion lands; if it stays a clear non-contributor, a title centered on "entropy divergence" oversells the one family that underperforms.
2. **Venue/format** — arXiv-only (no page limit, can keep full ablation detail) vs. a workshop submission (would force cutting §6.5's historical comparison and most of the appendix).
3. **Whether §6.5 (historical benchmark) belongs in the main paper at all**, or should move to the appendix as provenance/reproducibility material — it's honest and useful but is about a superseded pipeline, not the system being evaluated.
4. **Order of §7.1–7.4** — currently ordered by ablation magnitude (biggest effect first); could instead order by "what a reader deploying this system needs to know first" (black-box gap → calibration caveat → family-level detail), which might read better for a practitioner audience.
