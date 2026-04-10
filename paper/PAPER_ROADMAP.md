# Paper Roadmap: AED — Attention Entropy Divergence
## What to do before submitting to arXiv

**Estimated time to submission-ready: 2–3 weekends of focused work.**

---

## The arXiv No-PhD Question — Answered Directly

**You do not need a PhD. You do not need an institution.**

arXiv requires:
1. An account (free, anyone can register)
2. One endorsement from an **existing arXiv submitter** in cs.LG or cs.AI

**How to get endorsed (from easiest to harder):**

| Option | How | Success rate |
|--------|-----|-------------|
| **r/MachineLearning** | Post: "working on hallucination detection paper, looking for an arXiv endorser in cs.LG. Here's a draft: [link]" | High — the ML community endorses independent researchers regularly |
| **Alignment Forum / LessWrong** | AI safety community is uniquely receptive to independent work on hallucination/reliability | High for safety-adjacent work |
| **HuggingFace forums** | Post about the work with code link | Medium-high |
| **LinkedIn / Twitter/X** | Share the draft with the ML community | Medium |
| **Cold email** | Find a researcher who works on LLM reliability and email them the draft | Medium — requires a polished draft first |

**Author line to use:** `A-Kuo, Independent Researcher`

This is standard and respectable. Many cited papers use this exact designation.
Eliezer Yudkowsky, Gwern Branwen, and many others with significant ML contributions 
use "Independent Researcher" and are widely cited.

---

## What the Paper Already Has

The codebase is strong. Here's what exists vs. what's needed:

| Element | Status |
|---------|--------|
| Mathematical formulation | ✅ Fully documented in attention_analyzer.py |
| Shannon entropy computation | ✅ Implemented, tested, self-verified |
| Cross-layer KL divergence | ✅ Implemented, tested |
| Hypothesis testing framework | ✅ Full statistical framework in hypothesis_test.py |
| BiLSTM classifier | ✅ Full implementation in v2/detector.py |
| Logistic regression baseline | ✅ Implemented from scratch (numpy-only) |
| HaluEval benchmark | ✅ Referenced in detector.py docstring |
| Colab notebooks | ✅ v1_benchmark.ipynb and v2_full_pipeline.ipynb |
| CITATION.cff | ✅ Already in repo |
| CI/CD | ✅ GitHub Actions test.yml |
| AUROC number cited | ✅ BiLSTM ~0.96 vs LR ~0.84 in detector.py docstring |

---

## What Needs Doing (in priority order)

### 1. Run the benchmarks and record real numbers (1–2 days)
**This is the most important thing.** The paper currently has placeholder TODOs for F1, FPR@90%TPR, ablation scores, and latency numbers. These must be real numbers from actual runs.

```bash
# Run the Colab notebook locally or on Colab GPU
cd hallu-repo
# Follow colab/v1_benchmark.ipynb
```

Fill in the TODO placeholders in paper.tex with actual results.

**Minimum numbers needed:**
- Table 1: F1 and FPR@90%TPR for all 3 methods
- Table 2 (ablation): AUROC for entropy-only and KL-only
- One latency measurement (ms overhead per inference)

### 2. Add one more model (1 day of compute)
Running on Pythia-1.4B or Llama-3-8B would strengthen the paper significantly.
The paper is technically submittable with Pythia-160m only, but reviewers will
ask about generalization. Even "same trend holds on Pythia-1.4B" is valuable.

### 3. Add one figure (half day)
A figure showing mean KL by layer for hallucinated vs. non-hallucinated samples
would provide mechanistic evidence. This is straightforward to generate from
the existing code.

Example figure to generate:
```python
# Plot: layer_kl[hallucinated] vs layer_kl[not_hallucinated]
import matplotlib.pyplot as plt
# y-axis: mean D_KL^l across samples
# x-axis: layer index l
# two lines: hallucinated vs. not
```

### 4. Fix one incomplete citation (1 hour)
`yu2024attention` is marked TODO — either find the right paper or remove this citation.
All other citations are verified.

### 5. Final proofread (2–3 hours)
Remove all \TODO{} and \RESULT{} markers before submission.
Replace with actual numbers from step 1.

---

## Paper Structure Summary

The paper (paper.tex) has the following complete sections:
1. Abstract (complete — fill in numbers)
2. Introduction (complete)
3. Background and Related Work (complete)
4. Method (complete — all math is there)
5. Experiments (structure complete — needs real numbers)
6. Discussion (complete)
7. Conclusion (complete)
8. References (references.bib — 18 verified citations)
9. Appendix (complete — implementation details)

**Paper length:** ~6–7 pages in the ACL/EMNLP format, or ~5 pages in NeurIPS/ICML format.
arXiv doesn't have length requirements.

---

## After arXiv: What Happens Next

1. **Papers.with.code** — submit your repo URL once the paper is live. 
   The site will index it and others can find/cite it.

2. **HuggingFace Hub** — upload the trained BiLSTM checkpoint as a model.
   This dramatically increases discoverability.

3. **OpenReview** — consider submitting to NeurIPS 2026 Workshops on:
   - "Reliable and Responsible Foundation Models" 
   - "ML Safety Workshop"
   These workshops accept shorter papers and give feedback without requiring formal venue acceptance.

4. **Cite it in all related READMEs** — the crosscloud, clinical, and 
   CIPHER (if applicable) READMEs all reference "entropy as operational primitive."
   Once this paper exists, those references become credible.

---

## What This Paper Does for the Portfolio

Before paper:
> "I use entropy as an operational primitive" — unverifiable README claim

After paper:
> "I use entropy as an operational primitive — see [AED arXiv:2026.XXXXX]" — citable claim

Every other repo that references entropy-based uncertainty (crosscloud-ml-orchestration, 
Multi-Source-Clinical, AI-Safety) gains a citable foundation. The entire portfolio 
narrative becomes grounded in verifiable research.

This is why the synthesis agent called it the 10x leverage move.

---

## Files in this directory

- `paper.tex` — Full LaTeX paper. Compile with `pdflatex paper.tex`
- `references.bib` — BibTeX references (18 entries, all verified)
- `PAPER_ROADMAP.md` — This file

To get started immediately: run the Colab notebook to get your real numbers,
fill in the TODO values in paper.tex, and you're 80% of the way to submission.
