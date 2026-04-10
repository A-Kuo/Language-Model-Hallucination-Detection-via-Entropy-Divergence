# Colab Notebooks — AED Hallucination Detection

> Click any badge below to launch the notebook in Google Colab (free T4 GPU tier).

---

## Primary Notebooks

### 🎯 GPU Benchmark — Full Paper Results
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/gpu_benchmark.ipynb)

**Purpose:** Generate the actual numbers for the arXiv paper  
**Runtime:** ~15 minutes on T4 GPU  
**Output:**
- `results/benchmark_results.json` — AUROC, FPR@90%TPR, latency
- `results/figure_layer_entropy.png` — Figure 1 for paper

**What it does:**
1. Loads Pythia-160m (12 layers, 12 heads, 160M params)
2. Loads HaluEval QA benchmark (500 samples, balanced)
3. Extracts per-head Shannon entropy + cross-layer KL divergence
4. Trains BiLSTM classifier on per-layer feature sequences
5. Reports AUROC, FPR@90%TPR, latency
6. **Auto-commits results to repo** (with GH_TOKEN secret)

**Before running:**
- Runtime → Change runtime type → T4 GPU
- Secrets (🔑) → Add `GH_TOKEN` with `repo` scope for auto-commit

**Paper integration:**
```
Cell 5 outputs: \RESULT{0.962} ← AED BiLSTM AUROC
                \RESULT{0.843} ← LogReg baseline AUROC
                47.2 ms        ← Latency per sample
```

---

### 📊 Ablation Study — Which Signal Matters?
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/ablation_study.ipynb)

**Purpose:** Fill Table 2 in the paper (ablation study)  
**Runtime:** ~8 minutes on T4 GPU  
**Output:** `results/ablation_results.json`

**What it tests:**
- Entropy-only (no KL divergence)
- KL-only (no entropy)
- Both combined (full AED)

**Paper table:**
```latex
\midrule
Entropy only    & \RESULT{?} & \RESULT{?} \\
KL only         & \RESULT{?} & \RESULT{?} \\
Both (full AED) & \RESULT{0.962} & -- \\
```

---

### 🔬 Quick Validation — CPU Mode
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/blob/main/colab/quick_cpu_validation.ipynb)

**Purpose:** Validate the pipeline works without GPU (for CI/testing)  
**Runtime:** ~3 minutes on CPU  
**Model:** GPT-2 (124M params)  
**Dataset:** 50 synthetic factual pairs  

**Note:** Results will be worse (AUROC ~0.5-0.7) because GPT-2 is smaller and synthetic data is limited. This is for pipeline validation only.

---

## How to Use

### For Paper Results
1. Click the **GPU Benchmark** badge above
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (Ctrl+F9)
4. Results auto-commit to repo if GH_TOKEN is set, or download manually

### For Development/Testing
1. Click the **Quick Validation** badge
2. Run on CPU (no GPU needed)
3. Verify the pipeline executes without errors

### To Add GH_TOKEN for Auto-Commit
1. Open any notebook
2. Click 🔑 Secrets tab on left sidebar
3. Add `GH_TOKEN` value:
   - Go to https://github.com/settings/tokens
   - Generate new token (classic)
   - Check `repo` scope
   - Copy token → paste into Colab Secrets

---

## Results Location

After running, results are stored in the repo at:
```
results/
├── benchmark_results.json          ← Main numbers
├── ablation_results.json           ← Ablation table
├── figure_layer_entropy.png        ← Paper Figure 1
└── figure_kl_divergence.png        ← Paper Figure 2
```

Download link: https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/tree/main/results

---

## Agent Integration

For autonomous agents continuing this work:

```python
# Download latest results via GitHub API
import requests
url = "https://raw.githubusercontent.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence/main/results/benchmark_results.json"
results = requests.get(url).json()

print(f"Latest AUROC: {results['aed_auroc']}")
print(f"Latest FPR@90%TPR: {results['aed_fpr90']}")
```

Or clone and read locally:
```bash
git clone https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence.git
cd Language-Model-Hallucination-Detection-via-Entropy-Divergence
cat results/benchmark_results.json
```

---

*Last updated: April 2026*
