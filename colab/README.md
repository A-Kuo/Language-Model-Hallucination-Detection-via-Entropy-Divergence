# Colab Notebooks

Run the computationally heavy parts of this project on a free Colab T4 GPU.

| Notebook | Opens in Colab | What it does | Runtime |
|----------|---------------|--------------|---------|
| `full_pipeline.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Natural-Hallucination-Analysis/blob/main/colab/full_pipeline.ipynb) | Full pipeline: Claude QA gen → Pythia answers → judge → features → classifier | ~15-30 min |

See also `gpu_benchmark.ipynb`, `ablation_study.ipynb`, and `quick_cpu_validation.ipynb` (indexed in [`../COLABS.md`](../COLABS.md)) — **these three currently import from `run_experiment.py`, a module that was removed when the old `v1/`/`v2/` split was consolidated into a single flat codebase at the repo root; they need to be ported to the current `pipeline.py`/`detector.py` API before they'll run again.**

## Before Running

Add your Anthropic API key as a Colab Secret:

1. Left sidebar → 🔑 **Secrets**
2. Click **+ Add new secret**
3. Name: `ANTHROPIC_API_KEY`
4. Value: `sk-ant-...`
5. Toggle **Notebook access** ON

The notebook reads it with `google.colab.userdata.get('ANTHROPIC_API_KEY')` — it never touches the filesystem.

## Outputs

`full_pipeline.ipynb` downloads results automatically at the end:
- `pipeline_results.json` — full metrics + feature importance
- `dataset.jsonl` — labeled dataset (commit to `data/` for reproducibility)
- `detector.pkl` — trained classifier
- `feature_distributions.png` — feature family separation

## Using Outputs to Update the Repo

After running, update the benchmark tables in:
- `README.md` — root-level comparison table and Current Status section
