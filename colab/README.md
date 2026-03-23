# Colab Notebooks

Run the computationally heavy parts of this project on a free Colab T4 GPU.

| Notebook | Opens in Colab | What it does | Runtime |
|----------|---------------|--------------|---------|
| `v1_benchmark.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Natural-Hallucination-Analysis/blob/main/colab/v1_benchmark.ipynb) | Real GPU benchmark of attention analyzer + hypothesis test | ~5 min |
| `v2_full_pipeline.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Kuo/Natural-Hallucination-Analysis/blob/main/colab/v2_full_pipeline.ipynb) | Full pipeline: Claude QA gen → Pythia answers → judge → features → classifier | ~15-30 min |

## Before Running v2

Add your Anthropic API key as a Colab Secret:

1. Left sidebar → 🔑 **Secrets**
2. Click **+ Add new secret**
3. Name: `ANTHROPIC_API_KEY`
4. Value: `sk-ant-...`
5. Toggle **Notebook access** ON

The notebook reads it with `google.colab.userdata.get('ANTHROPIC_API_KEY')` — it never touches the filesystem.

## Outputs

Each notebook downloads results automatically at the end:

**v1:**
- `v1_benchmark_results.json` — real AUROC, latency, throughput numbers
- `v1_attention_signals.png` — entropy + KL distributions
- `v1_entropy_heatmap.png` — per-layer, per-head entropy heatmap

**v2:**
- `v2_pipeline_results.json` — full metrics + feature importance
- `v2_dataset.jsonl` — labeled dataset (commit to `data/` for reproducibility)
- `v2_detector.pkl` — trained classifier
- `v2_feature_distributions.png` — feature family separation

## Using Outputs to Update the Repo

After running, update the benchmarks tables in:
- `README.md` — root-level comparison table
- `v1/README.md` — Key Results table
- `v2/README.md` — add a real metrics table
