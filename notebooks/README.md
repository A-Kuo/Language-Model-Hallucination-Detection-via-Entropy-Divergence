# Notebooks — Kaggle GPU Testing

Two kernels, two different purposes:

| Kernel dir | Notebook | Purpose |
|---|---|---|
| `notebooks/` | `pytorch_dl_testbed.ipynb` | Synthetic sandbox — a fake pseudo-LM, no repo-module imports. Fast, cheap way to test architecture/signal-design hypotheses before touching real code. |
| `notebooks/real_pipeline_benchmark/` | `real_pipeline_benchmark.ipynb` | Clones the repo and calls the **real** `pipeline.py::run_real_pipeline()` against real HaluEval data and a real HuggingFace model. Every number it produces is a genuine benchmark of the actual detectors, at a scale (bigger model, more samples) that isn't practical on a laptop CPU. |

Both can be run three ways: locally/Colab (synthetic notebook only — the real-pipeline one needs GPU+internet to be worth running), manually on Kaggle, or via the GitHub Actions Kaggle GPU workflow below.

## The Kaggle GPU loop

**This never runs automatically.** It only fires when you deliberately trigger it — from the repo's Actions tab (select "Kaggle GPU Run" → "Run workflow", then pick which kernel from the `kernel_dir` dropdown) or via `gh workflow run kaggle_runner.yml -f kernel_dir=notebooks/real_pipeline_benchmark`. There is no `push` trigger; GPU kernel runs cost quota and time, so they only happen on your direct command.

What happens when you trigger it:
1. GitHub Actions pushes the chosen kernel directory to Kaggle (using that directory's `kernel-metadata.json`) and starts it running on a GPU.
2. The workflow polls Kaggle every 30s until the kernel finishes (or times out after 85 minutes — the full-scale runs can take a while on GPU).
3. Once complete, it downloads the kernel's output files. The sandbox kernel's results land flat in `notebooks/results/` (unchanged, for backward compatibility); any other kernel's results land in `notebooks/results/<kernel_dir_name>/` — e.g. `notebooks/results/real_pipeline_benchmark/`.
4. It commits that results directory back into the repo.

Both notebooks detect `/kaggle/working/` and write there directly when running as a Kaggle kernel, so downloaded files land flat (no nested `notebooks/outputs/`-style subtree).

## Prerequisites (one-time setup)

- A Kaggle account with **phone verification completed** — Kaggle requires this before it will run GPU-enabled kernels, even via the API.
- `KAGGLE_USERNAME` and `KAGGLE_KEY` set as GitHub Actions repo secrets (Settings → Secrets and variables → Actions). These are the exact environment variable names the official `kaggle` CLI reads for authentication — don't rename them.
- Each kernel directory's `kernel-metadata.json` has its `id` field set to `<your-kaggle-username>/<kernel-slug>` — already done for both kernels here (`pytorch-hallucination-testing` and `real-pipeline-benchmark`).
- **Before relying on CI**, run `kaggle kernels push -p <kernel_dir>/` locally once per kernel (with `kaggle` installed and your credentials configured) to confirm it actually creates and runs on Kaggle's side — this catches auth/schema issues interactively instead of burning a CI run on them.

## Reading results

- `notebooks/results/*.json` — the synthetic sandbox's three experiment outputs (`expA_architecture.json`, `expB_gamma_sweep.json`, `expC_confab_stress.json`), GPU-generated instead of local-CPU-generated.
- `notebooks/results/real_pipeline_benchmark/real_pipeline_benchmark.json` — the real pipeline's structured summary: CV AUROC+CI per detector, held-out metrics, ablation deltas, feature importances. This is the file to read when deciding whether to change a detector's hyperparameters or feature set based on real (not synthetic) evidence.

Read them directly — no additional tooling needed.
