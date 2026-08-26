# Notebooks — Kaggle GPU Testing

`pytorch_dl_testbed.ipynb` is a self-contained PyTorch experiment notebook (synthetic pseudo-LM, no repo-module imports — see its own intro cell for the research questions it tests). It can be run three ways:

1. **Locally / Colab** — just run the cells. Artifacts land in `notebooks/outputs/` (untracked scratch dir, not committed).
2. **Manually on Kaggle** — upload it as a Kaggle notebook yourself and run it interactively.
3. **Via the GitHub Actions Kaggle GPU workflow** (`.github/workflows/kaggle_runner.yml`) — see below.

## The Kaggle GPU loop

**This never runs automatically.** It only fires when you deliberately trigger it — from the repo's Actions tab (select "Kaggle GPU Run" → "Run workflow") or via `gh workflow run kaggle_runner.yml`. There is no `push` trigger; GPU kernel runs cost quota and time, so they only happen on your direct command.

What happens when you trigger it:
1. GitHub Actions pushes `pytorch_dl_testbed.ipynb` to Kaggle as a kernel (using `notebooks/kernel-metadata.json` for configuration) and starts it running on a GPU.
2. The workflow polls Kaggle every 30s until the kernel finishes (or times out after 30 minutes).
3. Once complete, it downloads the kernel's output files into `notebooks/results/`.
4. It commits `notebooks/results/` back into the repo.

Because the notebook detects `/kaggle/working/` and writes there directly when running as a Kaggle kernel, the downloaded files land flat: `notebooks/results/expA_architecture.json`, `expB_gamma_sweep.json`, `expC_confab_stress.json`, plus their corresponding `.png` figures — no nested subdirectory.

## Prerequisites (one-time setup)

- A Kaggle account with **phone verification completed** — Kaggle requires this before it will run GPU-enabled kernels, even via the API.
- `KAGGLE_USERNAME` and `KAGGLE_KEY` set as GitHub Actions repo secrets (Settings → Secrets and variables → Actions). These are the exact environment variable names the official `kaggle` CLI reads for authentication — don't rename them.
- `notebooks/kernel-metadata.json`'s `id` field set to `<your-kaggle-username>/pytorch-dl-testbed`.
- **Before relying on CI**, run `kaggle kernels push -p notebooks/` locally once (with `kaggle` installed and your credentials configured) to confirm the kernel actually creates and runs on Kaggle's side — this catches auth/schema issues interactively instead of burning a CI run on them.

## Reading results

Once a run has landed, `notebooks/results/*.json` holds the same three experiment outputs the notebook produces locally, just GPU-generated. Read them directly — no additional tooling needed.
