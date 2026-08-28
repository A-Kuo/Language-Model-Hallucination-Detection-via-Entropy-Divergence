# Streamlit demo

A small local model (Pythia-160m) answers questions — some correctly, some
hallucinated — while `CalibratedEntropyDetector` scores every answer live,
using the same feature pipeline documented in the main [README.md](../README.md)
§4. Two modes: **live generation** (you ask, the model answers, the detector
scores it in real time) and a **paired example browser** (real HaluEval
question pairs, correct vs. hallucinated, scored side by side).

## Setup (local)

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

`demo/detector.pkl` is already committed (see below), so this runs
immediately — no build step required. If you want to rebuild it (e.g. after
changing `--model` or `--num_samples`, or to pick up new HaluEval data):

```bash
python demo/build_detector.py
```

This downloads paired HaluEval samples, extracts real attention/entropy
features with Pythia-160m, and fits a `CalibratedEntropyDetector(score_index="auto")`
on them — the same detector class documented in README.md §4.3, not a toy
stand-in. Takes a few minutes on CPU.

## Deploying (Streamlit Community Cloud)

Point the app at `demo/app.py` as the main module and nothing else — no
custom requirements-file path needed. Two things make this actually work,
both hit as real failures while building this demo, not hypotheticals:

1. **`demo/requirements.txt` must be self-contained.** Streamlit Cloud installs
   from exactly *one* requirements file — its own deploy log confirms this
   ("Used: uv with .../demo/requirements.txt", nothing from root
   `requirements.txt` also ran) — so this file lists everything the demo needs,
   including `numpy`/`scipy` (which the root `requirements.txt` also lists, for
   the core package; the two files are not merged, so duplication here is
   required, not an oversight).
2. **`runtime.txt` (repo root) pins the Python version to 3.11.** Left
   unpinned, Streamlit Cloud picked the newest available Python (3.14 when
   this was hit), which has no prebuilt wheels yet for compiled packages like
   `scipy`/`torch` — pip/uv then falls back to building them from source,
   which can hang for a very long time on a resource-constrained free-tier
   container. That's what "the deploy loads forever" turned out to be, once
   the requirements-file issue above was ruled out.

## What it shows

- **RELIABLE / UNCERTAIN / UNRELIABLE** routing (README.md §5.4's 3-way
  decision layer), not just a raw probability — the headline UI element.
- The raw `P(hallucination)` and the routing thresholds it was compared
  against, so the verdict isn't a black box.
- Top contributing features for a live-generated answer, so "why did it flag
  this" has a concrete answer.

## Honest framing

Pythia-160m is a small **base** model — no instruction tuning, no retrieval
grounding. It hallucinates readily, especially on the trap questions in the
preset list (a nonexistent President, a fictional Einstein quote). That's
deliberate: the demo exists to show the detector catching a hallucination,
not to claim the underlying model is reliable. See README.md §8.1 for the
project's own limitations, including the confident-confabulation ceiling
that no feature in this repo (or this demo) can see past.

`demo/detector.pkl` (16KB) **is committed**, unlike most build artifacts in
this repo — Streamlit Community Cloud has no pre-deploy build-step hook, so
without a committed detector the deployed app would have nothing to load and
would `st.stop()` immediately. It's small and reproducible from
`build_detector.py`, so the usual "don't commit generated binaries" concern
doesn't really apply here; re-run `build_detector.py` and commit the new file
whenever you want the deployed demo to reflect a different model or dataset.
