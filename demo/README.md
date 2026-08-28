# Streamlit demo

A small local model (Pythia-160m) answers questions — some correctly, some
hallucinated — while `CalibratedEntropyDetector` scores every answer live,
using the same feature pipeline documented in the main [README.md](../README.md)
§4. Two modes: **live generation** (you ask, the model answers, the detector
scores it in real time) and a **paired example browser** (real HaluEval
question pairs, correct vs. hallucinated, scored side by side).

## Setup

```bash
pip install -r requirements-demo.txt
python demo/build_detector.py          # one-time: fits and saves demo/detector.pkl
streamlit run demo/app.py
```

`build_detector.py` downloads 400 paired HaluEval samples, extracts real
attention/entropy features with Pythia-160m, and fits a
`CalibratedEntropyDetector(score_index="auto")` on them — the same detector
class documented in README.md §4.3, not a toy stand-in. This takes a few
minutes on CPU. Re-run it any time you want to rebuild `demo/detector.pkl`
(e.g. after changing `--model` or `--num_samples`).

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

`demo/detector.pkl` is a build artifact (git-ignored) — it is not committed,
since it's reproducible from `build_detector.py` and would otherwise bloat
the repo with a binary that goes stale the moment the model or feature set
changes.
