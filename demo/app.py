"""
Streamlit demo — a small local language model answers questions (and
sometimes hallucinates), while a CalibratedEntropyDetector scores every
answer live using the same 24D attention + token-entropy feature pipeline
documented in README.md §4. See demo/README.md for setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibrated_entropy_detector import CalibratedEntropyDetector
from entropy_baselines import EntropyFeatureExtractor
from feature_engineer import AttentionFeatureEngineer
from pipeline import build_prompt_and_text, extract_attention_from_model, extract_logits_from_model

MODEL_NAME = "EleutherAI/pythia-160m"
DETECTOR_PATH = Path(__file__).parent / "detector.pkl"
SAMPLE_PAIRS_PATH = Path(__file__).parent / "sample_pairs.json"

PRESET_PROMPTS = [
    "What is the capital of France?",
    "Who wrote the novel '1984'?",
    "What year did the first person land on the Moon?",
    "What is the boiling point of water at sea level in Celsius?",
    "Who was the fourteenth President of a country that has never had a President?",
    "What did Albert Einstein say about the Great Wall of Mars?",
]

st.set_page_config(page_title="Hallucination Detector Demo", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, output_attentions=True, torch_dtype=torch.float32
    ).eval()
    return model, tokenizer


@st.cache_resource(show_spinner="Loading detector...")
def load_detector():
    if not DETECTOR_PATH.exists():
        return None
    return CalibratedEntropyDetector.load(str(DETECTOR_PATH))


@st.cache_data(show_spinner="Loading HaluEval pairs...")
def load_paired_examples():
    """
    Loads a small, pre-baked set of real HaluEval question/correct/hallucinated
    triples from demo/sample_pairs.json (generated once via
    DataGenerator.from_halueval() and committed — see that script's docstring
    below). This deliberately avoids calling from_halueval() live: it needs
    `datasets` plus a full download+parse of the HaluEval dataset from
    HuggingFace Hub, which happened unconditionally on every cold start
    (Streamlit re-executes the whole script on each run, including both tab
    bodies, regardless of which tab is visible) and was a real contributor to
    the deployed demo taking a very long time to show anything at all.
    """
    with open(SAMPLE_PAIRS_PATH, encoding="utf-8") as f:
        pairs = json.load(f)
    return [(p["question"], p["correct"], p["hallucinated"]) for p in pairs]


# To regenerate demo/sample_pairs.json:
#   python -c "
#   import json
#   from data_generator import DataGenerator
#   samples = DataGenerator.from_halueval(num_samples=60, seed=123)
#   by_q = {}
#   for s in samples: by_q.setdefault(s.question, {})[s.label] = s.model_answer
#   pairs = [{'question': q, 'correct': d['correct'], 'hallucinated': d['hallucinated']}
#            for q, d in by_q.items() if 'correct' in d and 'hallucinated' in d]
#   json.dump(pairs, open('demo/sample_pairs.json', 'w'), indent=2, ensure_ascii=False)
#   "


def score_text(model, tokenizer, detector, question: str, answer: str):
    engineer = AttentionFeatureEngineer(context_length=32)
    entropy_extractor = EntropyFeatureExtractor()

    prompt, text = build_prompt_and_text(tokenizer, question, answer)
    attentions, context_len = extract_attention_from_model(text, model, tokenizer, "cpu", prompt=prompt)
    attn_feats = engineer.extract(attentions, context_len)
    logits, token_ids, answer_start = extract_logits_from_model(text, model, tokenizer, "cpu", prompt=prompt)
    entropy_feats = entropy_extractor.extract(logits, answer_start=answer_start, target_ids=token_ids)

    feats = np.concatenate([attn_feats, entropy_feats])
    decision = detector.route_one(feats)
    return decision, feats


def render_decision(decision):
    msg = f"**{decision.label}** — P(hallucination) = {decision.p_hallucination:.1%} → {decision.action}"
    if decision.label == "RELIABLE":
        st.success(msg)
    elif decision.label == "UNCERTAIN":
        st.warning(msg)
    else:
        st.error(msg)
    st.progress(min(max(decision.p_hallucination, 0.0), 1.0))
    st.caption(
        f"thresholds fit on the calibration set — RELIABLE: p < {decision.threshold_reliable:.2f}, "
        f"UNRELIABLE: p ≥ {decision.threshold_unreliable:.2f} (see README.md §5.4)"
    )


st.title("🔍 Hallucination Detector — Live Demo")
st.write(
    "A small local language model (**Pythia-160m**, no instruction tuning, no retrieval) answers a "
    "question. A `CalibratedEntropyDetector` — fit on real HaluEval attention + token-entropy "
    "features, the same pipeline documented in this repo's README — scores the answer for "
    "hallucination risk as soon as it's generated."
)

model, tokenizer = load_model()
detector = load_detector()

if detector is None:
    st.error(
        f"No trained detector found at `{DETECTOR_PATH}`. Run `python demo/build_detector.py` "
        "first (see demo/README.md)."
    )
    st.stop()

tab_live, tab_pairs = st.tabs(["🎲 Live generation", "📋 Paired examples"])

with tab_live:
    st.subheader("Ask the model something")
    choice = st.selectbox(
        "Pick a preset question (or write your own below)",
        ["(write my own)"] + PRESET_PROMPTS,
    )
    question = st.text_input("Question", value="" if choice == "(write my own)" else choice)
    col_a, col_b = st.columns(2)
    max_new_tokens = col_a.slider("Max answer length (tokens)", 8, 80, 40)
    temperature = col_b.slider("Sampling temperature", 0.1, 1.5, 0.8)

    if st.button("Generate & Score", type="primary", disabled=not question.strip()):
        with st.spinner("Generating..."):
            prompt_only, _ = build_prompt_and_text(tokenizer, question, "")
            inputs = tokenizer(prompt_only, return_tensors="pt")
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

        st.markdown(f"**Model's answer:** {answer or '_(empty)_'}")
        if not answer:
            st.warning("Model produced an empty answer — try again or increase max tokens.")
        else:
            with st.spinner("Scoring..."):
                decision, feats = score_text(model, tokenizer, detector, question, answer)
            render_decision(decision)
            with st.expander("Top contributing features"):
                names = detector.feature_names or [f"feature_{i}" for i in range(len(feats))]
                top = sorted(zip(names, feats), key=lambda kv: -abs(kv[1]))[:8]
                for name, val in top:
                    st.write(f"`{name}` = {val:.3f}")

    st.caption(
        "Pythia-160m is a small **base** model — no instruction tuning, no retrieval grounding. "
        "It will confidently make things up, especially on the trap questions above (a nonexistent "
        "President, a fictional Einstein quote). That's the point: this demo shows the detector "
        "catching it, not a claim that the underlying model itself is reliable — see README.md §8.1."
    )

with tab_pairs:
    st.subheader("Real HaluEval pairs: same question, one correct answer, one hallucinated")
    pairs = load_paired_examples()
    idx = st.number_input("Pair index", min_value=0, max_value=len(pairs) - 1, value=0, step=1)
    question, correct_answer, halluc_answer = pairs[idx]
    st.markdown(f"**Question:** {question}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Correct answer**")
        st.write(correct_answer)
        with st.spinner("Scoring..."):
            decision, _ = score_text(model, tokenizer, detector, question, correct_answer)
        render_decision(decision)
    with col2:
        st.markdown("**Hallucinated answer**")
        st.write(halluc_answer)
        with st.spinner("Scoring..."):
            decision, _ = score_text(model, tokenizer, detector, question, halluc_answer)
        render_decision(decision)

    st.caption(
        "These pairs are ground truth from the HaluEval benchmark (Li et al., 2023) — the detector "
        "scores text it did not generate itself here, the same way it would score any other model's "
        "output."
    )
