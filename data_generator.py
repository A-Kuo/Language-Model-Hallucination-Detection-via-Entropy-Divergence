"""
Self-Data Generator — Synthetic Hallucination Dataset Creation
================================================================

Creates labeled datasets for training hallucination detectors without
any human annotation, following Lookback Lens (Chuang et al., EMNLP 2024):

    1. Generate factual QA pairs via Anthropic API (ground truth known)
    2. Prompt a local open-weight model to answer (may hallucinate)
    3. Use Claude as judge to label: correct / hallucinated / ambiguous
    4. Pair each (question, answer, label) with extracted attention features

This produces N labeled samples per call, scaling linearly with API budget.

Usage:
    generator = DataGenerator(anthropic_api_key="sk-...")
    dataset = generator.generate(num_samples=500, domains=["science", "history"])
    generator.save(dataset, "data/train.jsonl")

References:
    - Chuang et al. (2024) "Lookback Lens" — used an LLM-as-judge approach
    - Qi et al. (2026) "Frequency-Aware Attention" — QA-based evaluation
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class QAPair:
    """A question with its ground-truth answer."""
    question: str
    ground_truth: str
    domain: str
    difficulty: str  # "easy", "medium", "hard"
    source: str = "synthetic"  # where this QA came from


@dataclass
class LabeledSample:
    """A single labeled datapoint for hallucination detection."""
    question: str
    ground_truth: str
    model_answer: str
    label: Literal["correct", "hallucinated", "ambiguous"]
    domain: str
    difficulty: str
    judge_reasoning: str = ""
    model_name: str = ""
    sample_id: str = ""
    features: Optional[Dict[str, Any]] = None  # filled by feature engineer

    def __post_init__(self):
        if not self.sample_id:
            raw = f"{self.question}|{self.model_answer}|{self.model_name}"
            self.sample_id = hashlib.md5(raw.encode()).hexdigest()[:12]


# =========================================================================
# QA generation prompts
# =========================================================================

QA_GENERATION_PROMPT = """Generate {n} factual question-answer pairs about {domain}.

Requirements:
- Questions should have clear, unambiguous factual answers
- Answers should be 1-3 sentences, specific and verifiable
- Mix difficulty: some easy (common knowledge), some hard (specialized)
- Each answer MUST be factually correct

Return ONLY a JSON array, no other text:
[
  {{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}},
  ...
]"""

JUDGE_PROMPT = """You are a factual accuracy judge. Given a question, the correct answer, and a model's answer, determine if the model's answer is correct or hallucinated.

Question: {question}
Correct answer: {ground_truth}
Model's answer: {model_answer}

Respond ONLY with a JSON object:
{{"label": "correct|hallucinated|ambiguous", "reasoning": "one sentence explanation"}}

Rules:
- "correct": model answer is factually consistent with the correct answer (doesn't need to be word-for-word)
- "hallucinated": model answer contains factual errors, fabricated details, or contradicts the correct answer
- "ambiguous": cannot determine correctness (e.g., model refused to answer, or answer is too vague)"""


# =========================================================================
# DataGenerator
# =========================================================================

class DataGenerator:
    """
    Self-data creation pipeline for hallucination detection.

    Architecture (inspired by Lookback Lens §3.1):
        Claude (API) → QA pairs → Local model answers → Claude judges → Labels
    """

    # Domains that produce clear factual QA pairs
    DEFAULT_DOMAINS = [
        "science",
        "world_history",
        "geography",
        "mathematics",
        "technology",
        "biology",
        "economics",
        "literature",
    ]

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        judge_model: str = "claude-sonnet-4-20250514",
        qa_batch_size: int = 10,
        temperature: float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        anthropic_api_key : str
            Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        judge_model : str
            Model for QA generation and judging.
        qa_batch_size : int
            Number of QA pairs per API call.
        temperature : float
            Sampling temperature for QA generation (higher = more diverse).
        """
        self.judge_model = judge_model
        self.qa_batch_size = qa_batch_size
        self.temperature = temperature
        self._client = None
        self._api_key = anthropic_api_key

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # -----------------------------------------------------------------
    # Step 1: Generate QA pairs
    # -----------------------------------------------------------------

    def generate_qa_pairs(
        self,
        num_pairs: int = 50,
        domains: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """
        Generate factual QA pairs using Claude.

        Distributes evenly across domains. Each API call produces
        `qa_batch_size` pairs.
        """
        domains = domains or self.DEFAULT_DOMAINS
        pairs: List[QAPair] = []
        pairs_per_domain = max(1, num_pairs // len(domains))

        for domain in domains:
            remaining = pairs_per_domain
            while remaining > 0:
                batch = min(remaining, self.qa_batch_size)
                prompt = QA_GENERATION_PROMPT.format(n=batch, domain=domain)

                try:
                    response = self.client.messages.create(
                        model=self.judge_model,
                        max_tokens=2000,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    text = response.content[0].text.strip()
                    # Parse JSON array from response
                    raw = json.loads(text)
                    for item in raw:
                        pairs.append(QAPair(
                            question=item["question"],
                            ground_truth=item["answer"],
                            domain=domain,
                            difficulty=item.get("difficulty", "medium"),
                        ))

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"  Warning: failed to parse QA batch for {domain}: {e}")
                except Exception as e:
                    print(f"  Warning: API error for {domain}: {e}")

                remaining -= batch
                time.sleep(0.5)  # rate limiting

            if len(pairs) >= num_pairs:
                break

        return pairs[:num_pairs]

    # -----------------------------------------------------------------
    # Step 2: Get model answers (local open-weight model)
    # -----------------------------------------------------------------

    @staticmethod
    def get_model_answers(
        qa_pairs: List[QAPair],
        model_name: str = "EleutherAI/pythia-160m",
        max_new_tokens: int = 100,
        device: str = "cpu",
    ) -> List[str]:
        """
        Run a local HuggingFace model on each question.

        The local model is deliberately imperfect — its errors become
        the hallucination signal we want to detect.

        For larger models (Llama, Mistral, Phi), use device="cuda".
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model.to(device).eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        answers = []
        for i, qa in enumerate(qa_pairs):
            prompt = f"Question: {qa.question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the generated tokens (not the prompt)
            answer_ids = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            answers.append(answer)

            if (i + 1) % 25 == 0:
                print(f"    Generated {i+1}/{len(qa_pairs)} answers")

        return answers

    # -----------------------------------------------------------------
    # Step 3: Judge answers with Claude
    # -----------------------------------------------------------------

    def judge_answers(
        self,
        qa_pairs: List[QAPair],
        model_answers: List[str],
    ) -> List[LabeledSample]:
        """
        Use Claude as LLM-as-judge to label each (question, answer) pair.

        Following Lookback Lens's LLM-as-judge approach, which confirmed
        a 97% consistency rate with human annotation.
        """
        samples: List[LabeledSample] = []

        for i, (qa, answer) in enumerate(zip(qa_pairs, model_answers)):
            prompt = JUDGE_PROMPT.format(
                question=qa.question,
                ground_truth=qa.ground_truth,
                model_answer=answer,
            )

            try:
                response = self.client.messages.create(
                    model=self.judge_model,
                    max_tokens=200,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = response.content[0].text.strip()
                result = json.loads(text)

                samples.append(LabeledSample(
                    question=qa.question,
                    ground_truth=qa.ground_truth,
                    model_answer=answer,
                    label=result["label"],
                    domain=qa.domain,
                    difficulty=qa.difficulty,
                    judge_reasoning=result.get("reasoning", ""),
                ))

            except Exception as e:
                # On parse failure, label as ambiguous
                samples.append(LabeledSample(
                    question=qa.question,
                    ground_truth=qa.ground_truth,
                    model_answer=answer,
                    label="ambiguous",
                    domain=qa.domain,
                    difficulty=qa.difficulty,
                    judge_reasoning=f"judge_error: {e}",
                ))

            if (i + 1) % 25 == 0:
                print(f"    Judged {i+1}/{len(qa_pairs)}")
                time.sleep(0.3)

        return samples

    # -----------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------

    def generate(
        self,
        num_samples: int = 100,
        domains: Optional[List[str]] = None,
        local_model: str = "EleutherAI/pythia-160m",
        device: str = "cpu",
    ) -> List[LabeledSample]:
        """
        End-to-end data generation pipeline.

        Returns labeled samples ready for feature extraction.
        """
        print(f"[1/3] Generating {num_samples} QA pairs...")
        qa_pairs = self.generate_qa_pairs(num_samples, domains)
        print(f"  Generated {len(qa_pairs)} QA pairs across {len(set(q.domain for q in qa_pairs))} domains")

        print(f"\n[2/3] Getting answers from {local_model}...")
        answers = self.get_model_answers(qa_pairs, local_model, device=device)

        print(f"\n[3/3] Judging answers with {self.judge_model}...")
        samples = self.judge_answers(qa_pairs, answers)

        # Tag with model name
        for s in samples:
            s.model_name = local_model

        # Stats
        labels = [s.label for s in samples]
        print(f"\n  Dataset: {len(samples)} samples")
        print(f"  Correct: {labels.count('correct')} ({labels.count('correct')/len(labels)*100:.0f}%)")
        print(f"  Hallucinated: {labels.count('hallucinated')} ({labels.count('hallucinated')/len(labels)*100:.0f}%)")
        print(f"  Ambiguous: {labels.count('ambiguous')} ({labels.count('ambiguous')/len(labels)*100:.0f}%)")

        return samples

    # -----------------------------------------------------------------
    # Offline mode: generate from static questions
    # -----------------------------------------------------------------

    @staticmethod
    def from_trivia_qa(
        path: str = "data/triviaqa_sample.jsonl",
        num_samples: int = 500,
    ) -> List[QAPair]:
        """
        Load QA pairs from TriviaQA or similar dataset (offline, no API).

        Expected format per line: {"question": "...", "answer": "..."}
        """
        pairs = []
        with open(path) as f:
            for line in f:
                if len(pairs) >= num_samples:
                    break
                item = json.loads(line)
                pairs.append(QAPair(
                    question=item["question"],
                    ground_truth=item["answer"],
                    domain="trivia",
                    difficulty="medium",
                    source="triviaqa",
                ))
        return pairs

    @staticmethod
    def from_halueval(
        num_samples: int = 500,
        seed: int = 42,
    ) -> List["LabeledSample"]:
        """
        Load pre-labeled hallucination samples from HaluEval (no API required).

        HaluEval (Peng et al., 2023) provides QA pairs where each question has
        both a correct answer and a hallucinated answer, labeled by GPT-3.5.
        This gives us balanced, research-validated ground truth with zero API cost.

        Requires: pip install datasets

        Parameters
        ----------
        num_samples : int
            Total samples to return (split evenly: half correct, half hallucinated).
        seed : int
            Random seed for shuffling.

        Returns
        -------
        List[LabeledSample]  ready for feature extraction, no generation step needed.

        Reference
        ---------
        Peng et al. (2023). "HaluEval: A Large-Scale Hallucination Evaluation
        Benchmark for Large Language Models." EMNLP 2023.
        https://github.com/RUCAIBox/HaluEval
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace `datasets` required: pip install datasets"
            )

        import random
        rng = random.Random(seed)

        print("  Downloading HaluEval QA split from HuggingFace Hub...")
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        print(f"  Loaded {len(ds)} rows from HaluEval.")

        # Each row has one correct and one hallucinated answer — interleave both
        rows = list(ds)
        rng.shuffle(rows)

        half = num_samples // 2
        samples: List[LabeledSample] = []

        for row in rows[:half]:
            samples.append(LabeledSample(
                question=row["question"],
                ground_truth=row["right_answer"],
                model_answer=row["hallucinated_answer"],
                label="hallucinated",
                domain="knowledge",
                difficulty="medium",
                judge_reasoning="HaluEval ground truth",
                model_name="halueval_gpt3.5",
            ))

        for row in rows[:half]:
            samples.append(LabeledSample(
                question=row["question"],
                ground_truth=row["right_answer"],
                model_answer=row["right_answer"],
                label="correct",
                domain="knowledge",
                difficulty="medium",
                judge_reasoning="HaluEval ground truth",
                model_name="halueval_gpt3.5",
            ))

        rng.shuffle(samples)
        print(f"  HaluEval: {len(samples)} samples "
              f"({sum(1 for s in samples if s.label == 'hallucinated')} hallucinated, "
              f"{sum(1 for s in samples if s.label == 'correct')} correct)")
        return samples

    # -----------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------

    @staticmethod
    def save(samples: List[LabeledSample], path: str) -> None:
        """Save labeled samples to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                d = asdict(s)
                # Features may contain numpy arrays
                if d.get("features"):
                    d["features"] = {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in d["features"].items()
                    }
                f.write(json.dumps(d) + "\n")
        print(f"  Saved {len(samples)} samples to {path}")

    @staticmethod
    def load(path: str) -> List[LabeledSample]:
        """Load labeled samples from JSONL."""
        samples = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                features = d.pop("features", None)
                sample = LabeledSample(**d)
                sample.features = features
                samples.append(sample)
        return samples


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("DataGenerator — Structure Validation")
    print("=" * 50)

    # Test data structures
    qa = QAPair(
        question="What is the capital of France?",
        ground_truth="Paris",
        domain="geography",
        difficulty="easy",
    )
    print(f"  QAPair: {qa.question} → {qa.ground_truth}")

    sample = LabeledSample(
        question=qa.question,
        ground_truth=qa.ground_truth,
        model_answer="The capital of France is Paris.",
        label="correct",
        domain="geography",
        difficulty="easy",
        model_name="EleutherAI/pythia-160m",
    )
    assert sample.sample_id  # auto-generated
    assert len(sample.sample_id) == 12
    print(f"  LabeledSample: id={sample.sample_id}, label={sample.label}")

    # Test save/load roundtrip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        tmppath = f.name

    DataGenerator.save([sample], tmppath)
    loaded = DataGenerator.load(tmppath)
    assert len(loaded) == 1
    assert loaded[0].label == "correct"
    assert loaded[0].sample_id == sample.sample_id
    print(f"  Save/load roundtrip ✅")

    Path(tmppath).unlink()

    print(f"\n  All structure checks pass ✅")
    print(f"\n  To run full pipeline: set ANTHROPIC_API_KEY and call:")
    print(f"    gen = DataGenerator()")
    print(f"    dataset = gen.generate(num_samples=100)")
