"""
Multi-provider LLM testing framework for hallucination detection.

Abstracts different LLM API providers (Anthropic, OpenAI, HuggingFace) and runs
hallucination detection benchmarks across models to validate detector generalization.

Usage:
    python api_testing.py --providers anthropic openai huggingface --dataset halueval --samples 100
"""

import os
import json
import time
import argparse
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
try:
    from data_generator import DataGenerator, LabeledSample
    from feature_engineer import AttentionFeatureEngineer
    from detector import HallucinationDetector, DetectorMetrics
except ImportError:
    # For cases when run from parent directory
    from v2.data_generator import DataGenerator, LabeledSample
    from v2.feature_engineer import AttentionFeatureEngineer
    from v2.detector import HallucinationDetector, DetectorMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class BenchmarkResult:
    """Results from multi-provider benchmark run."""
    provider: str
    model_name: str
    num_samples: int
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_latency_ms: float
    ci_lower: float
    ci_upper: float
    cost_usd: float
    timestamp: str
    notes: str = ""


class LLMProvider(ABC):
    """Abstract base for LLM API providers."""

    @abstractmethod
    def generate_answer(self, question: str, max_tokens: int = 100) -> str:
        """Generate answer to question."""
        pass

    @abstractmethod
    def batch_generate(self, questions: List[str], max_tokens: int = 100) -> List[str]:
        """Generate answers for batch of questions."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return provider name."""
        pass

    @abstractmethod
    def get_cost_usd(self) -> float:
        """Return total API cost incurred."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.total_cost = 0.0
        self.calls_made = 0

    def generate_answer(self, question: str, max_tokens: int = 100) -> str:
        """Generate answer using Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": question}]
            )
            self.calls_made += 1
            # Rough cost estimate: $3 per M input tokens, $15 per M output tokens
            self.total_cost += (message.usage.input_tokens / 1e6 * 3) + (message.usage.output_tokens / 1e6 * 15)
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ""

    def batch_generate(self, questions: List[str], max_tokens: int = 100) -> List[str]:
        """Generate answers for batch."""
        answers = []
        for q in questions:
            answers.append(self.generate_answer(q, max_tokens))
            time.sleep(0.5)  # Rate limiting
        return answers

    def get_name(self) -> str:
        return f"Anthropic_{self.model.split('-')[1]}"

    def get_cost_usd(self) -> float:
        return self.total_cost


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-3.5, GPT-4)."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.total_cost = 0.0
        self.calls_made = 0

    def generate_answer(self, question: str, max_tokens: int = 100) -> str:
        """Generate answer using GPT."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            self.calls_made += 1
            # Cost estimates (as of 2026)
            # GPT-3.5: $0.50/M input, $1.50/M output
            # GPT-4: $3/M input, $6/M output
            cost_per_1k_input = 0.0005 if "3.5" in self.model else 0.003
            cost_per_1k_output = 0.0015 if "3.5" in self.model else 0.006
            self.total_cost += (response.usage.prompt_tokens / 1000 * cost_per_1k_input)
            self.total_cost += (response.usage.completion_tokens / 1000 * cost_per_1k_output)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""

    def batch_generate(self, questions: List[str], max_tokens: int = 100) -> List[str]:
        """Generate answers for batch."""
        answers = []
        for q in questions:
            answers.append(self.generate_answer(q, max_tokens))
            time.sleep(0.5)
        return answers

    def get_name(self) -> str:
        return f"OpenAI_{self.model.replace('-', '_')}"

    def get_cost_usd(self) -> float:
        return self.total_cost


class HuggingFaceInferenceProvider(LLMProvider):
    """HuggingFace Inference API provider."""

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("Install huggingface-hub: pip install huggingface-hub")

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")

        self.client = InferenceClient(token=hf_token)
        self.model = model
        self.total_cost = 0.0  # Free tier available
        self.calls_made = 0

    def generate_answer(self, question: str, max_tokens: int = 100) -> str:
        """Generate answer using HuggingFace Inference."""
        try:
            response = self.client.text_generation(
                prompt=question,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=0.7
            )
            self.calls_made += 1
            return response if isinstance(response, str) else response.get("generated_text", "")
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            return ""

    def batch_generate(self, questions: List[str], max_tokens: int = 100) -> List[str]:
        """Generate answers for batch."""
        answers = []
        for q in questions:
            answers.append(self.generate_answer(q, max_tokens))
            time.sleep(0.5)
        return answers

    def get_name(self) -> str:
        return f"HF_{self.model.split('/')[-1]}"

    def get_cost_usd(self) -> float:
        return self.total_cost


class CostTracker:
    """Track and report API costs."""

    def __init__(self):
        self.costs_by_provider = defaultdict(float)
        self.calls_by_provider = defaultdict(int)

    def add_cost(self, provider: str, cost: float):
        self.costs_by_provider[provider] += cost
        self.calls_by_provider[provider] += 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return cost summary by provider."""
        return {
            provider: {
                "total_cost_usd": cost,
                "num_calls": self.calls_by_provider[provider],
                "cost_per_call": cost / max(self.calls_by_provider[provider], 1)
            }
            for provider, cost in self.costs_by_provider.items()
        }


class MultiProviderBenchmark:
    """Run hallucination detection benchmark across multiple providers."""

    def __init__(self, local_model: str = "EleutherAI/pythia-160m", device: str = "cuda"):
        self.device = device
        self.local_model_name = local_model

        # Load local model for feature extraction
        logger.info(f"Loading local model: {local_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model,
            output_attentions=True,
            device_map=device
        )
        self.model.eval()

        self.feature_engineer = AttentionFeatureEngineer(context_length=128)
        self.detector = None
        self.cost_tracker = CostTracker()

    def run(
        self,
        providers: List[LLMProvider],
        dataset: str = "halueval",
        num_samples: int = 100,
        detector_path: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark across providers.

        Args:
            providers: List of LLMProvider instances
            dataset: "halueval" or "custom"
            num_samples: Number of samples to benchmark
            detector_path: Path to pre-trained detector pickle
            dry_run: If True, don't run actual API calls (for testing)

        Returns:
            Dictionary mapping provider name to BenchmarkResult
        """

        # Load test data
        logger.info(f"Loading {dataset} dataset...")
        if dataset == "halueval":
            data_gen = DataGenerator()
            samples = data_gen.from_halueval(num_samples=num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Load or train detector
        if detector_path and os.path.exists(detector_path):
            logger.info(f"Loading detector from {detector_path}")
            self.detector = HallucinationDetector.load(detector_path)
        else:
            logger.warning("No pre-trained detector provided; using random baseline")
            self.detector = HallucinationDetector(classifier_type="logreg")

        results = {}

        for provider in providers:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running benchmark for {provider.get_name()}")
            logger.info(f"{'='*60}")

            result = self._run_single_provider(
                provider=provider,
                samples=samples[:num_samples],
                dry_run=dry_run
            )

            results[provider.get_name()] = result
            logger.info(f"AUROC: {result.auroc:.4f} [CI: {result.ci_lower:.4f}, {result.ci_upper:.4f}]")
            logger.info(f"Cost: ${result.cost_usd:.2f}")

        return results

    def _run_single_provider(
        self,
        provider: LLMProvider,
        samples: List[LabeledSample],
        dry_run: bool = False
    ) -> BenchmarkResult:
        """Run benchmark for single provider."""

        predictions = []
        true_labels = []
        latencies = []

        num_correct = 0
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"Processing sample {i}/{len(samples)}")

            if dry_run:
                # For testing: use dummy predictions
                answer = "dummy answer"
                pred_label = np.random.choice([0, 1])
                latency_ms = 10.0
            else:
                # Generate answer using provider
                start = time.time()
                answer = provider.generate_answer(sample.question, max_tokens=100)
                latency_ms = (time.time() - start) * 1000

            # Extract features and predict
            if answer:
                try:
                    # Run feature extraction
                    inputs = self.tokenizer(
                        answer,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs.to(self.device))

                    attentions = outputs.attentions
                    features = self.feature_engineer.extract(attentions, context_len=32)

                    # Predict
                    probs = self.detector.predict_proba(features.reshape(1, -1))
                    pred_label = 1 if probs[0] > 0.5 else 0
                except Exception as e:
                    logger.warning(f"Feature extraction error: {e}")
                    pred_label = np.random.choice([0, 1])
            else:
                pred_label = 0

            true_label = 1 if sample.label == "hallucinated" else 0
            predictions.append(pred_label)
            true_labels.append(true_label)
            latencies.append(latency_ms)

            if pred_label == true_label:
                num_correct += 1

        # Compute metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # AUROC with bootstrap CI
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
            auroc = roc_auc_score(true_labels, predictions)
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
        except:
            auroc = num_correct / len(true_labels)
            accuracy = auroc
            precision = recall = f1 = 0.0

        # Bootstrap CI
        auroc_samples = []
        for _ in range(1000):
            idx = np.random.choice(len(true_labels), len(true_labels), replace=True)
            try:
                auroc_samples.append(roc_auc_score(true_labels[idx], predictions[idx]))
            except:
                pass

        if auroc_samples:
            ci_lower = np.percentile(auroc_samples, 2.5)
            ci_upper = np.percentile(auroc_samples, 97.5)
        else:
            ci_lower = ci_upper = auroc

        return BenchmarkResult(
            provider=provider.get_name(),
            model_name=provider.model if hasattr(provider, 'model') else "unknown",
            num_samples=len(samples),
            auroc=auroc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_latency_ms=np.mean(latencies),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            cost_usd=provider.get_cost_usd(),
            timestamp=datetime.now().isoformat(),
            notes=f"Latency std: {np.std(latencies):.2f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description="Multi-provider hallucination detection benchmark")
    parser.add_argument("--providers", nargs="+", default=["anthropic"],
                        help="LLM providers to benchmark: anthropic, openai, huggingface")
    parser.add_argument("--dataset", default="halueval", help="Dataset: halueval or custom")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--model", default="EleutherAI/pythia-160m", help="Local model for feature extraction")
    parser.add_argument("--detector", default=None, help="Path to pre-trained detector pickle")
    parser.add_argument("--output", default="results/api_benchmark.json", help="Output JSON file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Don't make actual API calls")

    args = parser.parse_args()

    # Initialize providers
    providers = []
    for prov in args.providers:
        if prov.lower() == "anthropic":
            try:
                providers.append(AnthropicProvider())
            except Exception as e:
                logger.error(f"Could not initialize Anthropic: {e}")
        elif prov.lower() == "openai":
            try:
                providers.append(OpenAIProvider())
            except Exception as e:
                logger.error(f"Could not initialize OpenAI: {e}")
        elif prov.lower() == "huggingface":
            try:
                providers.append(HuggingFaceInferenceProvider())
            except Exception as e:
                logger.error(f"Could not initialize HuggingFace: {e}")
        else:
            logger.warning(f"Unknown provider: {prov}")

    if not providers:
        logger.error("No valid providers initialized")
        return

    # Run benchmark
    benchmark = MultiProviderBenchmark(local_model=args.model, device=args.device)
    results = benchmark.run(
        providers=providers,
        dataset=args.dataset,
        num_samples=args.samples,
        detector_path=args.detector,
        dry_run=args.dry_run
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        results_dict = {k: asdict(v) for k, v in results.items()}
        json.dump(results_dict, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")

    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY TABLE")
    logger.info("="*80)
    logger.info(f"{'Provider':<30} {'AUROC':<10} {'CI':<25} {'Cost (USD)':<12}")
    logger.info("-"*80)
    for result in results.values():
        ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        logger.info(f"{result.provider:<30} {result.auroc:<10.4f} {ci_str:<25} ${result.cost_usd:>10.2f}")


if __name__ == "__main__":
    main()
