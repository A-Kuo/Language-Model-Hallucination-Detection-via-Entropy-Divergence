"""
Agent Routing MVP — Benchmark Runner
=======================================

The MVP's one evaluation script. Loads a small local model, builds a
calibration set by grading the model's OWN no-context answers against
SQuAD 2.0 gold answers (see benchmarks/tasks.py::build_calibration_set),
fits a CalibratedEntropyDetector on it, then runs the answer/retrieve/
abstain agent (agent/router.py) over a held-out benchmark split and reports
task success, answer accuracy, and abstention quality. Mirrors pipeline.py's
CLI/JSON-summary conventions.

Usage:
    python experiments/run_benchmark.py --num_samples 200
    python experiments/run_benchmark.py --num_samples 200 \\
        --results results/agent_routing_pythia160m_n200.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from abstention import area_under_risk_coverage, risk_coverage_curve
from agent.router import AgentAction, RoutingPolicy
from benchmarks.loaders import load_squad_v2
from benchmarks.schemas import AgentTraceRecord
from benchmarks.tasks import build_calibration_set, run_all_tasks
from calibrated_entropy_detector import CalibratedEntropyDetector


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None — Python's json module
    emits the bare (non-standard) literals NaN/Infinity by default, which
    strict JSON parsers (e.g. JS's JSON.parse) reject outright. `agent`
    summary stats are legitimately NaN when a rate has no denominator
    (e.g. answer_accuracy with zero graded answers — see the all-abstain
    finding this MVP's first real run produced)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _summarize(records: List[AgentTraceRecord]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"num_samples": 0}

    action_counts = {a.value: 0 for a in AgentAction}
    for r in records:
        action_counts[r.action_sequence[-1]] += 1

    graded = [r for r in records if r.correct is not None]
    abstained = [r for r in records if r.action_sequence[-1] == "abstain"]
    unanswerable = [r for r in records if not r.is_answerable]

    return {
        "num_samples": n,
        "action_counts": action_counts,
        "task_success_rate": float(np.mean([r.task_success for r in records])),
        "answer_accuracy": float(np.mean([r.correct for r in graded])) if graded else float("nan"),
        "retrieval_rate": float(np.mean(["retrieve" in r.action_sequence for r in records])),
        "abstain_rate": len(abstained) / n,
        # Of items the agent abstained on, fraction that were genuinely unanswerable.
        "abstain_precision": (
            float(np.mean([not r.is_answerable for r in abstained])) if abstained else float("nan")
        ),
        # Of genuinely unanswerable items, fraction the agent correctly abstained on.
        "abstain_recall": (
            float(np.mean([r.action_sequence[-1] == "abstain" for r in unanswerable]))
            if unanswerable else float("nan")
        ),
        "mean_latency_s": float(np.mean([r.latency_s for r in records])),
    }


def run_benchmark(
    num_samples: int = 200,
    calibration_fraction: float = 0.4,
    model_name: str = "EleutherAI/pythia-160m",
    seed: int = 42,
    max_new_tokens: int = 40,
    results_path: Optional[str] = None,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    items = load_squad_v2(num_samples=num_samples, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    split = int(calibration_fraction * len(items))
    calib_items = [items[i] for i in idx[:split]]
    bench_items = [items[i] for i in idx[split:]]
    print(f"  {len(calib_items)} calibration items, {len(bench_items)} benchmark items")

    # Two calibration sets on the SAME calibration items, not two disjoint
    # splits: each item generates both a no-context and a with-context
    # answer, so each detector is fit on the feature regime it will actually
    # score at inference time (see agent/router.py::RoutingPolicy's
    # docstring for why a single shared detector broke).
    print("\nBuilding no-context calibration set (graded against SQuAD gold answers)...")
    X_noctx, y_noctx = build_calibration_set(
        calib_items, model, tokenizer, max_new_tokens=max_new_tokens, device=device,
        use_context=False,
    )
    print("\nBuilding with-context calibration set (graded against SQuAD gold answers)...")
    X_ctx, y_ctx = build_calibration_set(
        calib_items, model, tokenizer, max_new_tokens=max_new_tokens, device=device,
        use_context=True,
    )

    n_correct_noctx = int((y_noctx == 0).sum())
    n_hallucinated_noctx = int((y_noctx == 1).sum())
    n_correct_ctx = int((y_ctx == 0).sum())
    n_hallucinated_ctx = int((y_ctx == 1).sum())
    print(f"  No-context calibration set:   {X_noctx.shape}, {n_hallucinated_noctx} hallucinated / {n_correct_noctx} correct")
    print(f"  With-context calibration set: {X_ctx.shape}, {n_hallucinated_ctx} hallucinated / {n_correct_ctx} correct")

    failing_modes = []
    if n_correct_noctx < 2:
        failing_modes.append(("no-context", n_correct_noctx, len(y_noctx)))
    if n_correct_ctx < 2:
        failing_modes.append(("with-context", n_correct_ctx, len(y_ctx)))

    if failing_modes:
        # CalibratedEntropyDetector needs >= 2 correct-answer examples to
        # fit its reference distribution (see calibrated_entropy_detector.py
        # ::fit). Rather than force that by inflating --num_samples until
        # luck produces enough, report this as what it is: a real finding
        # about how hard the failing calibration task is for this model,
        # not a bug to route around. The agent run needs BOTH detectors, so
        # it's skipped if either one can't be fit.
        for mode, n_correct, n_total in failing_modes:
            acc = n_correct / n_total if n_total else float("nan")
            print(
                f"\n  NOTE: only {n_correct}/{n_total} {mode} calibration answers were correct "
                f"({acc:.1%}) — too few to fit a CalibratedEntropyDetector reference distribution "
                "(needs >= 2)."
            )
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "device": device,
            "num_samples_total": len(items),
            "num_calibration": len(calib_items),
            "num_benchmark": len(bench_items),
            "calibration_no_context_num_correct": n_correct_noctx,
            "calibration_no_context_num_hallucinated": n_hallucinated_noctx,
            "calibration_with_context_num_correct": n_correct_ctx,
            "calibration_with_context_num_hallucinated": n_hallucinated_ctx,
            "agent": None,
            "note": (
                "One or both calibration sets had fewer than 2 correct answers — "
                "CalibratedEntropyDetector.fit() requires at least 2 to fit a reference "
                "distribution, so the agent run (which needs both detectors) was skipped. "
                f"Failing mode(s): {', '.join(m for m, _, _ in failing_modes)}."
            ),
        }
        if results_path:
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(_json_safe(summary), f, indent=2, default=str)
            print(f"\nResults summary saved to {results_path}")
        return summary

    detector_noctx = CalibratedEntropyDetector()
    detector_noctx.fit(X_noctx, y_noctx)
    detector_ctx = CalibratedEntropyDetector()
    detector_ctx.fit(X_ctx, y_ctx)
    policy = RoutingPolicy(detector_noctx, detector_ctx)

    print(f"\nRunning agent over {len(bench_items)} benchmark items...")
    records = run_all_tasks(
        bench_items, model, tokenizer, policy, max_new_tokens=max_new_tokens, device=device,
    )

    agent_summary = _summarize(records)

    # Abstention quality on each detector's own calibration set, via the
    # existing risk-coverage machinery (a detector-level view alongside the
    # agent-specific abstain_precision/recall above, which measures the
    # *agent's* abstain decisions specifically) — one AURC per detector,
    # since they're now fit on different feature regimes.
    aurc_noctx = area_under_risk_coverage(
        risk_coverage_curve(detector_noctx.predict_proba(X_noctx), y_noctx)
    )
    aurc_ctx = area_under_risk_coverage(
        risk_coverage_curve(detector_ctx.predict_proba(X_ctx), y_ctx)
    )

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "device": device,
        "num_samples_total": len(items),
        "num_calibration": len(calib_items),
        "num_benchmark": len(bench_items),
        "num_benchmark_used": len(records),
        "num_failed": len(bench_items) - len(records),
        "calibration_no_context_num_correct": n_correct_noctx,
        "calibration_no_context_num_hallucinated": n_hallucinated_noctx,
        "calibration_with_context_num_correct": n_correct_ctx,
        "calibration_with_context_num_hallucinated": n_hallucinated_ctx,
        "agent": agent_summary,
        "calibration_no_context_aurc": aurc_noctx,
        "calibration_with_context_aurc": aurc_ctx,
        "example_traces": [asdict(r) for r in records[:10]],
    }

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(_json_safe(summary), f, indent=2, default=str)
        print(f"\nResults summary saved to {results_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Agent routing MVP benchmark (answer / retrieve / abstain, SQuAD 2.0)"
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--calibration_fraction", type=float, default=0.4)
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  AGENT ROUTING MVP — answer / retrieve / abstain")
    print("=" * 60)

    summary = run_benchmark(
        num_samples=args.num_samples,
        calibration_fraction=args.calibration_fraction,
        model_name=args.model,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        results_path=args.results,
    )

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    if summary["agent"] is None:
        print(f"  {summary['note']}")
    else:
        for k, v in summary["agent"].items():
            print(f"  {k}: {v}")
        print(f"  calibration_no_context_aurc: {summary['calibration_no_context_aurc']}")
        print(f"  calibration_with_context_aurc: {summary['calibration_with_context_aurc']}")


if __name__ == "__main__":
    main()
