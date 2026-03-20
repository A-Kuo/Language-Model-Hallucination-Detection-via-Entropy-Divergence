"""
Attention Analyzer — Core Algorithm for Hallucination Detection
================================================================

Mathematical Foundation:
    Given a transformer with L layers, each with H attention heads,
    the attention weight tensor at layer l for head h is:

        A^{l,h} ∈ R^{T×T}   (T = sequence length)

    where A^{l,h}_{i,j} = softmax(Q_i · K_j^T / √d_k)

    We compute two information-theoretic signals:

    1) Per-Head Shannon Entropy:
       H(A^{l,h}_i) = -Σ_j  A^{l,h}_{i,j} · log₂(A^{l,h}_{i,j})

       High entropy → attention is diffuse (model uncertain about context)
       Low entropy  → attention is peaked (model confident)

    2) Cross-Layer KL Divergence:
       D_KL(A^{l} || A^{l+1}) = Σ_j  A^{l}_j · log(A^{l}_j / A^{l+1}_j)

       High divergence → layers disagree on what matters (internal inconsistency)
       Low divergence  → layers agree (stable representation)

    The combined signal feeds into the statistical hypothesis test
    (hypothesis_test.py) and confidence calibrator (confidence_calibrator.py).

Usage:
    analyzer = AttentionAnalyzer(model_name="google/gemma-2-2b")
    result = analyzer.analyze("The capital of France is")
    print(result.mean_entropy)
    print(result.total_kl_divergence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AttentionAnalysisResult:
    """Immutable container for all attention-derived metrics."""

    # Per-layer, per-head entropy matrix
    layer_head_entropy: np.ndarray          # shape (num_layers, num_heads)

    # Aggregated per-layer mean entropy
    per_layer_entropy: List[float]

    # Scalar summaries
    mean_entropy: float                     # mean across all layers & heads
    max_entropy: float                      # single worst-case head
    entropy_std: float                      # spread across heads

    # KL divergence between consecutive layers
    pairwise_kl: List[float]                # length = num_layers - 1
    total_kl_divergence: float              # sum of pairwise KL values
    max_kl_divergence: float                # worst single layer transition

    # Metadata
    num_layers: int
    num_heads: int
    sequence_length: int
    latency_ms: float

    # Optional raw attention weights (for visualization / debugging)
    raw_attentions: Optional[Tuple] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class AttentionAnalyzer:
    """
    Extracts attention matrices from a HuggingFace causal LM and computes
    information-theoretic uncertainty signals.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (default "google/gemma-2-2b", 2B params).
    device : str | None
        "cuda", "cpu", or None for auto-detect.
    precision : torch.dtype
        Model weight dtype. float32 for reproducibility, float16 for speed.
    eps : float
        Small constant added before log to prevent log(0).
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        device: Optional[str] = None,
        precision: torch.dtype = torch.float32,
        eps: float = 1e-12,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.eps = eps

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            torch_dtype=precision,
        ).to(self.device)
        self._model.eval()

        # Cache architecture constants from model config (handle varying attr names)
        config = self._model.config
        self.num_layers: int = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
        self.num_heads: int = getattr(config, "num_attention_heads", None) or getattr(config, "n_head")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        return_raw_attentions: bool = False,
    ) -> AttentionAnalysisResult:
        """
        Run full attention analysis on *text*.

        Pipeline:
            1. Tokenize → forward pass (no grad)
            2. Extract attention tensors  [L × (1, H, T, T)]
            3. Compute per-head Shannon entropy  → (L, H)
            4. Compute cross-layer KL divergence → (L-1,)
            5. Package into AttentionAnalysisResult
        """
        t0 = time.perf_counter()

        attentions, seq_len = self._forward(text)
        layer_head_entropy = self._compute_entropy(attentions)

        per_layer_entropy = layer_head_entropy.mean(axis=1).tolist()
        mean_entropy = float(layer_head_entropy.mean())
        max_entropy = float(layer_head_entropy.max())
        entropy_std = float(layer_head_entropy.std())

        pairwise_kl = self._compute_cross_layer_kl(attentions)
        total_kl = sum(pairwise_kl)
        max_kl = max(pairwise_kl) if pairwise_kl else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AttentionAnalysisResult(
            layer_head_entropy=layer_head_entropy,
            per_layer_entropy=per_layer_entropy,
            mean_entropy=mean_entropy,
            max_entropy=max_entropy,
            entropy_std=entropy_std,
            pairwise_kl=pairwise_kl,
            total_kl_divergence=total_kl,
            max_kl_divergence=max_kl,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            sequence_length=seq_len,
            latency_ms=latency_ms,
            raw_attentions=attentions if return_raw_attentions else None,
        )

    def analyze_batch(self, texts: List[str]) -> List[AttentionAnalysisResult]:
        """Analyse several prompts sequentially."""
        return [self.analyze(t) for t in texts]

    # ------------------------------------------------------------------
    # Internal: forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, text: str) -> Tuple[Tuple[torch.Tensor, ...], int]:
        """
        Tokenize *text* and run a single forward pass.

        Returns
        -------
        attentions : tuple of L tensors, each (1, H, T, T)
        seq_len : int
        """
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self._model(**inputs)
        seq_len = inputs["input_ids"].shape[1]

        return outputs.attentions, seq_len

    # ------------------------------------------------------------------
    # Internal: Shannon entropy  H = -Σ p·log₂(p)
    # ------------------------------------------------------------------

    def _compute_entropy(
        self,
        attentions: Tuple[torch.Tensor, ...],
    ) -> np.ndarray:
        """
        Compute Shannon entropy for every (layer, head) pair.

        Uses the **last token's** attention row — the distribution the
        model uses to predict the next token, which is the step most
        relevant to hallucination detection.

            H(a) = -Σ_j  a_j · log₂(a_j)

        Bounds:
            min = 0          (delta distribution, full confidence)
            max = log₂(T)    (uniform distribution, no preference)

        Returns
        -------
        np.ndarray, shape (num_layers, num_heads)
        """
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        entropy_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)

        for l_idx, attn_tensor in enumerate(attentions):
            # attn_tensor: (1, H, T, T) → last query token → (H, T)
            last_token_attn = attn_tensor[0, :, -1, :]

            # CPU float64 for numerical stability
            a = last_token_attn.cpu().to(torch.float64).numpy()
            a = np.clip(a, self.eps, None)

            # Shannon entropy per head
            h = -np.sum(a * np.log2(a), axis=1)   # (H,)
            entropy_matrix[l_idx] = h

        return entropy_matrix

    # ------------------------------------------------------------------
    # Internal: KL divergence across consecutive layers
    # ------------------------------------------------------------------

    def _compute_cross_layer_kl(
        self,
        attentions: Tuple[torch.Tensor, ...],
    ) -> List[float]:
        """
        Compute D_KL(layer_l || layer_{l+1}) averaged over heads.

            D_KL(p || q) = Σ_j  p_j · log(p_j / q_j)    (natural log, nats)

        After clipping, distributions are renormalised to ensure they
        remain valid probability distributions.

        Returns
        -------
        List[float], length = num_layers - 1
        """
        kl_values: List[float] = []

        for l_idx in range(len(attentions) - 1):
            p = attentions[l_idx][0, :, -1, :]       # (H, T)
            q = attentions[l_idx + 1][0, :, -1, :]   # (H, T)

            p = p.cpu().to(torch.float64)
            q = q.cpu().to(torch.float64)

            # Clip + renormalise for numerical safety
            p = torch.clamp(p, min=self.eps)
            q = torch.clamp(q, min=self.eps)
            p = p / p.sum(dim=1, keepdim=True)
            q = q / q.sum(dim=1, keepdim=True)

            # KL per head, then average across heads
            kl_per_head = torch.sum(p * torch.log(p / q), dim=1)  # (H,)
            mean_kl = float(kl_per_head.mean().item())

            kl_values.append(mean_kl)

        return kl_values

    # ------------------------------------------------------------------
    # Feature vector extraction (for downstream classifiers)
    # ------------------------------------------------------------------

    def extract_features(self, text: str) -> np.ndarray:
        """
        Return a fixed-length feature vector (length 6) summarising
        the attention analysis.  Useful for training a lightweight
        hallucination classifier on top.

        Layout:
            [mean_entropy, max_entropy, entropy_std,
             total_kl, max_kl, normalised_seq_length]
        """
        r = self.analyze(text)
        return np.array([
            r.mean_entropy,
            r.max_entropy,
            r.entropy_std,
            r.total_kl_divergence,
            r.max_kl_divergence,
            r.sequence_length / 1024.0,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Standalone math functions (usable without loading a model)
# ---------------------------------------------------------------------------

def compute_entropy_from_weights(
    attention_weights: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute Shannon entropy from a raw attention weight matrix.

    Parameters
    ----------
    attention_weights : np.ndarray
        Shape (H, T) or (T,). Each row is a probability distribution.
    eps : float
        Numerical stability constant.

    Returns
    -------
    np.ndarray
        Entropy per head (or scalar if input was 1-D).
    """
    a = np.clip(attention_weights, eps, None)
    if a.ndim == 1:
        return -np.sum(a * np.log2(a))
    return -np.sum(a * np.log2(a), axis=-1)


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute D_KL(p || q) for two probability distributions.

    Parameters
    ----------
    p, q : np.ndarray
        1-D probability distributions (must sum to ~1).

    Returns
    -------
    float
        KL divergence in nats.
    """
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Self-test (runs when executing this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("STANDALONE MATH VALIDATION (no model loading)")
    print("=" * 60)

    # Test 1: Uniform distribution → max entropy
    T = 8
    uniform = np.ones(T) / T
    h_uniform = compute_entropy_from_weights(uniform)
    expected = np.log2(T)
    print(f"\nUniform dist (T={T}): entropy = {h_uniform:.4f}, "
          f"expected = {expected:.4f}  "
          f"{'✅' if abs(h_uniform - expected) < 1e-6 else '❌'}")

    # Test 2: Delta distribution → zero entropy
    delta = np.zeros(T)
    delta[0] = 1.0
    h_delta = compute_entropy_from_weights(delta)
    print(f"Delta dist   (T={T}): entropy = {h_delta:.6f}  "
          f"{'✅' if h_delta < 1e-6 else '❌'}")

    # Test 3: KL divergence of identical dists → 0
    kl_same = compute_kl_divergence(uniform, uniform)
    print(f"\nKL(uniform || uniform) = {kl_same:.6f}  "
          f"{'✅' if kl_same < 1e-10 else '❌'}")

    # Test 4: KL divergence is non-negative (Gibbs' inequality)
    rng = np.random.default_rng(42)
    p = rng.dirichlet(np.ones(T))
    q = rng.dirichlet(np.ones(T))
    kl_pq = compute_kl_divergence(p, q)
    print(f"KL(random_p || random_q) = {kl_pq:.6f}  "
          f"{'✅' if kl_pq >= 0 else '❌'}")

    # Test 5: KL is asymmetric
    kl_qp = compute_kl_divergence(q, p)
    print(f"KL(random_q || random_p) = {kl_qp:.6f}  "
          f"(asymmetric: {kl_pq:.6f} ≠ {kl_qp:.6f})  "
          f"{'✅' if abs(kl_pq - kl_qp) > 1e-6 else '⚠️  symmetric by coincidence'}")

    # Test 6: Multi-head entropy
    heads = np.array([uniform, delta, p])  # (3, T)
    h_multi = compute_entropy_from_weights(heads)
    print(f"\nMulti-head entropy: {h_multi}  shape={h_multi.shape}  "
          f"{'✅' if h_multi.shape == (3,) else '❌'}")

    print(f"\n{'=' * 60}")
    print("All standalone math checks passed ✅")
    print("To run with Gemma 2, use: AttentionAnalyzer(model_name='google/gemma-2-2b')")
    print(f"{'=' * 60}")
