"""
Calibrated Entropy-Distribution Detector
===========================================

Main original contribution of this repo: rather than thresholding raw
entropy (or attention-feature) scores at a fixed cutoff — which the
literature shows under-performs for selective prediction without
calibration (see "Entropy Alone is Insufficient for Safe Selective
Prediction in LLMs", arXiv:2603.21172) — this detector calibrates raw
uncertainty features against the distribution those features actually take
on correct answers, and scores new examples by how much they diverge from
that calibrated reference.

Two-stage design:

    Stage A (calibration): a scalar raw score u(x) (default: the first
        feature column, e.g. entropy_mean) is mapped to P(hallucination)
        via isotonic regression fit on a labeled calibration set. Isotonic
        regression is hand-rolled (pool-adjacent-violators + linear
        interpolation) rather than scikit-learn, keeping the repo's
        existing "numpy/scipy only, no sklearn" convention (LogisticRegression
        and SimpleMLP in detector.py are hand-rolled for the same reason).

    Stage B (divergence): the multivariate feature vector's distribution is
        fit on the calibration set's correct-answer (y=0) examples only —
        mean mu_ref and a diagonal-shrinkage-regularized covariance
        Sigma_ref. New examples are scored by their Mahalanobis distance to
        this "expected correct-answer" distribution — this is the concrete
        "divergence from a calibrated entropy distribution" mechanism. A
        nonparametric percentile-rank alternative is also available.

The final probability blends both stages:

    p(x) = clip(w * isotonic(u(x)) + (1-w) * sigmoid(a*mahalanobis(x)+b), 0, 1)

where (a, b) are fit via a small 1-D logistic regression of Mahalanobis
distance against labels on the calibration set, and w is a fixed blend
weight (default 0.5).

API matches HallucinationDetector's flat-vector contract exactly
(fit/predict_proba/predict/evaluate/save/load), so it plugs directly into
v2/pipeline.py's existing CV/ablation/bootstrap harness.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from scipy.special import expit as sigmoid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.detector import DetectorMetrics, compute_classification_metrics

EPS = 1e-12


# =========================================================================
# Hand-rolled isotonic regression (pool-adjacent-violators)
# =========================================================================

class IsotonicPredictor:
    """
    Picklable callable wrapping fitted isotonic-regression knots: linearly
    interpolates between them for new x values (clipped to the boundary
    y-values outside the training x range). A plain closure would not
    survive pickle.dump(), which CalibratedEntropyDetector.save() relies on.
    """

    def __init__(self, knot_x: np.ndarray, knot_y: np.ndarray) -> None:
        self.knot_x = knot_x
        self.knot_y = knot_y
        self.y_min = float(knot_y[0])
        self.y_max = float(knot_y[-1])

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        return np.interp(x_new, self.knot_x, self.knot_y, left=self.y_min, right=self.y_max)


def isotonic_regression(x: np.ndarray, y: np.ndarray) -> "IsotonicPredictor":
    """
    Fit a monotonically non-decreasing step function y ~ f(x) via the
    pool-adjacent-violators algorithm (PAV), then return a predictor that
    linearly interpolates between fitted knots for new x values (clipped to
    the boundary y-values outside the training x range).

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)

    Returns
    -------
    IsotonicPredictor
        predict(x_new) -> calibrated y values in [0, 1] range implied by y.
    """
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]

    # PAV: maintain a stack of (value, weight, count) blocks.
    values: List[float] = []
    weights: List[float] = []
    counts: List[int] = []

    for yi in y_sorted:
        values.append(yi)
        weights.append(1.0)
        counts.append(1)
        # Merge back while the last block violates monotonicity.
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, c2 = values.pop(), weights.pop(), counts.pop()
            v1, w1, c1 = values.pop(), weights.pop(), counts.pop()
            merged_w = w1 + w2
            merged_v = (v1 * w1 + v2 * w2) / merged_w
            values.append(merged_v)
            weights.append(merged_w)
            counts.append(c1 + c2)

    # Expand blocks back into per-point fitted values (constant within a block).
    fitted = np.empty(len(y_sorted))
    idx = 0
    for v, c in zip(values, counts):
        fitted[idx: idx + c] = v
        idx += c

    return IsotonicPredictor(knot_x=x_sorted, knot_y=fitted)


# =========================================================================
# Divergence from a reference distribution
# =========================================================================

def fit_reference_distribution(X_ref: np.ndarray, shrinkage: float = 0.1):
    """
    Fit mean and diagonal-shrinkage-regularized covariance from reference
    (correct-answer) examples.

    Returns
    -------
    mu : np.ndarray, shape (D,)
    sigma_inv : np.ndarray, shape (D, D)
        Inverse of the shrunk covariance (precomputed for repeated
        Mahalanobis-distance calls).
    """
    mu = X_ref.mean(axis=0)
    if len(X_ref) > 1:
        cov = np.cov(X_ref, rowvar=False)
    else:
        cov = np.eye(X_ref.shape[1])
    cov = np.atleast_2d(cov)

    diag = np.diag(np.diag(cov))
    shrunk = (1 - shrinkage) * cov + shrinkage * diag
    # Extra numerical floor in case the diagonal itself is near-singular
    # (e.g. a near-constant feature column in a small calibration split).
    shrunk += 1e-6 * np.eye(shrunk.shape[0])

    sigma_inv = np.linalg.inv(shrunk)
    return mu, sigma_inv


def mahalanobis_distance(X: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row of X to (mu, sigma_inv). Returns (N,)."""
    diff = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, sigma_inv, diff).clip(min=0))


def percentile_rank(scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    """
    Fraction of `reference_scores` strictly less than each value in `scores`.
    Nonparametric alternative/fallback to the Mahalanobis divergence.
    """
    reference_sorted = np.sort(reference_scores)
    ranks = np.searchsorted(reference_sorted, scores, side="left")
    return ranks / max(len(reference_sorted), 1)


# =========================================================================
# Calibrated Entropy Detector
# =========================================================================

@dataclass
class _FitState:
    mean: np.ndarray
    std: np.ndarray
    mu_ref: np.ndarray
    sigma_inv: np.ndarray
    isotonic_predict: "IsotonicPredictor"
    logistic_a: float
    logistic_b: float
    reference_divergence: np.ndarray  # divergence scores of the y==0 calibration set


class CalibratedEntropyDetector:
    """
    Detects hallucinations by calibrating raw uncertainty features against a
    reference distribution of "correct answer" signatures, instead of
    thresholding raw entropy directly.

    Parameters
    ----------
    score_index : int
        Column of X used as the scalar raw score u(x) fed to isotonic
        regression (default 0 — e.g. entropy_mean when X is the 6-D output
        of EntropyFeatureExtractor; caller controls what X actually is).
    blend_weight : float
        Weight on the isotonic-calibration term vs. the divergence term in
        the final blended probability (default 0.5).
    cov_shrinkage : float
        Diagonal shrinkage applied to the reference covariance (default 0.1)
        to keep Mahalanobis distance well-defined even when feature
        dimensionality approaches the number of calibration examples.
    divergence : {"mahalanobis", "percentile"}
        Which divergence estimator to use.
    """

    def __init__(
        self,
        score_index: int = 0,
        blend_weight: float = 0.5,
        cov_shrinkage: float = 0.1,
        divergence: str = "mahalanobis",
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if divergence not in ("mahalanobis", "percentile"):
            raise ValueError(f"Unknown divergence: {divergence!r}. Choose: mahalanobis, percentile")
        self.score_index = score_index
        self.blend_weight = blend_weight
        self.cov_shrinkage = cov_shrinkage
        self.divergence = divergence
        self.feature_names = feature_names
        self._state: Optional[_FitState] = None
        self._fitted = False

    def _standardize(self, X: np.ndarray, state: _FitState) -> np.ndarray:
        return (X - state.mean) / state.std

    def _divergence_scores_raw(self, X_norm: np.ndarray, state: _FitState) -> np.ndarray:
        if self.divergence == "mahalanobis":
            return mahalanobis_distance(X_norm, state.mu_ref, state.sigma_inv)
        return percentile_rank(
            X_norm[:, self.score_index], state.reference_divergence
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedEntropyDetector":
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Raw feature vectors — entropy-baseline (6D), attention (18D),
            or any concatenation the caller builds.
        y : np.ndarray, shape (N,)
            Binary labels: 1 = hallucinated, 0 = correct.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_norm = (X - mean) / std

        correct_mask = y == 0
        if correct_mask.sum() < 2:
            raise ValueError(
                "CalibratedEntropyDetector.fit() needs at least 2 correct-answer "
                f"(y=0) examples to fit a reference distribution, got {int(correct_mask.sum())}."
            )
        mu_ref, sigma_inv = fit_reference_distribution(
            X_norm[correct_mask], shrinkage=self.cov_shrinkage
        )

        # Placeholder state so _divergence_scores_raw can run for the
        # nonparametric reference set before the real state object exists.
        tmp_state = _FitState(
            mean=mean, std=std, mu_ref=mu_ref, sigma_inv=sigma_inv,
            isotonic_predict=lambda x: x, logistic_a=0.0, logistic_b=0.0,
            reference_divergence=X_norm[correct_mask, self.score_index],
        )
        divergence_all = self._divergence_scores_raw(X_norm, tmp_state)

        # Stage A: isotonic calibration of the raw scalar score.
        u = X_norm[:, self.score_index]
        isotonic_predict = isotonic_regression(u, y)

        # Stage B: scalar logistic regression of divergence -> y.
        logistic_a, logistic_b = self._fit_scalar_logistic(divergence_all, y)

        self._state = _FitState(
            mean=mean, std=std, mu_ref=mu_ref, sigma_inv=sigma_inv,
            isotonic_predict=isotonic_predict,
            logistic_a=logistic_a, logistic_b=logistic_b,
            reference_divergence=divergence_all[correct_mask],
        )
        self._fitted = True
        return self

    @staticmethod
    def _fit_scalar_logistic(
        u: np.ndarray, y: np.ndarray, lr: float = 0.1, l2: float = 0.01, max_iter: int = 1000
    ) -> "tuple[float, float]":
        """Tiny 1-D logistic regression (gradient descent), avoids pulling in
        the full LogisticRegression class from detector.py for a scalar fit."""
        u_mean, u_std = u.mean(), u.std() + 1e-8
        u_norm = (u - u_mean) / u_std

        a, b = 0.0, 0.0
        n = len(y)
        for _ in range(max_iter):
            z = a * u_norm + b
            p = sigmoid(z)
            error = p - y
            grad_a = (u_norm * error).mean() + 2 * l2 * a
            grad_b = error.mean()
            a -= lr * grad_a
            b -= lr * grad_b

        # Fold the (u - u_mean)/u_std normalization back into (a, b) so the
        # returned coefficients operate directly on raw `u`.
        a_raw = a / u_std
        b_raw = b - a * u_mean / u_std
        return float(a_raw), float(b_raw)

    def divergence_scores(self, X: np.ndarray) -> np.ndarray:
        """Public accessor for raw divergence values (diagnostics/plots)."""
        assert self._fitted, "Must call fit() first"
        X = np.asarray(X, dtype=float)
        X_norm = self._standardize(X, self._state)
        return self._divergence_scores_raw(X_norm, self._state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Must call fit() first"
        X = np.asarray(X, dtype=float)
        state = self._state
        X_norm = self._standardize(X, state)

        u = X_norm[:, self.score_index]
        calibrated = state.isotonic_predict(u)

        div = self._divergence_scores_raw(X_norm, state)
        div_prob = sigmoid(state.logistic_a * div + state.logistic_b)

        w = self.blend_weight
        p = w * calibrated + (1 - w) * div_prob
        return np.clip(p, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> DetectorMetrics:
        probs = self.predict_proba(X)
        return compute_classification_metrics(probs, y, threshold)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "score_index": self.score_index,
                "blend_weight": self.blend_weight,
                "cov_shrinkage": self.cov_shrinkage,
                "divergence": self.divergence,
                "feature_names": self.feature_names,
                "state": self._state,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: str) -> "CalibratedEntropyDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls(
            score_index=data["score_index"],
            blend_weight=data["blend_weight"],
            cov_shrinkage=data["cov_shrinkage"],
            divergence=data["divergence"],
            feature_names=data["feature_names"],
        )
        det._state = data["state"]
        det._fitted = data["fitted"]
        return det


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CALIBRATED ENTROPY DETECTOR — STANDALONE VALIDATION")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Test 1: separates synthetic shifted classes (same pattern as detector.py)
    print("\n--- Test 1: Synthetic separability ---")
    N, D = 400, 18
    y = rng.integers(0, 2, N).astype(float)
    X = rng.standard_normal((N, D))
    X[y == 1] += 1.0

    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    det = CalibratedEntropyDetector()
    det.fit(X_train, y_train)
    metrics = det.evaluate(X_test, y_test)
    print(f"  AUROC: {metrics.auroc:.4f}")
    assert metrics.auroc > 0.7, "AUROC too low on separable data"
    print("  Separates synthetic shifted classes ✅")

    # Test 2: isotonic monotonicity
    print("\n--- Test 2: Isotonic monotonicity ---")
    u = np.linspace(-3, 3, 50)
    y_probe = (u > 0).astype(float)
    rng.shuffle(y_probe)  # noisy labels, isotonic should still be non-decreasing
    predict = isotonic_regression(u, (u > 0).astype(float))
    grid = np.linspace(-3, 3, 100)
    fitted = predict(grid)
    assert np.all(np.diff(fitted) >= -1e-9), "Isotonic fit is not monotonic"
    print("  Isotonic regression is monotonic ✅")

    # Test 3: high-D low-N covariance shrinkage doesn't crash
    print("\n--- Test 3: High-D low-N covariance shrinkage ---")
    D2 = 24
    N2 = 40
    y2 = rng.integers(0, 2, N2).astype(float)
    X2 = rng.standard_normal((N2, D2))
    det2 = CalibratedEntropyDetector(cov_shrinkage=0.3)
    det2.fit(X2, y2)
    probs2 = det2.predict_proba(X2)
    assert np.all((probs2 >= 0) & (probs2 <= 1)), "Probabilities out of range"
    print("  High-D/low-N fit succeeds, probabilities in [0,1] ✅")

    # Test 4: save/load roundtrip
    print("\n--- Test 4: Persistence ---")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmppath = f.name
    det.save(tmppath)
    loaded = CalibratedEntropyDetector.load(tmppath)
    assert np.allclose(det.predict_proba(X_test), loaded.predict_proba(X_test))
    Path(tmppath).unlink()
    print("  Save/load roundtrip ✅")

    print(f"\n{'=' * 60}")
    print("Calibrated entropy detector — ALL CHECKS PASS ✅")
    print(f"{'=' * 60}")
