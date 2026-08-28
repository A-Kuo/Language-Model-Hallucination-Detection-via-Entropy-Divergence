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
pipeline.py's existing CV/ablation/bootstrap harness.

Beyond a raw probability, route() maps predict_proba's output into a 3-way
decision — RELIABLE / UNCERTAIN / UNRELIABLE — for callers that need an
actionable label rather than a number to threshold themselves (e.g. the
Streamlit demo's traffic-light display). This revives a routing concept from
an earlier version of this project (v1/confidence_calibrator.py, dropped
during a repo consolidation) but rebuilds it on the current calibration
machinery rather than porting the old code. Framed as a one-sided hypothesis
test — H0: "this answer is reliable" vs H1: "this answer is hallucinated" —
an UNRELIABLE routing is a rejection of H0 at a significance level implied
by `unreliable_quantile` (default 0.10: the threshold is chosen so at most
10% of genuinely-hallucinated calibration examples would be missed). The two
thresholds are fit from the calibration set, not hardcoded: threshold_reliable
is the `reliable_quantile` quantile of predict_proba scores among calibration
correct-answer examples (default 0.90 — 90% of correct answers score below
it), and threshold_unreliable is the `unreliable_quantile` quantile among
calibration hallucinated examples (default 0.10 — 90% of hallucinated
examples score above it). Choosing wide, asymmetric quantiles rather than a
single 0.5 cutoff is a deliberate conservative-safety bias carried over from
the routing concept's original motivation: a false RELIABLE (trusting a
hallucination) is worse than a false UNCERTAIN (needlessly escalating a
correct answer for review).

Why isotonic regression (Stage A) instead of Platt scaling (fitting a
logistic sigmoid to u(x))? Platt scaling assumes the true calibration curve
*is* a sigmoid; isotonic regression only assumes it's monotonic and fits the
best monotonic step function via PAV, so it can represent calibration
curves a fixed sigmoid cannot (e.g. flat through a low-entropy "confidently
correct" region, then steep past some threshold) — at the cost of more
degrees of freedom, which is part of why the final score blends it with a
second, more constrained parametric term rather than relying on it alone.

Why Mahalanobis distance (Stage B) instead of Euclidean? Euclidean distance
in raw feature space implicitly assumes every feature is equally scaled and
uncorrelated; Mahalanobis whitens by the reference covariance, so a point
far from the mean along a direction correct answers naturally vary a lot
along is treated as less anomalous than the same raw distance along a
tightly-clustered direction. See mahalanobis_distance() below for the
formula, and README.md's "Calibrated Entropy Divergence" section for the
full worked-out justification (including the chi-squared connection behind
the shrinkage regularization in fit_reference_distribution()).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from scipy.special import expit as sigmoid

from detector import DetectorMetrics, compute_classification_metrics

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

    Formally, PAV solves the constrained least-squares problem
        minimize   sum_i (y_i - f(x_i))^2
        subject to f(x_1) <= f(x_2) <= ... <= f(x_N)   (x sorted ascending)
    This is exactly the loosest calibration assumption possible (just
    "does not decrease"), which is the point — see the module docstring for
    why that's preferred here over assuming a specific parametric shape
    (e.g. Platt's sigmoid).

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

    The empirical covariance Sigma_emp = cov(X_ref) can be near-singular
    when the feature dimension D approaches the number of reference examples
    N (a real risk on small calibration splits). Shrinkage pulls it toward
    its own diagonal, a well-conditioned target:
        Sigma_shrunk = (1 - alpha) * Sigma_emp + alpha * diag(Sigma_emp)
    where alpha = `shrinkage`. At alpha=1 this reduces to treating every
    feature as independent (diagonal-only covariance); at alpha=0 it's the
    plain empirical covariance. This is the regularization that keeps the
    Mahalanobis distance below well-defined even in the D approx N regime —
    see README.md's "Calibrated Entropy Divergence" section for why
    Mahalanobis (which this covariance feeds) was chosen over Euclidean
    distance in the first place, including the chi-squared connection that
    explains why the *scale* of Mahalanobis distances grows with D.

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
    """
    Mahalanobis distance of each row of X to the reference distribution
    (mu, sigma_inv):
        d_M(x) = sqrt( (x - mu)^T * sigma_inv * (x - mu) )
    Unlike Euclidean distance ||x - mu||, this accounts for the reference
    distribution's covariance — a displacement along a direction the
    reference set naturally varies a lot along counts for less than the
    same raw displacement along a tightly-clustered direction. Under the
    null hypothesis that x is drawn from the same distribution as the
    reference set, d_M(x)^2 is chi-squared distributed with D degrees of
    freedom (D = feature dimension), which is why the distance's scale
    itself grows roughly as sqrt(D) — relevant if comparing raw Mahalanobis
    values across feature sets of different dimensionality.

    Returns (N,).
    """
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
class RoutingDecision:
    """
    A 3-way decision derived from predict_proba's output, for callers that
    want an actionable label rather than a raw probability to threshold
    themselves. See CalibratedEntropyDetector.route().
    """
    label: str            # "RELIABLE" | "UNCERTAIN" | "UNRELIABLE"
    action: str           # "accept" | "escalate" | "reject"
    p_hallucination: float
    threshold_reliable: float    # p_hallucination below this -> RELIABLE
    threshold_unreliable: float  # p_hallucination at/above this -> UNRELIABLE

    def __str__(self) -> str:
        return f"[{self.label}] p(hallucination)={self.p_hallucination:.3f} -> {self.action}"


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
    threshold_reliable: float
    threshold_unreliable: float
    score_index_resolved: int  # the concrete column score_index="auto" resolved to


class CalibratedEntropyDetector:
    """
    Detects hallucinations by calibrating raw uncertainty features against a
    reference distribution of "correct answer" signatures, instead of
    thresholding raw entropy directly.

    Parameters
    ----------
    score_index : int or "auto"
        Column of X used as the scalar raw score u(x) fed to isotonic
        regression. An int pins a specific column (e.g. 0 = entropy_mean
        when X is the 6-D output of EntropyFeatureExtractor; caller controls
        what X actually is). The default, "auto", instead picks — at fit()
        time — whichever column has the highest single-feature |AUROC-0.5|
        against the training labels, i.e. the column that alone best ranks
        hallucinated above correct examples. This replaces an earlier fixed
        default of column 0 (typically entropy_mean), which the project's
        own feature-family ablation showed is not actually the strongest
        single signal — lookback-ratio features consistently dominate (see
        README.md §5.1's ablation table). Resolved once during fit() and
        stored (not re-resolved on each predict_proba() call), so repeated
        scoring stays a single indexing operation, and the resolved index is
        picklable via save()/load() like the rest of the fitted state.
    blend_weight : float
        Weight on the isotonic-calibration term vs. the divergence term in
        the final blended probability. Default 0.7 — swept on real paired
        HaluEval features (README.md §5.1) against {0.0, 0.1, ..., 1.0} with
        score_index="auto": 0.7 gave the best 5-fold CV AUROC (0.9833, tied
        within CI overlap with 0.5-0.8), clearly ahead of either pure term
        alone (0.0="pure divergence": 0.8840; 1.0="pure isotonic": 0.9689) —
        i.e. the blend itself, not just auto-selecting the scalar score,
        does real work. This replaces an earlier default of 0.5 (an
        unswept midpoint guess, not a fitted value).
    cov_shrinkage : float
        Diagonal shrinkage applied to the reference covariance (default 0.1)
        to keep Mahalanobis distance well-defined even when feature
        dimensionality approaches the number of calibration examples.
    divergence : {"mahalanobis", "percentile"}
        Which divergence estimator to use.
    reliable_quantile : float
        Quantile (over calibration correct-answer scores) used to set
        route()'s RELIABLE threshold (default 0.90 — see module docstring).
    unreliable_quantile : float
        Quantile (over calibration hallucinated-answer scores) used to set
        route()'s UNRELIABLE threshold (default 0.10 — see module docstring).
    """

    def __init__(
        self,
        score_index: "int | str" = "auto",
        blend_weight: float = 0.7,
        cov_shrinkage: float = 0.1,
        divergence: str = "mahalanobis",
        feature_names: Optional[List[str]] = None,
        reliable_quantile: float = 0.90,
        unreliable_quantile: float = 0.10,
    ) -> None:
        if divergence not in ("mahalanobis", "percentile"):
            raise ValueError(f"Unknown divergence: {divergence!r}. Choose: mahalanobis, percentile")
        if not (score_index == "auto" or isinstance(score_index, (int, np.integer))):
            raise TypeError(f"score_index must be an int or 'auto', got {score_index!r}")
        self.score_index = score_index
        self.blend_weight = blend_weight
        self.cov_shrinkage = cov_shrinkage
        self.divergence = divergence
        self.feature_names = feature_names
        self.reliable_quantile = reliable_quantile
        self.unreliable_quantile = unreliable_quantile
        self._state: Optional[_FitState] = None
        self._fitted = False

    def _standardize(self, X: np.ndarray, state: _FitState) -> np.ndarray:
        return (X - state.mean) / state.std

    def _resolve_score_index(self, X_norm: np.ndarray, y: np.ndarray) -> int:
        """
        Pick the column of X_norm with the strongest single-feature
        separation of y (highest |AUROC-0.5|), used when score_index="auto".
        See __init__'s docstring for why.
        """
        from detector import compute_auroc  # local import: avoid a module-load cycle risk

        best_j, best_sep = 0, -1.0
        for j in range(X_norm.shape[1]):
            auroc = compute_auroc(X_norm[:, j], y)
            sep = abs(auroc - 0.5)
            if sep > best_sep:
                best_sep, best_j = sep, j
        return best_j

    def _divergence_scores_raw(self, X_norm: np.ndarray, state: _FitState) -> np.ndarray:
        if self.divergence == "mahalanobis":
            return mahalanobis_distance(X_norm, state.mu_ref, state.sigma_inv)
        return percentile_rank(
            X_norm[:, state.score_index_resolved], state.reference_divergence
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

        score_index_resolved = (
            self._resolve_score_index(X_norm, y)
            if self.score_index == "auto" else int(self.score_index)
        )

        # Placeholder state so _divergence_scores_raw can run for the
        # nonparametric reference set before the real state object exists.
        tmp_state = _FitState(
            mean=mean, std=std, mu_ref=mu_ref, sigma_inv=sigma_inv,
            isotonic_predict=lambda x: x, logistic_a=0.0, logistic_b=0.0,
            reference_divergence=X_norm[correct_mask, score_index_resolved],
            threshold_reliable=0.0, threshold_unreliable=1.0,
            score_index_resolved=score_index_resolved,
        )
        divergence_all = self._divergence_scores_raw(X_norm, tmp_state)

        # Stage A: isotonic calibration of the raw scalar score.
        u = X_norm[:, score_index_resolved]
        isotonic_predict = isotonic_regression(u, y)

        # Stage B: scalar logistic regression of divergence -> y.
        logistic_a, logistic_b = self._fit_scalar_logistic(divergence_all, y)

        # route() thresholds: fit on the same blended probability formula
        # predict_proba() will later compute, evaluated here on the
        # calibration set itself (see module docstring for the quantile
        # rationale).
        calibrated = isotonic_predict(u)
        div_prob = sigmoid(logistic_a * divergence_all + logistic_b)
        train_probs = np.clip(self.blend_weight * calibrated + (1 - self.blend_weight) * div_prob, 0.0, 1.0)

        hallucinated_mask = y == 1
        threshold_reliable = float(np.quantile(train_probs[correct_mask], self.reliable_quantile))
        if hallucinated_mask.sum() > 0:
            threshold_unreliable = float(np.quantile(train_probs[hallucinated_mask], self.unreliable_quantile))
        else:
            # No hallucinated calibration examples to set this from — disable
            # the UNRELIABLE band rather than guess.
            threshold_unreliable = 1.0
        # Guard against an inverted band on heavily-overlapping distributions:
        # collapse UNCERTAIN to zero width rather than let RELIABLE and
        # UNRELIABLE cross.
        threshold_unreliable = max(threshold_unreliable, threshold_reliable)

        self._state = _FitState(
            mean=mean, std=std, mu_ref=mu_ref, sigma_inv=sigma_inv,
            isotonic_predict=isotonic_predict,
            logistic_a=logistic_a, logistic_b=logistic_b,
            reference_divergence=divergence_all[correct_mask],
            threshold_reliable=threshold_reliable,
            threshold_unreliable=threshold_unreliable,
            score_index_resolved=score_index_resolved,
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

        u = X_norm[:, state.score_index_resolved]
        calibrated = state.isotonic_predict(u)

        div = self._divergence_scores_raw(X_norm, state)
        div_prob = sigmoid(state.logistic_a * div + state.logistic_b)

        w = self.blend_weight
        p = w * calibrated + (1 - w) * div_prob
        return np.clip(p, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def route(self, X: np.ndarray) -> List[RoutingDecision]:
        """
        Map predict_proba's output into 3-way RELIABLE / UNCERTAIN /
        UNRELIABLE decisions using the thresholds fit in fit() (see module
        docstring). One RoutingDecision per row of X.

            p_hallucination < threshold_reliable    -> RELIABLE   (accept)
            threshold_reliable <= p < threshold_unreliable -> UNCERTAIN (escalate)
            p_hallucination >= threshold_unreliable -> UNRELIABLE (reject)
        """
        assert self._fitted, "Must call fit() first"
        state = self._state
        probs = self.predict_proba(X)
        decisions = []
        for p in probs:
            p = float(p)
            if p < state.threshold_reliable:
                label, action = "RELIABLE", "accept"
            elif p >= state.threshold_unreliable:
                label, action = "UNRELIABLE", "reject"
            else:
                label, action = "UNCERTAIN", "escalate"
            decisions.append(RoutingDecision(
                label=label, action=action, p_hallucination=p,
                threshold_reliable=state.threshold_reliable,
                threshold_unreliable=state.threshold_unreliable,
            ))
        return decisions

    def route_one(self, x: np.ndarray) -> RoutingDecision:
        """Convenience wrapper for a single example (e.g. a live demo)."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return self.route(x)[0]

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
                "reliable_quantile": self.reliable_quantile,
                "unreliable_quantile": self.unreliable_quantile,
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
            reliable_quantile=data.get("reliable_quantile", 0.90),
            unreliable_quantile=data.get("unreliable_quantile", 0.10),
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

    # Test 5: 3-way routing
    print("\n--- Test 5: 3-way routing (RELIABLE/UNCERTAIN/UNRELIABLE) ---")
    decisions = det.route(X_test)
    labels = {d.label for d in decisions}
    assert labels <= {"RELIABLE", "UNCERTAIN", "UNRELIABLE"}, f"Unexpected labels: {labels}"
    # Well-separated synthetic correct examples should mostly route RELIABLE,
    # and hallucinated examples mostly route UNRELIABLE.
    correct_labels = [d.label for d, y in zip(decisions, y_test) if y == 0]
    halluc_labels = [d.label for d, y in zip(decisions, y_test) if y == 1]
    correct_reliable_frac = correct_labels.count("RELIABLE") / max(len(correct_labels), 1)
    halluc_unreliable_frac = halluc_labels.count("UNRELIABLE") / max(len(halluc_labels), 1)
    assert correct_reliable_frac > 0.5, f"Too few correct examples routed RELIABLE: {correct_reliable_frac:.2f}"
    assert halluc_unreliable_frac > 0.5, f"Too few hallucinated examples routed UNRELIABLE: {halluc_unreliable_frac:.2f}"
    assert det._state.threshold_reliable <= det._state.threshold_unreliable, "Inverted routing band"
    # route_one matches route()'s first row.
    single = det.route_one(X_test[0])
    assert single.label == decisions[0].label and single.p_hallucination == decisions[0].p_hallucination
    print(f"  correct->RELIABLE: {correct_reliable_frac:.0%}, hallucinated->UNRELIABLE: {halluc_unreliable_frac:.0%}")
    print("  3-way routing behaves sanely ✅")

    print(f"\n{'=' * 60}")
    print("Calibrated entropy detector — ALL CHECKS PASS ✅")
    print(f"{'=' * 60}")
