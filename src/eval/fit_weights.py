"""Fit the scorer's weights from labeled data instead of hand-picking them.

Solves for the non-negative weights (summing to 1, same interpretable
structure as src/scoring/interjection.py's weighted sum) over
[pause_score, completion_score, topic_stability_score] that best predict
your labels -- calibrated instead of eyeballed.

With this few labeled examples, the point isn't to produce a final answer
-- it's to see which sub-scores are actually pulling in the right
direction. Leave-one-out cross-validation is used throughout so the
reported precision isn't just in-sample overfitting on 29 points.

Run: python -m src.eval.fit_weights <clip_stem> [<clip_stem> ...]
"""

import random
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

from src.eval.evaluate import RANDOM_TRIALS, THRESHOLD, load_labeled_windows

FEATURE_NAMES = ["pause", "completion", "topic"]


def precision_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float | None, int]:
    selected = labels[scores >= threshold]
    if len(selected) == 0:
        return None, 0
    return float(selected.mean()), len(selected)


def random_baseline(labels: np.ndarray, k: int, trials: int = RANDOM_TRIALS) -> float | None:
    if k == 0:
        return None
    pool = labels.tolist()
    return sum(sum(random.sample(pool, k)) / k for _ in range(trials)) / trials


def main():
    stems = sys.argv[1:]
    if not stems:
        print("Usage: python -m src.eval.fit_weights <clip_stem> [<clip_stem> ...]", file=sys.stderr)
        sys.exit(1)

    labeled = []
    for stem in stems:
        labeled.extend(load_labeled_windows(stem))

    if len(labeled) < 5:
        print(f"Only {len(labeled)} labeled windows -- too few to fit anything meaningful.", file=sys.stderr)
        sys.exit(1)

    X = np.array([[w.pause_score, w.completion_score, w.topic_stability_score] for w, _ in labeled])
    y = np.array([int(good) for _, good in labeled])

    print(f"Fitting on {len(labeled)} labeled windows across {len(stems)} clip(s).")
    print(f"Base rate of 'good': {y.mean():.0%}\n")

    full_fit = LinearRegression(positive=True, fit_intercept=False).fit(X, y)
    weights = full_fit.coef_
    total = weights.sum()
    normalized = weights / total if total > 0 else weights

    print("Fitted weights (non-negative, normalized to sum to 1):")
    for name, w in zip(FEATURE_NAMES, normalized):
        print(f"  {name:>10}: {w:.3f}")
    if total == 0:
        print("  (all three coefficients came back zero -- none of the sub-scores predict your labels linearly)")
    print(f"\nCurrent hand-picked weights: pause=0.40, completion=0.35, topic=0.25")
    print()

    # Leave-one-out: refit on n-1 each time and predict the held-out point,
    # so this doesn't just report in-sample overfitting.
    loo = LeaveOneOut()
    loo_scores = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        model = LinearRegression(positive=True, fit_intercept=False).fit(X[train_idx], y[train_idx])
        loo_scores[test_idx] = model.predict(X[test_idx])

    model_p, k = precision_at(loo_scores, y, THRESHOLD)
    rand_p = random_baseline(y, k)
    print(f"Leave-one-out precision@{THRESHOLD} with fitted weights: ", end="")
    if model_p is None:
        print("no held-out windows scored >= threshold")
    else:
        lift = model_p - rand_p if rand_p is not None else None
        print(f"{model_p:.0%} (n={k}) vs random baseline {rand_p:.0%}" + (f" ({lift:+.0%})" if lift is not None else ""))

    print("\nCaveat: n is small -- treat this as a directional signal, not a final weight set. More labels will sharpen it.")


if __name__ == "__main__":
    main()
