"""Evaluate the interjection scorer against the public Krisp-AI/
turn-taking-test-v1 benchmark: precision/recall/F1 on predicting genuine
turn-transition moments ("shift") vs. within-turn pauses ("hold"), plus a
random-selection baseline and a threshold-free correlation/AUC check.

Run: python -m src.eval.evaluate_krisp [--limit N] [--threshold 0.6]
"""

import argparse
import random
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.eval.krisp_adapter import load_dataframe, score_row
from src.utils.config import HF_TOKEN

DEFAULT_THRESHOLD = 0.6
RANDOM_TRIALS = 2000


def random_baseline_precision(labels: np.ndarray, k: int, trials: int = RANDOM_TRIALS) -> float | None:
    if k == 0:
        return None
    pool = labels.tolist()
    return sum(sum(random.sample(pool, k)) / k for _ in range(trials)) / trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N rows (for a quick check)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    print("Loading Krisp-AI/turn-taking-test-v1...", file=sys.stderr)
    df = load_dataframe(HF_TOKEN)
    if args.limit:
        # Random, not a prefix slice -- rows are grouped by speaker/session,
        # so the first N rows aren't a representative sample of the labels.
        df = df.sample(n=min(args.limit, len(df)), random_state=0)
    n = len(df)
    print(f"Scoring {n} examples...", file=sys.stderr)

    scores = np.zeros(n)
    labels = np.zeros(n, dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        ex = score_row(row)
        scores[i] = ex.interject_score
        labels[i] = ex.label
        if (i + 1) % 250 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}", file=sys.stderr)

    predicted = scores >= args.threshold
    base_rate = labels.mean()

    print(f"\n=== Krisp-AI/turn-taking-test-v1 (n={n}) ===")
    print(f"base rate of 'shift' (real turn transition): {base_rate:.1%}")
    print(f"threshold: {args.threshold}, predicted positive: {int(predicted.sum())}")

    if predicted.sum() == 0 or predicted.sum() == n:
        print("degenerate: model predicts the same class for everything at this threshold")
    else:
        precision = precision_score(labels, predicted, zero_division=0)
        recall = recall_score(labels, predicted, zero_division=0)
        f1 = f1_score(labels, predicted, zero_division=0)
        accuracy = (predicted == labels).mean()
        print(f"  precision: {precision:.1%}")
        print(f"  recall:    {recall:.1%}")
        print(f"  f1:        {f1:.1%}")
        print(f"  accuracy:  {accuracy:.1%}")

        k = int(predicted.sum())
        rand_p = random_baseline_precision(labels, k)
        if rand_p is not None:
            print(f"\n  random baseline precision (same n positive, {RANDOM_TRIALS} trials): {rand_p:.1%}")
            print(f"  lift over random: {precision - rand_p:+.1%}")

    auc = roc_auc_score(labels, scores)
    print(f"\nROC AUC: {auc:.3f}  (0.5 = random, 1.0 = perfect, <0.5 = predicting backwards)")

    np.save("/tmp/krisp_scores.npy", scores)
    np.save("/tmp/krisp_labels.npy", labels)


if __name__ == "__main__":
    main()
