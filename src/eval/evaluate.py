"""Evaluate the interjection scorer against human-labeled ground truth:
precision at a score threshold, compared to a random-selection baseline.

Requires, per clip, both data/processed/<stem>.segments.json (from the
audio pipeline) and data/processed/<stem>.labels.json (from
src/eval/labeling.py).

Run: python -m src.eval.evaluate <clip_stem> [<clip_stem> ...]
"""

import json
import random
import sys

from src.audio.pipeline import ProcessedSegment
from src.eval.labeling import labels_path, segments_path, window_key
from src.scoring.interjection import ScoredPauseWindow, score_conversation

THRESHOLD = 0.6
RANDOM_TRIALS = 2000

LabeledWindow = tuple[ScoredPauseWindow, bool]


def load_labeled_windows(stem: str) -> list[LabeledWindow]:
    segments = [ProcessedSegment(**s) for s in json.loads(segments_path(stem).read_text())]
    windows = score_conversation(segments)
    labels = json.loads(labels_path(stem).read_text()) if labels_path(stem).exists() else {}

    return [(w, labels[window_key(w.start, w.end)]) for w in windows if window_key(w.start, w.end) in labels]


def precision_at_threshold(labeled: list[LabeledWindow], threshold: float) -> tuple[float | None, int]:
    """Precision of "windows the model says to interject on" -- of the ones
    scored >= threshold, what fraction were actually labeled good?"""
    selected = [good for w, good in labeled if w.interject_score >= threshold]
    if not selected:
        return None, 0
    return sum(selected) / len(selected), len(selected)


def random_baseline(labeled: list[LabeledWindow], k: int, trials: int = RANDOM_TRIALS) -> float | None:
    """Expected precision of picking k windows uniformly at random instead of
    using the model, estimated by Monte Carlo so it reads as an actual
    comparison rather than a plugged-in formula. (In expectation this equals
    the base rate of "good" windows -- the simulation just makes that
    concrete and gives a distribution to sanity-check against.)"""
    if k == 0 or not labeled:
        return None
    outcomes = [good for _, good in labeled]
    trial_precisions = [sum(random.sample(outcomes, k)) / k for _ in range(trials)]
    return sum(trial_precisions) / len(trial_precisions)


def report(label: str, labeled: list[LabeledWindow]) -> None:
    if not labeled:
        print(f"--- {label}: no labeled windows ---\n")
        return

    base_rate = sum(good for _, good in labeled) / len(labeled)
    model_p, k = precision_at_threshold(labeled, THRESHOLD)
    rand_p = random_baseline(labeled, k)

    print(f"--- {label} ---")
    print(f"  labeled windows: {len(labeled)}  (base rate of 'good': {base_rate:.0%})")
    if model_p is None:
        print(f"  no windows scored >= {THRESHOLD}; nothing to evaluate at this threshold")
    else:
        lift = model_p - rand_p if rand_p is not None else None
        print(f"  model precision@{THRESHOLD}: {model_p:.0%}  (n={k})")
        print(f"  random baseline (same n, {RANDOM_TRIALS} random trials): {rand_p:.0%}")
        if lift is not None:
            print(f"  lift over random: {lift:+.0%}")
    print()


def main():
    stems = sys.argv[1:]
    if not stems:
        print("Usage: python -m src.eval.evaluate <clip_stem> [<clip_stem> ...]", file=sys.stderr)
        sys.exit(1)

    all_labeled: list[LabeledWindow] = []
    for stem in stems:
        if not labels_path(stem).exists():
            print(f"Skipping '{stem}': no labels yet (run `python -m src.eval.labeling {stem}` first).\n")
            continue
        labeled = load_labeled_windows(stem)
        all_labeled.extend(labeled)
        report(stem, labeled)

    if len(stems) > 1 and all_labeled:
        report(f"combined across {len(stems)} clips", all_labeled)


if __name__ == "__main__":
    main()
