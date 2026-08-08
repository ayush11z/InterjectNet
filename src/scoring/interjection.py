"""Rule-based interjection scoring.

Given the segment stream from src/audio/pipeline.py, score every real pause
window between utterances on how good a moment it is for another speaker to
jump in. Deliberately not ML: a small weighted sum over interpretable
prosodic + semantic signals, so it's explainable and the weights can be
nudged live while watching a demo, instead of retraining anything.

Three signals per window, each mapped to [0, 1]:
  - pause_score:    how long the silence is (longer -> more clearly "open")
  - completion_score: whether the last speaker's pitch/energy trended down
                       (falling = sounds finished) or stayed flat/rose
                       (sounds like they're mid-thought / trailing off)
  - topic_stability_score: how semantically close the last utterance is to
                       the recent conversation (a stable topic is a safer
                       moment to add something relevant; an abrupt topic
                       shift is a worse moment to pile on)
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.audio.pipeline import ProcessedSegment

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None

# Weights must sum to 1. Starting point tuned by eyeballing the sample
# clips -- these are meant to be nudged, not treated as final.
PAUSE_WEIGHT = 0.40
COMPLETION_WEIGHT = 0.35
TOPIC_WEIGHT = 0.25

PAUSE_TIME_CONSTANT = 0.7  # seconds; how fast pause_score saturates toward 1
PITCH_SLOPE_SCALE = 80.0  # Hz/sec; how sharply a falling pitch pushes completion_score up
ENERGY_SLOPE_SCALE = 0.01  # amplitude/sec; same idea for the loudness envelope
TOPIC_CONTEXT_WINDOW = 4  # how many prior utterances count as "recent topic"


@dataclass
class ScoredPauseWindow:
    start: float
    end: float
    duration: float
    prev_speaker: str
    prev_text: str
    next_speaker: str
    next_text: str
    pause_score: float
    completion_score: float
    topic_stability_score: float
    interject_score: float


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedder


def _pause_score(duration: float) -> float:
    """Longer pauses look more like an opening; saturates so a 3s silence
    doesn't score much higher than a 2s one -- both are clearly "open"."""
    return float(1 - np.exp(-duration / PAUSE_TIME_CONSTANT))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def _completion_score(pitch_slope: float | None, energy_slope: float | None) -> float:
    """Falling pitch and/or a decaying loudness envelope read as a finished
    statement; flat/rising read as mid-thought. Missing signals fall back to
    neutral (0.5) rather than dragging the average down."""
    components = []
    if pitch_slope is not None:
        components.append(_sigmoid(-pitch_slope / PITCH_SLOPE_SCALE))
    if energy_slope is not None:
        components.append(_sigmoid(-energy_slope / ENERGY_SLOPE_SCALE))

    if not components:
        return 0.5
    return float(np.mean(components))


def _topic_stability_scores(texts: list[str]) -> list[float]:
    """Cosine similarity between each utterance and the mean embedding of
    the few utterances before it. High similarity -> the topic hasn't moved
    -> safer moment for an agent to say something relevant. `texts` must
    already be filtered to non-empty, spoken utterances in order."""
    embeddings = _get_embedder().encode(texts, normalize_embeddings=True)

    scores = []
    for i in range(len(embeddings)):
        context = embeddings[max(0, i - TOPIC_CONTEXT_WINDOW) : i]
        if len(context) == 0:
            scores.append(0.5)  # no prior spoken context yet
            continue
        context_vec = np.mean(context, axis=0)
        context_vec /= np.linalg.norm(context_vec) + 1e-8
        cosine_sim = float(np.dot(embeddings[i], context_vec))
        scores.append(max(0.0, min(1.0, (cosine_sim + 1) / 2)))  # [-1, 1] -> [0, 1]
    return scores


def score_conversation(segments: list[ProcessedSegment]) -> list[ScoredPauseWindow]:
    """Score every real pause window between consecutive *spoken* utterances.

    Diarization interleaves plenty of empty-text segments between real
    utterances (crosstalk bleed, breaths, silence picked up as "speech" with
    nothing transcribed). Those have no completion/topic signal worth
    scoring, and worse, they're often positioned to overlap the speech
    around them -- so treating every raw segment-to-segment gap as a
    candidate window ends up discarding almost all genuine pauses along
    with the noise. Instead, we first collapse to the sequence of spoken
    utterances only, then measure pauses between *those*. A pause here can
    still legitimately come out <= 0 (the next spoken utterance started
    before this one finished, i.e. talked over it) and is skipped."""
    spoken = [s for s in segments if s.text.strip()]
    topic_scores = _topic_stability_scores([s.text for s in spoken])

    windows = []
    for i in range(len(spoken) - 1):
        prev, nxt = spoken[i], spoken[i + 1]
        duration = nxt.start - prev.end
        if duration <= 0:
            continue

        pause_score = _pause_score(duration)
        completion_score = _completion_score(prev.pitch_slope, prev.energy_slope)
        topic_score = topic_scores[i]

        interject_score = (
            PAUSE_WEIGHT * pause_score + COMPLETION_WEIGHT * completion_score + TOPIC_WEIGHT * topic_score
        )

        windows.append(
            ScoredPauseWindow(
                start=round(prev.end, 3),
                end=round(nxt.start, 3),
                duration=round(duration, 3),
                prev_speaker=prev.speaker,
                prev_text=prev.text,
                next_speaker=nxt.speaker,
                next_text=nxt.text,
                pause_score=round(pause_score, 3),
                completion_score=round(completion_score, 3),
                topic_stability_score=round(topic_score, 3),
                interject_score=round(interject_score, 3),
            )
        )
    return windows


if __name__ == "__main__":
    # Scores a pipeline JSON output (from `python -m src.audio.pipeline`)
    # without re-running diarization/transcription -- fast to iterate on
    # weights against clips already processed once.
    import json
    import sys
    from dataclasses import asdict

    if len(sys.argv) != 2:
        print("Usage: python -m src.scoring.interjection <pipeline_output.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        raw_segments = json.load(f)
    segments = [ProcessedSegment(**s) for s in raw_segments]

    scored = score_conversation(segments)
    print(json.dumps([asdict(w) for w in scored], indent=2))

