"""Adapter for the Krisp-AI/turn-taking-test-v1 benchmark on HuggingFace.

Each row is a single-speaker audio clip ending in silence, labeled "hold"
(the same speaker continues after the pause -- not a real turn transition)
or "shift" (a different speaker takes over -- a genuine Transition
Relevance Place). That maps directly onto our question: shift = a real
opening for someone else to speak, hold = not.

This benchmark gives isolated clips with no prior conversational history,
so there's nothing to compute topic_stability_score against. The adapter
scores each example with topic fixed at neutral (0.5) -- same fallback
src/scoring/interjection.py already uses when a real conversation has no
prior context yet -- so this evaluates pause_score and completion_score
specifically, using the *exact* functions and weights from the production
scorer (not a reimplementation), so a result here says something true
about the actual model.

Deliberately kept separate from src/audio/pipeline.py: these clips are
already pre-segmented at the silence point, so there's no diarization or
transcription step to run here, just feature extraction.
"""

import io
from dataclasses import dataclass

import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

from src.audio.features import extract_features
from src.scoring import interjection

DATASET_REPO = "Krisp-AI/turn-taking-test-v1"
DATASET_FILE = "data/test.parquet"
NEUTRAL_TOPIC_SCORE = 0.5


@dataclass
class KrispExample:
    filename: str
    interject_score: float
    pause_score: float
    completion_score: float
    label: bool  # True = "shift" (real turn transition), False = "hold"


def load_dataframe(hf_token: str | None = None) -> pd.DataFrame:
    path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset", token=hf_token)
    return pd.read_parquet(path)


def score_row(row: pd.Series) -> KrispExample:
    y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    speech_end = row["duration"] - row["last_silence_duration"]

    feats = extract_features(y, sr, 0.0, speech_end)
    pause_score = interjection._pause_score(row["last_silence_duration"])
    completion_score = interjection._completion_score(feats.pitch_slope, feats.energy_slope)

    score = (
        interjection.PAUSE_WEIGHT * pause_score
        + interjection.COMPLETION_WEIGHT * completion_score
        + interjection.TOPIC_WEIGHT * NEUTRAL_TOPIC_SCORE
    )

    return KrispExample(
        filename=row["filename"],
        interject_score=score,
        pause_score=pause_score,
        completion_score=completion_score,
        label=row["label"] == "shift",
    )
