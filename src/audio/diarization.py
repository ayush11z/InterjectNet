"""Speaker diarization: who spoke when."""

from dataclasses import dataclass

from pyannote.audio import Pipeline

from src.utils.config import DIARIZATION_MODEL, HF_TOKEN
from src.utils.hf_compat import patch_pyannote_hf_compat

patch_pyannote_hf_compat()

_pipeline: Pipeline | None = None


@dataclass
class SpeakerTurn:
    speaker: str
    start: float  # seconds
    end: float  # seconds


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
    return _pipeline


def diarize(audio_path: str) -> list[SpeakerTurn]:
    """Run diarization on an audio file, returning speaker turns in time order."""
    pipeline = _get_pipeline()
    annotation = pipeline(audio_path)

    turns = [
        SpeakerTurn(speaker=speaker, start=segment.start, end=segment.end)
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start)
    return turns
