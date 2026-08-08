"""Transcription via faster-whisper."""

from dataclasses import dataclass

from faster_whisper import WhisperModel

from src.utils.config import WHISPER_MODEL_SIZE

_model: WhisperModel | None = None


@dataclass
class TranscribedWord:
    word: str
    start: float  # seconds
    end: float  # seconds


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # CPU + int8 keeps this usable on a laptop for a live demo.
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> list[TranscribedWord]:
    """Transcribe an audio file, returning word-level timestamps."""
    model = _get_model()
    segments, _info = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for segment in segments:
        for word in segment.words:
            words.append(TranscribedWord(word=word.word.strip(), start=word.start, end=word.end))
    return words
