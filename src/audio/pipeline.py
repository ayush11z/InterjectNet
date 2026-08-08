"""End-to-end audio pipeline: file in, timestamped speaker+text+features stream out."""

import json
import sys
from dataclasses import asdict, dataclass

import librosa

from src.audio.diarization import diarize
from src.audio.features import AcousticFeatures, extract_features
from src.audio.segments import Utterance, build_utterances
from src.audio.transcription import transcribe

SAMPLE_RATE = 16000  # matches what pyannote/whisper expect internally


@dataclass
class ProcessedSegment:
    speaker: str
    start: float
    end: float
    text: str
    pause_before: float
    pause_after: float
    pitch_mean_hz: float | None
    pitch_slope: float | None
    energy_rms: float
    speaking_rate_wps: float  # words per second


def process_audio(audio_path: str) -> list[ProcessedSegment]:
    """Run diarization + transcription + feature extraction on an audio file."""
    print(f"[1/4] Loading audio: {audio_path}", file=sys.stderr)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    print("[2/4] Diarizing speakers...", file=sys.stderr)
    turns = diarize(audio_path)

    print("[3/4] Transcribing...", file=sys.stderr)
    words = transcribe(audio_path)

    print("[4/4] Merging + extracting features...", file=sys.stderr)
    utterances = build_utterances(turns, words)

    segments = [_to_segment(u, y, sr) for u in utterances]
    print(f"Done: {len(segments)} segments.", file=sys.stderr)
    return segments


def _to_segment(u: Utterance, y, sr: int) -> ProcessedSegment:
    feats: AcousticFeatures = extract_features(y, sr, u.start, u.end)
    duration = u.end - u.start
    word_count = len(u.words)
    speaking_rate = word_count / duration if duration > 0 else 0.0

    return ProcessedSegment(
        speaker=u.speaker,
        start=round(u.start, 3),
        end=round(u.end, 3),
        text=u.text,
        pause_before=round(u.pause_before, 3),
        pause_after=round(u.pause_after, 3),
        pitch_mean_hz=round(feats.pitch_mean_hz, 1) if feats.pitch_mean_hz else None,
        pitch_slope=round(feats.pitch_slope, 1) if feats.pitch_slope else None,
        energy_rms=round(feats.energy_rms, 4),
        speaking_rate_wps=round(speaking_rate, 2),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.audio.pipeline <audio_path>", file=sys.stderr)
        sys.exit(1)

    result = process_audio(sys.argv[1])
    print(json.dumps([asdict(s) for s in result], indent=2))
