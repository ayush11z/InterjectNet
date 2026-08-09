"""Recompute acoustic features (pitch/energy) for already-cached segments
without re-running diarization/transcription -- the slow parts. Useful
after tweaking src/audio/features.py, since speaker/text/pause timing
don't change, only the pitch/energy fields.

Run: python -m src.audio.refresh_features <clip_stem>
"""

import json
import sys
from dataclasses import asdict

import librosa

from src.audio.features import extract_features
from src.audio.pipeline import SAMPLE_RATE, ProcessedSegment
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")


def find_audio_path(stem: str):
    for ext in AUDIO_EXTENSIONS:
        candidate = DATA_RAW_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.audio.refresh_features <clip_stem>", file=sys.stderr)
        sys.exit(1)

    stem = sys.argv[1]
    segments_path = DATA_PROCESSED_DIR / f"{stem}.segments.json"
    if not segments_path.exists():
        print(f"No cached segments for '{stem}'.", file=sys.stderr)
        sys.exit(1)

    audio_path = find_audio_path(stem)
    if audio_path is None:
        print(f"No audio file found at data/raw/{stem}.<ext>.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {audio_path.name}...", file=sys.stderr)
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    segments = [ProcessedSegment(**s) for s in json.loads(segments_path.read_text())]

    for seg in segments:
        feats = extract_features(y, sr, seg.start, seg.end)
        seg.pitch_mean_hz = round(feats.pitch_mean_hz, 1) if feats.pitch_mean_hz else None
        seg.pitch_slope = round(feats.pitch_slope, 1) if feats.pitch_slope else None
        seg.energy_rms = round(feats.energy_rms, 4)
        seg.energy_slope = round(feats.energy_slope, 5) if feats.energy_slope else None

    segments_path.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    print(f"Refreshed features for {len(segments)} segments -> {segments_path}")


if __name__ == "__main__":
    main()
