"""Interactive CLI to label pause windows as good/bad interjection moments,
building the ground truth that evaluate.py checks the scorer against.

Labeling is deliberately kept separate from scoring: it only shows you the
speaker/text context for each window, never the model's own score, so your
judgment isn't anchored by what the model already thinks. When ffplay is
available it also plays the actual audio around each pause (a few seconds
of lead-in through the resumption), since most of what the model scores --
pause length, pitch, energy -- isn't something text can tell you either.

Run: python -m src.eval.labeling <clip_stem>
(clip_stem = the filename without extension, matching
data/processed/<clip_stem>.segments.json -- run the pipeline or the
Streamlit demo on the clip first if that file doesn't exist yet. The
matching audio file is expected at data/raw/<clip_stem>.<ext>.)
"""

import json
import shutil
import subprocess
import sys

from src.audio.pipeline import ProcessedSegment
from src.scoring.interjection import score_conversation
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")
LEAD_IN_SECONDS = 4.0  # how much audio before the pause to play, for context
TRAIL_SECONDS = 1.5  # how much to play after the pause resumes


def segments_path(stem: str):
    return DATA_PROCESSED_DIR / f"{stem}.segments.json"


def labels_path(stem: str):
    return DATA_PROCESSED_DIR / f"{stem}.labels.json"


def window_key(start: float, end: float) -> str:
    return f"{start}|{end}"


def find_audio_path(stem: str):
    for ext in AUDIO_EXTENSIONS:
        candidate = DATA_RAW_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def play_snippet(audio_path, start: float, end: float) -> None:
    clip_start = max(0.0, start - LEAD_IN_SECONDS)
    duration = (end - clip_start) + TRAIL_SECONDS
    subprocess.run(
        [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-ss", str(clip_start), "-t", str(duration), str(audio_path),
        ],
        check=False,
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.eval.labeling <clip_stem>", file=sys.stderr)
        sys.exit(1)

    stem = sys.argv[1]
    if not segments_path(stem).exists():
        print(f"No cached segments for '{stem}' -- run the pipeline on it first.", file=sys.stderr)
        sys.exit(1)

    audio_path = find_audio_path(stem)
    can_play = audio_path is not None and shutil.which("ffplay") is not None
    if audio_path is None:
        print(f"Note: no audio file found at data/raw/{stem}.<ext> -- labeling from transcript text only.")
    elif not can_play:
        print("Note: ffplay not found on PATH -- labeling from transcript text only (install ffmpeg for audio playback).")

    segments = [ProcessedSegment(**s) for s in json.loads(segments_path(stem).read_text())]
    windows = score_conversation(segments)

    lp = labels_path(stem)
    labels = json.loads(lp.read_text()) if lp.exists() else {}
    already = len(labels)

    print(f"Labeling {len(windows)} pause windows for '{stem}' ({already} already labeled).")
    prompt = "Good interject moment? [y/n/s/q]: " if not can_play else "Good interject moment? [y/n/s/q, r to replay]: "
    print(f"Is this a good moment for someone else to jump in? {prompt}\n")

    for w in windows:
        key = window_key(w.start, w.end)
        if key in labels:
            continue

        print(f"[{w.start:.1f}s -> {w.end:.1f}s, {w.duration:.2f}s pause]")
        print(f"  {w.prev_speaker}: ...{w.prev_text[-120:]}")
        print(f"  -> {w.next_speaker}: {w.next_text[:120]}")

        if can_play:
            play_snippet(audio_path, w.start, w.end)

        while True:
            ans = input(f"  {prompt}").strip().lower()
            if can_play and ans == "r":
                play_snippet(audio_path, w.start, w.end)
                continue
            if ans in ("y", "n", "s", "q"):
                break

        if ans == "q":
            break
        if ans != "s":
            labels[key] = ans == "y"
        print()

    lp.write_text(json.dumps(labels, indent=2))
    print(f"\nSaved {len(labels)} labels ({len(labels) - already} new) to {lp}")


if __name__ == "__main__":
    main()
