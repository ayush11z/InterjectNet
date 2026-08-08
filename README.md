# InterjectNet

Predicts **when** an AI agent should speak up in a live multi-speaker conversation — not *what* it would say, just the timing/interjection decision. Models natural turn-taking cues: pauses, completed vs. trailing-off intonation, and topic stability.

## How it works (pipeline)

```
audio  ->  diarization (pyannote.audio)      -> speaker-labeled segments
       ->  transcription (faster-whisper)    -> text per segment
       ->  feature extraction (librosa)      -> pitch, energy, pause duration, speaking rate
       ->  interjection scoring (rule-based) -> 0-1 "interject now" score per pause window
       ->  demo UI (Streamlit)               -> live visualization of scores
```

## Project structure

```
src/
  audio/      # diarization, transcription, feature extraction
  scoring/    # interjection scoring logic
  utils/      # shared helpers (audio I/O, config)
demo/         # Streamlit app
data/
  raw/        # input audio clips (gitignored)
  processed/  # cached pipeline outputs (gitignored)
models/       # local model weights/caches (gitignored)
tests/        # unit tests + evaluation harness
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

pyannote.audio requires a Hugging Face token (accept the model license at
`pyannote/speaker-diarization-3.1` and set `HF_TOKEN` in a `.env` file).

## Status

- [x] Stage 1 — project scaffold
- [ ] Stage 2 — audio pipeline
- [ ] Stage 3 — interjection scoring
- [ ] Stage 4 — live demo
- [ ] Stage 5 — evaluation
