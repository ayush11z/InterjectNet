"""Shared paths and settings for the InterjectNet pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Required for pyannote.audio to download gated diarization models from the Hub.
HF_TOKEN = os.environ.get("HF_TOKEN")

# faster-whisper model size: tiny/base/small/medium/large-v3.
# "small" is a good speed/accuracy tradeoff for a live demo on CPU.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
