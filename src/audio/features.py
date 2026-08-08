"""Low-level acoustic features (pitch, energy) for a slice of audio."""

from dataclasses import dataclass

import librosa
import numpy as np

# pyin's voiced-pitch search range, tuned for speech (not music).
_FMIN = librosa.note_to_hz("C2")  # ~65 Hz
_FMAX = librosa.note_to_hz("C6")  # ~1047 Hz


@dataclass
class AcousticFeatures:
    pitch_mean_hz: float | None  # None if no voiced frames were found
    pitch_slope: float | None  # Hz/sec trend over the segment; negative = falling pitch
    energy_rms: float


def extract_features(y: np.ndarray, sr: int, start: float, end: float) -> AcousticFeatures:
    """Extract pitch and energy for the audio between `start` and `end` seconds."""
    clip = y[int(start * sr) : int(end * sr)]

    if len(clip) < sr * 0.05:  # too short (<50ms) to say anything meaningful
        return AcousticFeatures(pitch_mean_hz=None, pitch_slope=None, energy_rms=0.0)

    energy_rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))

    f0, voiced_flag, _voiced_prob = librosa.pyin(clip, fmin=_FMIN, fmax=_FMAX, sr=sr)
    voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])

    if len(voiced_f0) < 2:
        return AcousticFeatures(pitch_mean_hz=None, pitch_slope=None, energy_rms=energy_rms)

    pitch_mean_hz = float(np.mean(voiced_f0))
    # Slope of a linear fit over the voiced frames, in Hz/sec -- a rough proxy
    # for whether the speaker's intonation is falling (statement-like, "done
    # talking") or flat/rising (trailing off, likely to continue).
    voiced_times = np.flatnonzero(voiced_flag) * (len(clip) / sr) / len(voiced_flag)
    pitch_slope = float(np.polyfit(voiced_times, voiced_f0, 1)[0])

    return AcousticFeatures(pitch_mean_hz=pitch_mean_hz, pitch_slope=pitch_slope, energy_rms=energy_rms)
