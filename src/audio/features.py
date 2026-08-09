"""Low-level acoustic features (pitch, energy) for a slice of audio."""

from dataclasses import dataclass

import librosa
import numpy as np

# pyin's voiced-pitch search range, tuned for speech (not music).
_FMIN = librosa.note_to_hz("C2")  # ~65 Hz
_FMAX = librosa.note_to_hz("C6")  # ~1047 Hz

# How much of the *end* of an utterance to look at for completion cues
# (falling pitch/energy). A linear fit over an entire multi-sentence,
# minutes-long monologue just averages away any real end-of-utterance
# intonation contour -- what actually signals "I'm done" is how the last
# second or two sounds, not the whole utterance's trend.
TAIL_SECONDS = 2.0


@dataclass
class AcousticFeatures:
    pitch_mean_hz: float | None  # None if no voiced frames were found, mean over the whole segment
    pitch_slope: float | None  # Hz/sec trend over the last TAIL_SECONDS; negative = falling pitch
    energy_rms: float  # mean over the whole segment
    energy_slope: float | None  # amplitude/sec trend over the last TAIL_SECONDS; negative = trailing off


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    return float(np.polyfit(times, values, 1)[0])


def extract_features(y: np.ndarray, sr: int, start: float, end: float) -> AcousticFeatures:
    """Extract pitch and energy for the audio between `start` and `end` seconds."""
    clip = y[int(start * sr) : int(end * sr)]

    if len(clip) < sr * 0.05:  # too short (<50ms) to say anything meaningful
        return AcousticFeatures(pitch_mean_hz=None, pitch_slope=None, energy_rms=0.0, energy_slope=None)

    energy_rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
    clip_duration = len(clip) / sr
    tail_start = max(0.0, clip_duration - TAIL_SECONDS)

    rms = librosa.feature.rms(y=clip)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    tail_mask = rms_times >= tail_start
    energy_slope = _linear_slope(rms_times[tail_mask], rms[tail_mask])

    f0, voiced_flag, _voiced_prob = librosa.pyin(clip, fmin=_FMIN, fmax=_FMAX, sr=sr)
    f0_times = librosa.times_like(f0, sr=sr)
    voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])
    pitch_mean_hz = float(np.mean(voiced_f0)) if len(voiced_f0) >= 1 else None

    tail_voiced_mask = voiced_flag & (f0_times >= tail_start)
    # Slope of a linear fit over the tail's voiced frames, in Hz/sec -- a
    # rough proxy for whether the speaker's intonation is falling
    # (statement-like, "done talking") or flat/rising (mid-thought).
    pitch_slope = _linear_slope(f0_times[tail_voiced_mask], f0[tail_voiced_mask])

    return AcousticFeatures(
        pitch_mean_hz=pitch_mean_hz, pitch_slope=pitch_slope, energy_rms=energy_rms, energy_slope=energy_slope
    )
