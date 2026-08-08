"""Merge diarization turns + transcribed words into speaker-labeled utterances."""

from dataclasses import dataclass, field

from src.audio.diarization import SpeakerTurn
from src.audio.transcription import TranscribedWord


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str
    words: list[TranscribedWord] = field(default_factory=list)
    pause_before: float = 0.0  # seconds of silence since the previous utterance ended
    pause_after: float = 0.0  # seconds of silence until the next utterance starts


def _nearest_turn(word: TranscribedWord, turns: list[SpeakerTurn]) -> SpeakerTurn:
    midpoint = (word.start + word.end) / 2
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn
    # fell in a gap between turns (diarization boundary noise) -> nearest by distance
    return min(turns, key=lambda t: min(abs(midpoint - t.start), abs(midpoint - t.end)))


def build_utterances(turns: list[SpeakerTurn], words: list[TranscribedWord]) -> list[Utterance]:
    """One utterance per diarization turn, with its words and text attached."""
    if not turns:
        return []

    words_by_turn: dict[int, list[TranscribedWord]] = {i: [] for i in range(len(turns))}
    turns_sorted = sorted(range(len(turns)), key=lambda i: turns[i].start)
    turns_ordered = [turns[i] for i in turns_sorted]

    for word in words:
        turn = _nearest_turn(word, turns_ordered)
        words_by_turn[turns_ordered.index(turn)].append(word)

    utterances = []
    for i, turn in enumerate(turns_ordered):
        turn_words = words_by_turn[i]
        text = " ".join(w.word for w in turn_words)
        utterances.append(
            Utterance(speaker=turn.speaker, start=turn.start, end=turn.end, text=text, words=turn_words)
        )

    # utterances with no words (diarization found speech, whisper didn't
    # transcribe anything there -- e.g. laughter, backchannel noise) are
    # still useful for pause timing, so we keep them.
    for i in range(len(utterances)):
        prev_end = utterances[i - 1].end if i > 0 else utterances[i].start
        utterances[i].pause_before = max(0.0, utterances[i].start - prev_end)
        next_start = utterances[i + 1].start if i + 1 < len(utterances) else utterances[i].end
        utterances[i].pause_after = max(0.0, next_start - utterances[i].end)

    return utterances
