"""Live demo: pick a conversation clip, see the interject-score timeline,
and watch a simulated live playback with a marker lighting up at good
moments for an agent to jump in.

Run with: streamlit run demo/app.py
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src.*` imports work run from anywhere

from src.audio.pipeline import ProcessedSegment, process_audio  # noqa: E402
from src.scoring import interjection  # noqa: E402
from src.scoring.interjection import ScoredPauseWindow, score_conversation  # noqa: E402
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR  # noqa: E402

st.set_page_config(page_title="InterjectNet", layout="wide")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
SPEAKER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
GOOD_COLOR, MAYBE_COLOR, BAD_COLOR = "#2ca02c", "#d4a017", "#c44e52"


def list_audio_files() -> list[Path]:
    if not DATA_RAW_DIR.exists():
        return []
    return sorted(p for p in DATA_RAW_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def cache_path_for(audio_path: Path) -> Path:
    return DATA_PROCESSED_DIR / f"{audio_path.stem}.segments.json"


@st.cache_data(show_spinner=False)
def load_or_process_segments(audio_path_str: str) -> list[dict]:
    """Diarization + transcription are slow (~real-time on CPU); cache the
    result to disk per clip so re-visiting a clip in the demo is instant."""
    audio_path = Path(audio_path_str)
    cache_path = cache_path_for(audio_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    segments = process_audio(str(audio_path))
    raw = [asdict(s) for s in segments]
    cache_path.write_text(json.dumps(raw))
    return raw


def speaker_color(speaker: str, speakers: list[str]) -> str:
    return SPEAKER_COLORS[speakers.index(speaker) % len(SPEAKER_COLORS)]


def score_color(score: float, threshold: float) -> str:
    if score >= threshold:
        return GOOD_COLOR
    if score >= threshold - 0.15:
        return MAYBE_COLOR
    return BAD_COLOR


def build_timeline_figure(
    segments: list[ProcessedSegment],
    windows: list[ScoredPauseWindow],
    threshold: float,
    now: float | None = None,
):
    speakers = sorted({s.speaker for s in segments})
    fig, (ax_speakers, ax_score) = plt.subplots(
        2, 1, figsize=(11, 4), sharex=True, height_ratios=[1.1, 1], gridspec_kw={"hspace": 0.1}
    )

    for s in segments:
        if not s.text.strip():
            continue
        y = speakers.index(s.speaker)
        ax_speakers.barh(y, s.end - s.start, left=s.start, height=0.6, color=speaker_color(s.speaker, speakers))
    ax_speakers.set_yticks(range(len(speakers)))
    ax_speakers.set_yticklabels(speakers)
    ax_speakers.set_title("Who's talking", loc="left", fontsize=10)
    ax_speakers.margins(y=0.3)

    xs = [(w.start + w.end) / 2 for w in windows]
    ys = [w.interject_score for w in windows]
    colors = [score_color(s, threshold) for s in ys]
    ax_score.scatter(xs, ys, c=colors, s=45, zorder=3, edgecolors="white", linewidths=0.5)
    ax_score.axhline(threshold, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.set_ylabel("Interject score")
    ax_score.set_xlabel("Time (s)")
    ax_score.set_title("Good moments to jump in", loc="left", fontsize=10)

    if now is not None:
        for ax in (ax_speakers, ax_score):
            ax.axvline(now, color="black", linewidth=1.5, alpha=0.85)

    fig.tight_layout()
    return fig


def status_for_time(
    t: float,
    segments: list[ProcessedSegment],
    windows: list[ScoredPauseWindow],
):
    for s in segments:
        if s.text.strip() and s.start <= t <= s.end:
            return "talking", s, None
    for w in windows:
        if w.start <= t <= w.end:
            return "pause", None, w
    return "idle", None, None


def main():
    st.title("InterjectNet")
    st.caption(
        "Predicts *when* an AI agent should speak up in a live multi-speaker conversation -- "
        "not what it would say, just the timing."
    )

    files = list_audio_files()
    if not files:
        st.warning(f"No audio files found in `{DATA_RAW_DIR}`. Drop a .mp3/.wav there and refresh.")
        return

    cached_files = [p for p in files if cache_path_for(p).exists()]
    default_index = files.index(cached_files[0]) if cached_files else 0

    with st.sidebar:
        st.header("Clip")

        def _label(p: Path) -> str:
            tag = "✅ ready" if cache_path_for(p).exists() else "⏳ not processed yet"
            return f"{p.name}  [{tag}]"

        selected = st.selectbox("Choose a conversation", files, index=default_index, format_func=_label)
        if not cache_path_for(selected).exists():
            st.warning(
                "This clip hasn't been processed yet. Diarization + transcription run on CPU and can take "
                "several minutes -- the whole app is unresponsive while it runs, so kick it off before you "
                "need the demo, not during it."
            )

        st.header("Scoring weights")
        st.caption("Rule-based, not ML -- nudge these live and rescoring is instant.")
        pause_w = st.slider("Pause length", 0.0, 1.0, interjection.PAUSE_WEIGHT, 0.05)
        completion_w = st.slider("Completion (pitch/energy)", 0.0, 1.0, interjection.COMPLETION_WEIGHT, 0.05)
        topic_w = st.slider("Topic stability", 0.0, 1.0, interjection.TOPIC_WEIGHT, 0.05)
        weight_total = pause_w + completion_w + topic_w
        if weight_total > 0:
            interjection.PAUSE_WEIGHT = pause_w / weight_total
            interjection.COMPLETION_WEIGHT = completion_w / weight_total
            interjection.TOPIC_WEIGHT = topic_w / weight_total

        threshold = st.slider("Interject threshold", 0.0, 1.0, 0.6, 0.05)

        if st.button("Reprocess this clip"):
            cache_path_for(selected).unlink(missing_ok=True)
            load_or_process_segments.clear()

    already_cached = cache_path_for(selected).exists()
    spinner_msg = (
        f"Loading {selected.name}..."
        if already_cached
        else f"Processing {selected.name} for the first time (diarization + transcription, a few minutes)..."
    )
    with st.spinner(spinner_msg):
        raw_segments = load_or_process_segments(str(selected))
    segments = [ProcessedSegment(**s) for s in raw_segments]
    windows = score_conversation(segments)

    if not windows:
        st.error("No scoreable pause windows found in this clip.")
        return

    st.subheader("Conversation overview")
    st.pyplot(build_timeline_figure(segments, windows, threshold), clear_figure=True)

    col_audio, col_stats = st.columns([2, 1])
    with col_audio:
        st.audio(str(selected))
    with col_stats:
        good = sum(1 for w in windows if w.interject_score >= threshold)
        st.metric("Good interjection moments", f"{good} / {len(windows)}")

    st.subheader("Simulated live playback")
    st.caption("Steps through the conversation with the marker moving in real time; play the audio above alongside it.")
    speed = st.select_slider("Playback speed", options=[1, 2, 4, 8], value=2)
    total_duration = segments[-1].end

    if st.button("Play", type="primary"):
        chart_slot = st.empty()
        status_slot = st.empty()
        progress_slot = st.progress(0.0)

        start = time.time()
        while True:
            elapsed = (time.time() - start) * speed
            if elapsed > total_duration:
                break

            state, seg, win = status_for_time(elapsed, segments, windows)
            if state == "talking":
                status_slot.markdown(f"### \U0001f534 {seg.speaker} is talking\n> {seg.text}")
            elif state == "pause":
                emoji = "\U0001f7e2" if win.interject_score >= threshold else "\U0001f7e1"
                status_slot.markdown(
                    f"### {emoji} Pause -- interject score **{win.interject_score:.2f}**\n"
                    f"pause {win.pause_score:.2f} &middot; completion {win.completion_score:.2f} "
                    f"&middot; topic {win.topic_stability_score:.2f}"
                )
            else:
                status_slot.markdown("### ⚪ ...")

            chart_slot.pyplot(build_timeline_figure(segments, windows, threshold, now=elapsed), clear_figure=True)
            progress_slot.progress(min(1.0, elapsed / total_duration))
            time.sleep(0.3)

        status_slot.markdown("### ✅ Done")

    st.subheader("All pause windows")
    df = pd.DataFrame(
        [
            {
                "start": w.start,
                "end": w.end,
                "duration": w.duration,
                "prev_speaker": w.prev_speaker,
                "prev_text": w.prev_text,
                "next_speaker": w.next_speaker,
                "next_text": w.next_text,
                "interject_score": w.interject_score,
                "pause": w.pause_score,
                "completion": w.completion_score,
                "topic": w.topic_stability_score,
            }
            for w in windows
        ]
    ).sort_values("interject_score", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
