"""Speech to timestamped segments with faster-whisper (CTranslate2, int8 on CPU)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(
    audio: Path,
    model_size: str = "small",
    language: str | None = None,
    on_segment: Callable[[Segment], None] | None = None,
) -> tuple[list[Segment], str]:
    """Returns (segments, detected_language). Accepts any format ffmpeg can decode."""
    from faster_whisper import WhisperModel  # heavy import, keep it lazy

    model = WhisperModel(model_size, device="auto", compute_type="int8")
    raw, info = model.transcribe(
        str(audio),
        language=language,
        vad_filter=True,  # Silero VAD drops silence, which also stops hallucinated filler
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,  # avoids repetition loops on long recordings
    )
    segments = []
    for s in raw:
        seg = Segment(s.start, s.end, s.text.strip())
        segments.append(seg)
        if on_segment:
            on_segment(seg)
    return segments, info.language


def stamp(seconds: float) -> str:
    """Human timestamp: 75.3 -> '01:15', 3725 -> '1:02:05'."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _srt_time(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{i}\n{_srt_time(s.start, ',')} --> {_srt_time(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    cues = [f"{_srt_time(s.start, '.')} --> {_srt_time(s.end, '.')}\n{s.text}\n" for s in segments]
    return "WEBVTT\n\n" + "\n".join(cues)


def to_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{stamp(s.start)}] {s.text}" for s in segments)
