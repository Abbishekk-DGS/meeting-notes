import json

from meeting_notes.summarize import (
    Notes,
    drop_task_echoes,
    render_markdown,
    summarize,
    window_segments,
)
from meeting_notes.transcribe import Segment, stamp, to_srt, to_vtt

SEGMENTS = [
    Segment(0.0, 4.2, "Okay let's start, the launch is moving to March."),
    Segment(4.2, 9.9, "Priya will update the pricing page by Friday."),
    Segment(65.5, 70.0, "Do we still need the beta waitlist?"),
]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, schema=None):
        self.calls.append(prompt)
        if schema is None:
            return "Launch moves to March; pricing page update owned by Priya."
        return json.dumps(
            {
                "key_points": [{"text": "Launch moves to March", "time": "00:00"}],
                "decisions": [{"text": "Launch in March", "time": "00:00"}],
                "action_items": [
                    {
                        "task": "Update pricing page",
                        "owner": "Priya",
                        "due": "Friday",
                        "time": "00:04",
                    }
                ],
                "open_questions": [{"text": "Keep beta waitlist?", "time": "01:05"}],
            }
        )


def test_timestamps():
    assert stamp(75.3) == "01:15"
    assert stamp(3725) == "1:02:05"


def test_srt_and_vtt_format():
    srt = to_srt(SEGMENTS)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:04,200\n")
    assert "00:01:05,500 --> 00:01:10,000" in srt
    assert to_vtt(SEGMENTS).startswith("WEBVTT\n\n00:00:00.000 --> 00:00:04.200")


def test_windows_never_split_segments():
    windows = window_segments(SEGMENTS, max_words=20)
    assert len(windows) == 2
    assert windows[0].count("[") == 2 and "[01:05]" in windows[1]


def test_map_reduce_flow():
    llm = FakeLLM()
    notes = summarize(SEGMENTS, llm, max_words=20, log=lambda _: None)
    # 2 map calls + 1 reduce + 1 tldr
    assert len(llm.calls) == 4
    assert notes.action_items[0]["owner"] == "Priya"
    assert "March" in notes.tldr


def test_parse_handles_code_fences():
    notes = Notes.parse('```json\n{"key_points": [], "decisions": [], "action_items": []}\n```')
    assert notes.open_questions == []


def test_markdown_render():
    notes = Notes.parse(FakeLLM().generate("x", schema={}))
    notes.tldr = "Short."
    md = render_markdown(notes, "Weekly Sync", 70.0, "en")
    assert "# Weekly Sync" in md
    assert "| Update pricing page | Priya | Friday | `00:04` |" in md
    assert "Duration 01:10" in md


def test_decisions_that_echo_tasks_are_dropped():
    notes = Notes(
        decisions=[
            {"text": "Priya will update the pricing page by Friday", "time": "00:06"},
            {"text": "Launch moves to March", "time": "00:00"},
        ],
        action_items=[{"task": "Update pricing page", "owner": "Priya", "due": "Friday"}],
    )
    assert [d["text"] for d in drop_task_echoes(notes).decisions] == ["Launch moves to March"]
