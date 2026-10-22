"""Map-reduce summarization that scales to multi-hour recordings on a small local model.

Map:    each transcript window -> JSON notes (points, decisions, actions, questions)
Reduce: merge window notes -> one de-duplicated set + TL;DR
Every item keeps the timestamp of the moment it came from, so notes stay verifiable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .transcribe import Segment, stamp

_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else f"http://{_host}"

NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points": {"type": "array", "items": {"$ref": "#/$defs/item"}},
        "decisions": {"type": "array", "items": {"$ref": "#/$defs/item"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["task", "owner", "due", "time"],
            },
        },
        "open_questions": {"type": "array", "items": {"$ref": "#/$defs/item"}},
    },
    "required": ["key_points", "decisions", "action_items", "open_questions"],
    "$defs": {
        "item": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "time": {"type": "string"}},
            "required": ["text", "time"],
        }
    },
}

MAP_PROMPT = """You are an expert meeting note-taker. From the transcript excerpt, extract:
- key_points: important information discussed
- decisions: anything agreed or decided ("we decided", "let's go with", "agreed")
- action_items: every task someone is asked or volunteers to do. owner = the person's name,
  or "unassigned". due = the deadline if one is said, else "".
- open_questions: questions left unresolved
"time" is the [mm:ss] timestamp of the line where the item was said, written as mm:ss.
Only include what is in the excerpt. Reply with JSON only.

Example excerpt:
[00:00] We agreed to ship the mobile app first. Sam, can you book the venue by Monday?
[00:09] Still not sure who owns support.
Example answer:
{{"key_points": [{{"text": "Mobile app ships before web", "time": "00:00"}}],
 "decisions": [{{"text": "Ship the mobile app first", "time": "00:00"}}],
 "action_items": [{{"task": "Book the venue", "owner": "Sam", "due": "Monday", "time": "00:00"}}],
 "open_questions": [{{"text": "Who owns support?", "time": "00:09"}}]}}

Transcript excerpt:
{chunk}"""

REDUCE_PROMPT = """Merge these partial meeting notes into one final set. Remove duplicates, keep the
earliest timestamp for merged items, keep every distinct action item. Reply with JSON only.

Partial notes:
{notes}"""

TLDR_PROMPT = """Write a 2-3 sentence TL;DR of this meeting from its notes. Plain text only.

{notes}"""


class LLM(Protocol):
    def generate(self, prompt: str, schema: dict | None = None) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:3b", url: str = OLLAMA_URL, num_ctx: int = 8192):
        self.model, self.url, self.num_ctx = model, url, num_ctx

    def generate(self, prompt: str, schema: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": self.num_ctx},
        }
        if schema:
            payload["format"] = schema  # Ollama constrains decoding to this JSON schema
        resp = httpx.post(f"{self.url}/api/generate", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["response"]


@dataclass
class Notes:
    key_points: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    open_questions: list[dict] = field(default_factory=list)
    tldr: str = ""

    @classmethod
    def parse(cls, raw: str) -> Notes:
        data = json.loads(_strip_fences(raw))
        notes = cls(**{k: list(data.get(k) or []) for k in NOTES_SCHEMA["required"]})
        for key in NOTES_SCHEMA["required"]:
            for item in getattr(notes, key):
                if isinstance(item, dict):
                    item["time"] = str(item.get("time", "")).strip("[]() ")
        return notes

    def as_json(self) -> str:
        return json.dumps({k: getattr(self, k) for k in NOTES_SCHEMA["required"]}, indent=1)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return raw


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def drop_task_echoes(notes: Notes, overlap: float = 0.5) -> Notes:
    """Small models often restate action items as decisions; keep each fact in one section."""
    tasks = [_words(f"{a.get('task', '')} {a.get('owner', '')}") for a in notes.action_items]

    def is_echo(item: dict) -> bool:
        words = _words(item.get("text", ""))
        return bool(words) and any(len(words & t) / len(words) >= overlap for t in tasks)

    notes.decisions = [d for d in notes.decisions if not is_echo(d)]
    return notes


def window_segments(segments: list[Segment], max_words: int = 1500) -> list[str]:
    """Group segments into windows of ~max_words, never splitting a segment."""
    windows, current, count = [], [], 0
    for seg in segments:
        words = len(seg.text.split())
        if current and count + words > max_words:
            windows.append("\n".join(current))
            current, count = [], 0
        current.append(f"[{stamp(seg.start)}] {seg.text}")
        count += words
    if current:
        windows.append("\n".join(current))
    return windows


def summarize(segments: list[Segment], llm: LLM, max_words: int = 1500, log=print) -> Notes:
    windows = window_segments(segments, max_words)
    partials: list[Notes] = []
    for i, chunk in enumerate(windows, 1):
        log(f"  map {i}/{len(windows)}")
        partials.append(Notes.parse(llm.generate(MAP_PROMPT.format(chunk=chunk), NOTES_SCHEMA)))

    # Reduce in groups so the merge prompt always fits the context window.
    while len(partials) > 1:
        log(f"  reduce {len(partials)} partial notes")
        merged = []
        for i in range(0, len(partials), 4):
            group = partials[i : i + 4]
            if len(group) == 1:
                merged.append(group[0])
                continue
            joined = "\n\n".join(n.as_json() for n in group)
            merged.append(
                Notes.parse(llm.generate(REDUCE_PROMPT.format(notes=joined), NOTES_SCHEMA))
            )
        partials = merged

    notes = drop_task_echoes(partials[0] if partials else Notes())
    notes.tldr = llm.generate(TLDR_PROMPT.format(notes=notes.as_json())).strip()
    return notes


def render_markdown(notes: Notes, title: str, duration: float, language: str) -> str:
    def items(rows: list[dict]) -> str:
        return "\n".join(f"- {r.get('text', '')} `{r.get('time', '')}`" for r in rows) or "- None"

    actions = "\n".join(
        f"| {a.get('task', '')} | {a.get('owner') or 'unassigned'} | {a.get('due') or '-'} "
        f"| `{a.get('time', '')}` |"
        for a in notes.action_items
    )
    action_table = (
        f"| Task | Owner | Due | At |\n|---|---|---|---|\n{actions}" if actions else "- None"
    )
    return f"""# {title}

*Duration {stamp(duration)} · language: {language} · generated locally*

## TL;DR
{notes.tldr}

## Key points
{items(notes.key_points)}

## Decisions
{items(notes.decisions)}

## Action items
{action_table}

## Open questions
{items(notes.open_questions)}
"""
