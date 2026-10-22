"""meeting-notes: turn a recording into meeting notes, entirely on your machine.

meeting-notes meeting.m4a                    # notes + transcript + subtitles
meeting-notes lecture.mp4 --whisper medium --model llama3.1:8b
meeting-notes --from-transcript out/meeting.json   # re-summarize without re-transcribing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .summarize import OllamaLLM, render_markdown, summarize
from .transcribe import Segment, stamp, to_srt, to_transcript, to_vtt, transcribe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meeting-notes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("audio", nargs="?", help="audio or video file")
    parser.add_argument("--from-transcript", help="reuse a saved <name>.json transcript")
    parser.add_argument("--whisper", default="small", help="tiny|base|small|medium|large-v3")
    parser.add_argument("--language", default=None, help="force language, e.g. en, ta, hi")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model for summaries")
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_transcript:
        data = json.loads(Path(args.from_transcript).read_text())
        segments = [Segment(**s) for s in data["segments"]]
        language, name = data["language"], Path(args.from_transcript).stem
    elif args.audio:
        audio = Path(args.audio)
        name = audio.stem
        print(f"Transcribing {audio.name} with whisper-{args.whisper}...")
        t0 = time.time()
        segments, language = transcribe(
            audio,
            args.whisper,
            args.language,
            on_segment=lambda s: print(f"  [{stamp(s.start)}] {s.text}"),
        )
        print(f"Transcribed in {time.time() - t0:.0f}s ({language})")
        (out / f"{name}.json").write_text(
            json.dumps({"language": language, "segments": [asdict(s) for s in segments]})
        )
        (out / f"{name}.srt").write_text(to_srt(segments))
        (out / f"{name}.vtt").write_text(to_vtt(segments))
        (out / f"{name}.txt").write_text(to_transcript(segments))
    else:
        parser.error("give an audio file or --from-transcript")

    if not segments:
        print("No speech found.", file=sys.stderr)
        return 1

    print(f"Summarizing with {args.model}...")
    notes = summarize(segments, OllamaLLM(args.model))
    title = args.title or name.replace("_", " ").replace("-", " ").title()
    md = render_markdown(notes, title, segments[-1].end, language)
    (out / f"{name}.notes.md").write_text(md)
    print(md)
    print(f"Saved to {out}/{name}.notes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
