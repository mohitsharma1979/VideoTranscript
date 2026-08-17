from __future__ import annotations

import json
from pathlib import Path

from .models import Chapter, Segment


def timestamp(seconds: float, separator: str = ".") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def write_outputs(
    output_dir: Path,
    source: Path,
    segments: list[Segment],
    chapters: list[Chapter],
    metadata: dict,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    files = {
        "md": output_dir / f"{stem}.transcript.md",
        "json": output_dir / f"{stem}.transcript.json",
        "srt": output_dir / f"{stem}.srt",
        "vtt": output_dir / f"{stem}.vtt",
    }

    markdown = [f"# Transcript: {source.name}", "", "## Chapters", ""]
    markdown.extend(
        f"- [{timestamp(c.start)[:-4]}] **{c.title}**" for c in chapters
    )
    markdown += ["", "## Timestamped transcript", ""]
    chapter_by_segment = {c.segment_start: c for c in chapters}
    for index, segment in enumerate(segments):
        if index in chapter_by_segment:
            chapter = chapter_by_segment[index]
            markdown += [f"### {chapter.number}. {chapter.title}", ""]
        markdown.append(f"**[{timestamp(segment.start)[:-4]}]** {segment.text}")
        markdown.append("")
    files["md"].write_text("\n".join(markdown), encoding="utf-8")

    payload = {
        "source": str(source.resolve()),
        "metadata": metadata,
        "chapters": [c.to_dict() for c in chapters],
        "segments": [s.to_dict() for s in segments],
    }
    files["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    srt = []
    vtt = ["WEBVTT", ""]
    for index, segment in enumerate(segments, 1):
        srt += [str(index), f"{timestamp(segment.start, ',')} --> {timestamp(segment.end, ',')}", segment.text, ""]
        vtt += [f"{timestamp(segment.start)} --> {timestamp(segment.end)}", segment.text, ""]
    files["srt"].write_text("\n".join(srt), encoding="utf-8")
    files["vtt"].write_text("\n".join(vtt), encoding="utf-8")
    return list(files.values())

