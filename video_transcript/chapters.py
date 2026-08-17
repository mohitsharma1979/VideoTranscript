from __future__ import annotations

import re

from .models import Chapter, Segment


def _title(text: str, max_words: int = 8) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,!?:;-\n\t")
    words = cleaned.split()
    result = " ".join(words[:max_words]) or "Untitled section"
    if len(words) > max_words:
        result += "…"
    return result[0].upper() + result[1:]


def build_chapters(
    segments: list[Segment], target_seconds: float = 180.0, pause_seconds: float = 2.5
) -> list[Chapter]:
    """Group transcript segments, preferring natural pauses near target duration."""
    if not segments:
        return []
    chapters: list[Chapter] = []
    start_index = 0
    for index in range(1, len(segments)):
        elapsed = segments[index].start - segments[start_index].start
        gap = segments[index].start - segments[index - 1].end
        should_split = elapsed >= target_seconds and gap >= pause_seconds
        force_split = elapsed >= target_seconds * 1.5
        if should_split or force_split:
            chapters.append(_make_chapter(chapters, segments, start_index, index - 1))
            start_index = index
    chapters.append(_make_chapter(chapters, segments, start_index, len(segments) - 1))
    return chapters


def _make_chapter(
    existing: list[Chapter], segments: list[Segment], start: int, end: int
) -> Chapter:
    return Chapter(
        number=len(existing) + 1,
        start=segments[start].start,
        end=segments[end].end,
        title=_title(segments[start].text),
        segment_start=start,
        segment_end=end,
    )

