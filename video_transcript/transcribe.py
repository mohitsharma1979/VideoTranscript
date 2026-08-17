from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import psutil

from .media import duration, extract_audio
from .models import Segment


class TranscriptionError(RuntimeError):
    pass


def _segments_from_json(path: Path, audio_offset: float) -> list[Segment]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload["transcription"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TranscriptionError("whisper.cpp produced invalid JSON output.") from exc

    segments = []
    for entry in entries:
        text = str(entry.get("text", "")).strip()
        offsets = entry.get("offsets", {})
        if not text:
            continue
        try:
            start = audio_offset + float(offsets["from"]) / 1000
            end = audio_offset + float(offsets["to"]) / 1000
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError("whisper.cpp output contains invalid timestamps.") from exc
        segments.append(Segment(max(0.0, start), max(start, end), text))
    return segments


def transcribe_video(
    video_path: Path,
    work_dir: Path,
    model_path: Path,
    vad_model_path: Path,
    chunk_seconds: float = 900.0,
    overlap_seconds: float = 3.0,
    language: str | None = None,
    threads: int = 4,
    cpu_only: bool = False,
    progress: Callable[[str], None] | None = None,
    chunk_metrics: list[dict] | None = None,
) -> tuple[list[Segment], dict]:
    """Transcribe bounded chunks and merge them onto the source video's timeline."""
    executable = shutil.which("whisper-cli")
    if executable is None:
        raise TranscriptionError(
            "whisper-cli was not found. Install it with: brew install whisper-cpp"
        )
    if not model_path.is_file():
        raise TranscriptionError(f"Whisper model was not found: {model_path}")
    if not vad_model_path.is_file():
        raise TranscriptionError(f"VAD model was not found: {vad_model_path}")
    if chunk_seconds <= 0 or overlap_seconds < 0:
        raise TranscriptionError("Chunk length must be positive and overlap cannot be negative.")
    if overlap_seconds * 2 >= chunk_seconds:
        raise TranscriptionError("Chunk overlap must be less than half the chunk length.")

    notify = progress or (lambda _: None)
    metrics = chunk_metrics if chunk_metrics is not None else []
    video_duration = duration(video_path)
    chunk_count = math.ceil(video_duration / chunk_seconds)
    all_segments: list[Segment] = []
    detected_languages: list[str] = []
    audio_path = work_dir / "chunk.wav"
    output_base = work_dir / "chunk-result"
    output_json = output_base.with_suffix(".json")

    for index in range(chunk_count):
        nominal_start = index * chunk_seconds
        nominal_end = min(video_duration, nominal_start + chunk_seconds)
        extraction_start = max(0.0, nominal_start - overlap_seconds)
        extraction_end = min(video_duration, nominal_end + overlap_seconds)
        notify(
            f"Chunk {index + 1}/{chunk_count}: "
            f"{_clock(nominal_start)}–{_clock(nominal_end)}"
        )
        chunk_started = time.perf_counter()
        chunk_record = {
            "number": index + 1,
            "nominal_start_seconds": nominal_start,
            "nominal_end_seconds": nominal_end,
            "extraction_start_seconds": extraction_start,
            "extraction_end_seconds": extraction_end,
            "status": "running",
        }
        metrics.append(chunk_record)
        try:
            extraction_started = time.perf_counter()
            extract_audio(
                video_path, audio_path, extraction_start, extraction_end - extraction_start
            )
            chunk_record["extraction_seconds"] = time.perf_counter() - extraction_started
            chunk_record["temporary_audio_bytes"] = audio_path.stat().st_size
            command = [
                executable,
                "-m", str(model_path),
                "-f", str(audio_path),
                "-l", language or "auto",
                "-t", str(threads),
                "-bs", "5",
                "--vad", "-vm", str(vad_model_path),
                "-ojf", "-of", str(output_base),
                "-np",
            ]
            if cpu_only:
                command.append("--no-gpu")
            transcribe_started = time.perf_counter()
            returncode, stderr, peak_rss = _run_monitored(command)
            chunk_record["transcription_seconds"] = time.perf_counter() - transcribe_started
            chunk_record["peak_memory_bytes"] = peak_rss
            if returncode != 0:
                detail = _last_error(stderr)
                raise TranscriptionError(
                    f"whisper.cpp failed on chunk {index + 1}/{chunk_count}: {detail}"
                )
            chunk_segments = _segments_from_json(output_json, extraction_start)
            # Overlapping context improves boundary recognition. Each nominal chunk owns
            # segments whose midpoint lies inside its non-overlapping time window.
            all_segments.extend(
                segment
                for segment in chunk_segments
                if nominal_start <= (segment.start + segment.end) / 2 < nominal_end
            )
            chunk_record["segments_created"] = sum(
                1
                for segment in chunk_segments
                if nominal_start <= (segment.start + segment.end) / 2 < nominal_end
            )
            try:
                result_payload = json.loads(output_json.read_text(encoding="utf-8"))
                detected = result_payload.get("result", {}).get("language")
                if detected:
                    detected_languages.append(str(detected))
            except (OSError, json.JSONDecodeError):
                pass
            chunk_record["status"] = "completed"
        except Exception as exc:
            chunk_record["status"] = "failed"
            chunk_record["error"] = str(exc)
            raise
        finally:
            # Never retain potentially large audio chunks after their attempt.
            audio_path.unlink(missing_ok=True)
            output_json.unlink(missing_ok=True)
            chunk_record["elapsed_seconds"] = time.perf_counter() - chunk_started
            chunk_record["temporary_files_removed"] = (
                not audio_path.exists() and not output_json.exists()
            )

    all_segments.sort(key=lambda segment: (segment.start, segment.end))
    selected_language = language or _most_common(detected_languages)
    metadata = {
        "language": selected_language,
        "duration": video_duration,
        "model": str(model_path.resolve()),
        "engine": "whisper.cpp",
        "acceleration": "cpu" if cpu_only else "metal",
        "chunk_seconds": chunk_seconds,
        "chunk_overlap_seconds": overlap_seconds,
        "chunk_count": chunk_count,
    }
    return all_segments, metadata


def _run_monitored(command: list[str], interval: float = 0.05) -> tuple[int, str, int]:
    """Run whisper-cli and sample its process-tree resident memory."""
    import tempfile

    peak_rss = 0
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors, text=True)
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            peak_rss = max(peak_rss, _process_tree_rss(monitored))
            time.sleep(interval)
        peak_rss = max(peak_rss, _process_tree_rss(monitored))
        returncode = process.wait()
        errors.seek(0)
        stderr = errors.read()
    return returncode, stderr, peak_rss


def _process_tree_rss(process: psutil.Process) -> int:
    total = 0
    try:
        total += process.memory_info().rss
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        return total
    for child in children:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            pass
    return total


def _clock(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _last_error(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown whisper.cpp error"


def _most_common(values: list[str]) -> str | None:
    return max(set(values), key=values.count) if values else None
