from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi"}


class MediaError(RuntimeError):
    pass


def validate_input(path: Path) -> None:
    if not path.exists():
        raise MediaError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise MediaError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        choices = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise MediaError(f"Unsupported file type '{path.suffix}'. Supported: {choices}")
    if shutil.which("ffmpeg") is None:
        raise MediaError("FFmpeg was not found. Install it and ensure 'ffmpeg' is on PATH.")
    if shutil.which("ffprobe") is None:
        raise MediaError("FFprobe was not found. Install FFmpeg and ensure 'ffprobe' is on PATH.")


def duration(input_path: Path) -> float:
    validate_input(input_path)
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(input_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MediaError(f"Could not read video duration: {result.stderr.strip()}")
    try:
        value = float(result.stdout.strip())
    except ValueError as exc:
        raise MediaError("FFprobe returned an invalid video duration.") from exc
    if value <= 0:
        raise MediaError("The video has no measurable duration.")
    return value


def extract_audio(
    input_path: Path,
    output_path: Path,
    start: float = 0.0,
    length: float | None = None,
) -> None:
    """Extract mono, 16 kHz PCM audio suitable for speech recognition."""
    validate_input(input_path)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(input_path),
    ]
    if length is not None:
        command += ["-t", f"{length:.3f}"]
    command += [
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown FFmpeg error"
        raise MediaError(f"Could not extract audio: {detail}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise MediaError("FFmpeg produced no audio. The video may not contain an audio track.")
