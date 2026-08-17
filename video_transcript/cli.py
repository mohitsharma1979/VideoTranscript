from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from .chapters import build_chapters
from .media import MediaError, validate_input
from .output import write_outputs
from .run_log import bytes_label, write_run_log
from .transcribe import TranscriptionError, transcribe_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ggml-small.bin"
DEFAULT_VAD_MODEL = PROJECT_ROOT / "models" / "ggml-silero-v6.2.0.bin"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="video-transcript",
        description="Transcribe a local video offline and create simple timestamped chapters.",
    )
    result.add_argument("video", type=Path, help="Local .mov, .mp4, .mkv, or .avi file")
    result.add_argument("-o", "--output-dir", type=Path, default=Path("transcripts"))
    result.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Run telemetry directory")
    result.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Local whisper.cpp GGML model")
    result.add_argument("--vad-model-path", type=Path, default=DEFAULT_VAD_MODEL, help="Local whisper.cpp VAD model")
    result.add_argument("--language", help="Language code such as en; auto-detected by default")
    result.add_argument("--cpu-only", action="store_true", help="Disable Metal GPU acceleration")
    result.add_argument("--threads", type=int, default=4, help="CPU threads used by whisper.cpp")
    result.add_argument("--chunk-minutes", type=float, default=15.0, help="Audio chunk length (default: 15)")
    result.add_argument("--chunk-overlap-seconds", type=float, default=3.0, help="Context around chunk boundaries")
    result.add_argument("--chapter-minutes", type=float, default=3.0, help="Approximate chapter length")
    result.add_argument("--pause-seconds", type=float, default=2.5, help="Preferred pause at chapter boundary")
    return result


def run(args: argparse.Namespace, chunk_metrics: list[dict]) -> tuple[list[Path], dict]:
    validate_input(args.video)
    if args.chapter_minutes <= 0 or args.pause_seconds < 0 or args.threads <= 0:
        raise ValueError("Chapter minutes and threads must be positive; pause seconds cannot be negative.")
    with tempfile.TemporaryDirectory(prefix="video-transcript-") as directory:
        segments, metadata = transcribe_video(
            args.video,
            Path(directory),
            args.model_path,
            args.vad_model_path,
            args.chunk_minutes * 60,
            args.chunk_overlap_seconds,
            args.language,
            args.threads,
            args.cpu_only,
            progress=lambda message: print(message, file=sys.stderr),
            chunk_metrics=chunk_metrics,
        )
    if not segments:
        raise TranscriptionError("No speech was detected in the video.")
    chapters = build_chapters(
        segments, args.chapter_minutes * 60, args.pause_seconds
    )
    files = write_outputs(args.output_dir, args.video, segments, chapters, metadata)
    details = {
        "metadata": metadata,
        "segments_created": len(segments),
        "chapters_created": len(chapters),
    }
    return files, details


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    started_at = datetime.now().astimezone()
    started_clock = time.perf_counter()
    run_id = str(uuid.uuid4())
    chunk_metrics: list[dict] = []
    files: list[Path] = []
    details: dict = {}
    status = "failed"
    error: str | None = None
    exit_code = 1
    try:
        args = parser().parse_args(argv)
        files, details = run(args, chunk_metrics)
        status = "completed"
        exit_code = 0
    except (MediaError, TranscriptionError, ValueError) as exc:
        error = str(exc)
        print(f"Error: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        status = "cancelled"
        error = "Cancelled by user"
        exit_code = 130
        print("\nCancelled.", file=sys.stderr)
    finally:
        if args is not None:
            completed_at = datetime.now().astimezone()
            record = _run_record(
                run_id,
                args,
                status,
                started_at,
                completed_at,
                time.perf_counter() - started_clock,
                chunk_metrics,
                files,
                details,
                error,
            )
            try:
                log_path = write_run_log(args.log_dir, record)
                print(f"Run log: {log_path}", file=sys.stderr)
            except OSError as exc:
                print(f"Warning: could not write run log: {exc}", file=sys.stderr)
    if status == "completed":
        print("Done. Created:")
        for path in files:
            print(f"  {path}")
        peak = max((item.get("peak_memory_bytes", 0) for item in chunk_metrics), default=0)
        print(f"Elapsed: {record['summary']['elapsed_seconds']:.2f} seconds")
        print(f"Peak chunk memory: {bytes_label(peak)}")
    return exit_code


def _run_record(
    run_id: str,
    args: argparse.Namespace,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    elapsed: float,
    chunks: list[dict],
    files: list[Path],
    details: dict,
    error: str | None,
) -> dict:
    metadata = details.get("metadata", {})
    try:
        source_size = args.video.stat().st_size
    except OSError:
        source_size = None
    peak = max((item.get("peak_memory_bytes", 0) for item in chunks), default=0) or None
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "source": {
            "name": args.video.name,
            "path": str(args.video.resolve()),
            "size_bytes": source_size,
            "size_human": bytes_label(source_size),
        },
        "video": {
            "duration_seconds": metadata.get("duration"),
            "engine": metadata.get("engine", "whisper.cpp"),
            "acceleration": metadata.get("acceleration", "cpu" if args.cpu_only else "metal"),
            "model": str(args.model_path.resolve()),
            "chunk_seconds": args.chunk_minutes * 60,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "chunks_planned": metadata.get("chunk_count"),
        },
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "summary": {
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak,
            "peak_memory_human": bytes_label(peak),
            "chunks_processed": sum(item.get("status") == "completed" for item in chunks),
            "segments_created": details.get("segments_created"),
            "chapters_created": details.get("chapters_created"),
        },
        "chunks": chunks,
        "outputs": [str(path.resolve()) for path in files],
    }
    if error:
        record["error"] = error
    return record


if __name__ == "__main__":
    raise SystemExit(main())
