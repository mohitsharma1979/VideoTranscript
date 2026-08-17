"""Benchmark the project's current faster-whisper configuration."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
RESULTS = ROOT / "results"
FILES = [AUDIO / "beginning.wav", AUDIO / "middle.wav", AUDIO / "end.wav"]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_started = time.perf_counter()
    model = WhisperModel("small", device="auto", compute_type="default")
    model_load_seconds = time.perf_counter() - model_started
    runs = []

    for audio in FILES:
        run_started = time.perf_counter()
        raw_segments, info = model.transcribe(
            str(audio), language=None, vad_filter=True, beam_size=5
        )
        segments = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in raw_segments
            if s.text.strip()
        ]
        elapsed = time.perf_counter() - run_started
        payload = {
            "engine": "faster-whisper",
            "file": audio.name,
            "elapsed_seconds": elapsed,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
        }
        (RESULTS / f"faster-{audio.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        runs.append({key: value for key, value in payload.items() if key != "segments"})
        print(f"{audio.name}: {elapsed:.3f}s, {len(segments)} segments", flush=True)

    summary = {
        "engine": "faster-whisper",
        "model": "small",
        "configuration": {"device": "auto", "compute_type": "default", "beam_size": 5, "vad": True},
        "model_load_seconds": model_load_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runs": runs,
    }
    (RESULTS / "faster-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
