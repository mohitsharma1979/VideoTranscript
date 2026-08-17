from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def bytes_label(value: int | None) -> str | None:
    if value is None:
        return None
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            break
        amount /= 1024
    return f"{amount:.2f} {unit}"


def write_run_log(log_dir: Path, record: dict[str, Any]) -> Path:
    """Write one detailed record and append a compact JSON Lines index entry."""
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = datetime.fromisoformat(record["completed_at"])
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(record["source"]["name"]).stem)
    filename = f"{completed:%Y%m%dT%H%M%S%z}-{safe_stem}.json"
    path = log_dir / filename
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "run_id": record["run_id"],
        "status": record["status"],
        "file": record["source"]["name"],
        "size_bytes": record["source"].get("size_bytes"),
        "video_duration_seconds": record.get("video", {}).get("duration_seconds"),
        "chunks_processed": record.get("summary", {}).get("chunks_processed", 0),
        "chapters_created": record.get("summary", {}).get("chapters_created"),
        "elapsed_seconds": record.get("summary", {}).get("elapsed_seconds"),
        "peak_memory_bytes": record.get("summary", {}).get("peak_memory_bytes"),
        "completed_at": record["completed_at"],
        "log_file": filename,
    }
    with (log_dir / "index.jsonl").open("a", encoding="utf-8") as index:
        index.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return path
