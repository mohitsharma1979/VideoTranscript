import json
import tempfile
import unittest
from pathlib import Path

from video_transcript.run_log import bytes_label, write_run_log


class RunLogTests(unittest.TestCase):
    def test_bytes_label(self):
        self.assertEqual(bytes_label(1_048_576), "1.00 MiB")
        self.assertIsNone(bytes_label(None))

    def test_writes_detail_and_summary_index(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            record = {
                "run_id": "test-run",
                "status": "completed",
                "source": {"name": "demo video.mp4", "size_bytes": 123},
                "video": {"duration_seconds": 10},
                "summary": {
                    "chunks_processed": 1,
                    "chapters_created": 1,
                    "elapsed_seconds": 2.5,
                    "peak_memory_bytes": 1000,
                },
                "completed_at": "2026-08-10T12:00:00+09:00",
            }
            detail = write_run_log(log_dir, record)
            self.assertTrue(detail.exists())
            index = [json.loads(line) for line in (log_dir / "index.jsonl").read_text().splitlines()]
            self.assertEqual(index[0]["file"], "demo video.mp4")
            self.assertEqual(index[0]["peak_memory_bytes"], 1000)
