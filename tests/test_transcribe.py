import json
import sys
import tempfile
import unittest
from pathlib import Path

from video_transcript.transcribe import TranscriptionError, _run_monitored, _segments_from_json


class TranscriptionTests(unittest.TestCase):
    def test_parses_whisper_cpp_json_with_global_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text(json.dumps({
                "transcription": [
                    {"text": " Hello world. ", "offsets": {"from": 250, "to": 1750}}
                ]
            }))
            segments = _segments_from_json(output, audio_offset=900)
            self.assertEqual(segments[0].text, "Hello world.")
            self.assertEqual(segments[0].start, 900.25)
            self.assertEqual(segments[0].end, 901.75)

    def test_rejects_invalid_whisper_cpp_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("not-json")
            with self.assertRaisesRegex(TranscriptionError, "invalid JSON"):
                _segments_from_json(output, audio_offset=0)

    def test_monitors_child_peak_memory(self):
        command = [
            sys.executable,
            "-c",
            "import time; data=bytearray(20_000_000); time.sleep(0.2)",
        ]
        returncode, stderr, peak = _run_monitored(command, interval=0.01)
        self.assertEqual(returncode, 0)
        self.assertEqual(stderr, "")
        self.assertGreater(peak, 10_000_000)
