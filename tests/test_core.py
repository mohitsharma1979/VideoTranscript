import json
import tempfile
import unittest
from pathlib import Path

from video_transcript.chapters import build_chapters
from video_transcript.models import Segment
from video_transcript.output import timestamp, write_outputs


class CoreTests(unittest.TestCase):
    def test_timestamp_formats(self):
        self.assertEqual(timestamp(3661.234), "01:01:01.234")
        self.assertEqual(timestamp(1.2, ","), "00:00:01,200")

    def test_chapters_split_at_pause_after_target(self):
        segments = [
            Segment(0, 10, "Welcome to the video."),
            Segment(10, 70, "This is the first topic."),
            Segment(74, 85, "Now we discuss the second topic."),
        ]
        chapters = build_chapters(segments, target_seconds=60, pause_seconds=2.5)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[1].start, 74)
        self.assertEqual(chapters[1].title, "Now we discuss the second topic")

    def test_writes_all_output_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "demo.mp4"
            source.touch()
            segments = [Segment(0, 1.5, "Hello world."), Segment(2, 3, "Goodbye.")]
            chapters = build_chapters(segments)
            files = write_outputs(tmp_path / "out", source, segments, chapters, {"language": "en"})
            self.assertEqual({path.suffix for path in files}, {".md", ".json", ".srt", ".vtt"})
            data = json.loads((tmp_path / "out/demo.transcript.json").read_text())
            self.assertEqual(data["segments"][0]["text"], "Hello world.")
