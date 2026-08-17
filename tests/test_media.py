import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from video_transcript.media import MediaError, duration, extract_audio, validate_input


class MediaTests(unittest.TestCase):
    def test_rejects_unsupported_file(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "clip.webm"
            file.touch()
            with self.assertRaisesRegex(MediaError, "Unsupported"):
                validate_input(file)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "FFmpeg unavailable")
    def test_extracts_audio_from_synthetic_mp4(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            video = tmp_path / "synthetic.mp4"
            audio = tmp_path / "audio.wav"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video),
            ], check=True)
            extract_audio(video, audio)
            self.assertGreater(audio.stat().st_size, 1000)
            self.assertAlmostEqual(duration(video), 1.0, delta=0.1)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "FFmpeg unavailable")
    def test_extracts_bounded_audio_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            video = tmp_path / "synthetic.mp4"
            audio = tmp_path / "chunk.wav"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=3",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video),
            ], check=True)
            extract_audio(video, audio, start=1.0, length=0.5)
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
            ], capture_output=True, text=True, check=True)
            self.assertAlmostEqual(float(probe.stdout), 0.5, delta=0.1)
