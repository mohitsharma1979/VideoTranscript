"""Compare benchmark transcript agreement without treating either engine as ground truth."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NAMES = ("beginning", "middle", "end")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def main() -> None:
    comparisons = []
    for name in NAMES:
        faster = json.loads((RESULTS / f"faster-{name}.json").read_text())
        cpp = json.loads((RESULTS / f"whisper-{name}.json").read_text())
        faster_text = " ".join(segment["text"] for segment in faster["segments"])
        cpp_text = " ".join(segment["text"] for segment in cpp["transcription"])
        faster_words, cpp_words = words(faster_text), words(cpp_text)
        distance = edit_distance(faster_words, cpp_words)
        comparisons.append({
            "sample": name,
            "faster_words": len(faster_words),
            "whisper_cpp_words": len(cpp_words),
            "word_count_difference_percent": round(abs(len(faster_words) - len(cpp_words)) / max(len(faster_words), 1) * 100, 3),
            "word_sequence_similarity": round(SequenceMatcher(None, faster_words, cpp_words, autojunk=False).ratio(), 6),
            "symmetric_word_edit_rate": round(distance / max(len(faster_words), len(cpp_words), 1), 6),
            "faster_segments": len(faster["segments"]),
            "whisper_cpp_segments": len(cpp["transcription"]),
        })
    output = {"note": "Agreement metrics compare engines; neither transcript is ground truth.", "samples": comparisons}
    (RESULTS / "agreement-summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
