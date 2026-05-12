from dataclasses import dataclass
from pathlib import Path

import pysrt


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _to_seconds(t) -> float:
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0


def load_srt(path: Path | str) -> list[Segment]:
    subs = pysrt.open(str(path), encoding="utf-8")
    segments = [
        Segment(
            start=_to_seconds(s.start),
            end=_to_seconds(s.end),
            text=s.text.replace("\n", " ").strip(),
        )
        for s in subs
    ]
    segments.sort(key=lambda x: x.start)
    return segments
