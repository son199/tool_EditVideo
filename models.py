from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

@dataclass
class Segment:
    start: float
    end: float
    text: str

@dataclass
class Scene:
    index: int
    image_path: Path
    script: str
    start: float
    end: float
    duration: float
    zoom_kind: Literal["in", "out"]
    zoom_amount: Optional[float] = None      # per-scene override; None = dùng global
    pan: Optional[str] = None                # per-scene override; None = dùng global
    subtitle_enabled: bool = True             # tắt sub cho scene cụ thể
    transition_in: Optional[str] = None       # transition trước scene này; None = global mix
