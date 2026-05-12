import json
from pathlib import Path
from typing import Any


def load_scenes(path: Path | str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"scenes.json must be a JSON array, got {type(data).__name__}")
    for i, scene in enumerate(data):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene at index {i} is not an object: {scene!r}")
        if "scene" not in scene:
            raise ValueError(f"Scene at index {i} missing required 'scene' field: {scene!r}")
    data.sort(key=lambda s: int(s["scene"]))
    return data
