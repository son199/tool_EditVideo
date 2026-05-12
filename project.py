"""Project workspace management cho Scene Editor.

Mỗi project có 1 thư mục trong output/_projects/<project_id>/ chứa:
- state.json     — scene metadata + config + dirty flags
- scenes.json    — copy của input JSON (giữ làm reference)
- voice.srt      — copy SRT
- voice.<ext>    — copy voice
- images/        — copy ảnh user upload
- scenes/        — scene_NNNN.mp4 đã render (cache)
- final.mp4      — output cuối cùng

Editor chỉ cần re-render scene dirty trong scenes/, sau đó build_final()
concat lại → tiết kiệm 80% thời gian so với render lại từ đầu.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from config import OUTPUT_DIR, RenderConfig
from utils.parse_srt import Segment


PROJECTS_DIR = OUTPUT_DIR / "_projects"
STATE_FILENAME = "state.json"


def new_project_id() -> str:
    return f"proj_{int(time.time())}"


def project_path(pid: str) -> Path:
    return PROJECTS_DIR / pid


def project_files(pid: str) -> dict:
    p = project_path(pid)
    voice_glob = list(p.glob("voice.*")) if p.exists() else []
    return {
        "root":         p,
        "state":        p / STATE_FILENAME,
        "scenes_json":  p / "scenes.json",
        "voice_srt":    p / "voice.srt",
        "voice":        voice_glob[0] if voice_glob else None,
        "images_dir":   p / "images",
        "scenes_dir":   p / "scenes",
        "final_mp4":    p / "final.mp4",
    }


def init_workspace(pid: str) -> dict:
    """Tạo thư mục project + subdirs. Trả về dict đường dẫn."""
    p = project_path(pid)
    p.mkdir(parents=True, exist_ok=True)
    (p / "images").mkdir(exist_ok=True)
    (p / "scenes").mkdir(exist_ok=True)
    return project_files(pid)


def _scene_to_dict(scene) -> dict:
    return {
        "index": scene.index,
        "image_path": str(scene.image_path),
        "script": scene.script,
        "start": scene.start,
        "end": scene.end,
        "duration": scene.duration,
        "zoom_kind": scene.zoom_kind,
        "zoom_amount": scene.zoom_amount,
        "pan": scene.pan,
        "subtitle_enabled": getattr(scene, "subtitle_enabled", True),
        "transition_in": getattr(scene, "transition_in", None),
    }


def save_state(
    pid: str,
    scenes,
    segments: list[Segment],
    config: RenderConfig,
    extra: Optional[dict] = None,
):
    p = project_path(pid)
    state = {
        "project_id": pid,
        "last_modified": time.time(),
        "config": asdict(config),
        "scenes": [_scene_to_dict(s) for s in scenes],
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
        ],
    }
    if extra:
        state.update(extra)
    (p / STATE_FILENAME).write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_state(pid: str) -> Optional[dict]:
    sp = project_path(pid) / STATE_FILENAME
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_projects() -> list[dict]:
    if not PROJECTS_DIR.exists():
        return []
    items = []
    for p in PROJECTS_DIR.iterdir():
        if not p.is_dir():
            continue
        state = load_state(p.name)
        if not state:
            continue
        items.append({
            "project_id": p.name,
            "last_modified": state.get("last_modified", 0),
            "n_scenes": len(state.get("scenes", [])),
            "has_final": (p / "final.mp4").exists(),
        })
    items.sort(key=lambda x: -x["last_modified"])
    return items
