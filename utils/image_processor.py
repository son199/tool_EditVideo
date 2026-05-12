import re
from pathlib import Path

_NUM_RE = re.compile(r"(\d+)")
_SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}


def _scene_num(path: Path) -> int:
    m = _NUM_RE.search(path.stem)
    return int(m.group(1)) if m else 10**9


def list_images(images_dir: Path | str, scene_count: int) -> list[Path]:
    dir_ = Path(images_dir)
    if not dir_.is_dir():
        raise ValueError(f"Images directory not found: {dir_}")
    candidates = [p for p in dir_.iterdir() if p.suffix.lower() in _SUPPORTED]
    candidates.sort(key=_scene_num)
    if len(candidates) < scene_count:
        raise ValueError(
            f"Cần {scene_count} ảnh trong {dir_}, chỉ tìm thấy {len(candidates)}"
        )
    return candidates[:scene_count]
