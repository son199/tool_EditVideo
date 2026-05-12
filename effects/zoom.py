"""Ken Burns effect: zoom (in/out) + pan (5 directions).

Approach: pre-scale image lớn hơn canvas một chút để có room cho cả zoom và
pan. Mỗi frame compute viewport rectangle trong base coords, crop, resize về
canvas.

Pan semantics ("camera motion direction"):
- "none": center, không pan
- "left-right": camera quét sang phải (chủ thể từ trái sang phải trong khung)... wait
  thực ra: nội dung trôi qua khung. "left-right" = bắt đầu nhìn về phía
  TRÁI của ảnh, kết thúc nhìn về PHẢI → camera quét sang phải
- "right-left": camera quét sang trái
- "up-down": camera quét xuống (bắt đầu nhìn trên, kết thúc nhìn dưới)
- "down-up": camera quét lên
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_PAN_MARGIN = 0.20  # 20% extra so cả khi zoom = max van con room pan


def _prepare_base_cv2(image_path: Path, width: int, height: int, max_zoom: float, has_pan: bool) -> np.ndarray:
    """Pre-scale ảnh dùng OpenCV để cover canvas với room cho zoom + pan."""
    pan_factor = (1.0 + _PAN_MARGIN) if has_pan else 1.0
    target_w = int(round(width * max_zoom * pan_factor))
    target_h = int(round(height * max_zoom * pan_factor))

    img = cv2.imread(str(image_path))
    if img is None:
        # fallback PIL nếu cv2 không đọc được
        pil = Image.open(str(image_path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    ih, iw = img.shape[:2]
    scale = max(target_w / iw, target_h / ih)
    new_w = max(int(round(iw * scale)), target_w)
    new_h = max(int(round(ih * scale)), target_h)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to (target_w, target_h)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img[top:top + target_h, left:left + target_w]

    # cv2 BGR → RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_PAN_TABLE = {
    "left-right": (-1.0, 1.0, "x"),
    "right-left": (1.0, -1.0, "x"),
    "up-down":    (-1.0, 1.0, "y"),
    "down-up":    (1.0, -1.0, "y"),
}


def _pan_offset(pan: str, progress: float, max_x: float, max_y: float) -> tuple[float, float]:
    entry = _PAN_TABLE.get(pan)
    if entry is None:
        return 0.0, 0.0
    start, end, axis = entry
    factor = start + (end - start) * progress
    if axis == "x":
        return max_x * factor, 0.0
    return 0.0, max_y * factor


def _zoom_at(progress: float, zoom_kind: str, zoom_start: float, zoom_end: float) -> float:
    if zoom_kind == "in":
        return zoom_start + (zoom_end - zoom_start) * progress
    return zoom_end - (zoom_end - zoom_start) * progress


def ken_burns(
    image_path: Path | str,
    width: int,
    height: int,
    duration: float,
    zoom_kind: str = "in",
    zoom_start: float = 1.0,
    zoom_end: float = 1.2,
    pan: str = "none",
    fps: int = 30,
):
    has_pan = pan not in (None, "", "none")
    max_zoom = max(zoom_start, zoom_end, 1.0)
    base = _prepare_base_cv2(Path(image_path), width, height, max_zoom, has_pan)
    base_h, base_w = base.shape[:2]
    cx0 = base_w / 2.0
    cy0 = base_h / 2.0

    n_frames = max(1, int(round(duration * fps)))
    frames: list[np.ndarray] = []

    for fi in range(n_frames):
        progress = fi / (n_frames - 1) if n_frames > 1 else 1.0
        z = _zoom_at(progress, zoom_kind, zoom_start, zoom_end)

        sw = min(max(int(round(width * z)), 2), base_w)
        sh = min(max(int(round(height * z)), 2), base_h)
        max_x_off = (base_w - sw) / 2.0
        max_y_off = (base_h - sh) / 2.0

        dx, dy = _pan_offset(pan, progress, max_x_off, max_y_off)
        cx, cy = cx0 + dx, cy0 + dy

        x0 = max(0, min(int(round(cx - sw / 2)), base_w - sw))
        y0 = max(0, min(int(round(cy - sh / 2)), base_h - sh))

        cropped = base[y0:y0 + sh, x0:x0 + sw]
        if (sw, sh) != (width, height):
            cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(cropped)

    return frames  # list[np.ndarray], shape (H, W, 3), dtype uint8
