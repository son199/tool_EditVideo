"""Subtitle/caption overlay cho video.

Approach: Pillow render text → numpy RGBA → ImageClip (per_line) hoặc VideoClip
(animated sync modes). Tránh MoviePy TextClip vì cần ImageMagick.

Public API:
- `SubtitleStyle` dataclass
- `SUBTITLE_PRESETS` dict
- `FONT_OPTIONS` list
- `SYNC_MODES`, `SYNC_MODE_LABELS` (vi)
- `make_subtitle_clips(segments, canvas_size, sync_mode, style, video_duration)`
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from moviepy.video.VideoClip import VideoClip


_WIN_FONTS_DIR = Path("C:/Windows/Fonts")

_FONT_FILES = {
    "Arial":          "arial.ttf",
    "Arial Bold":     "arialbd.ttf",
    "Tahoma":         "tahoma.ttf",
    "Tahoma Bold":    "tahomabd.ttf",
    "Segoe UI":       "segoeui.ttf",
    "Segoe UI Bold":  "segoeuib.ttf",
    "Calibri":        "calibri.ttf",
    "Calibri Bold":   "calibrib.ttf",
    "Verdana":        "verdana.ttf",
}

FONT_OPTIONS = list(_FONT_FILES.keys())

SYNC_MODES = ["per_line", "typewriter", "karaoke", "phrase"]
SYNC_MODE_LABELS = {
    "per_line":   "Per-line (nguyên câu mỗi dòng SRT)",
    "typewriter": "Typewriter (ký tự hiện dần)",
    "karaoke":    "Karaoke (highlight từng từ)",
    "phrase":     "Phrase (chia theo dấu phẩy/chấm)",
}


@dataclass
class SubtitleStyle:
    font_name: str = "Arial Bold"
    font_size: int = 64
    color: tuple[int, int, int] = (255, 255, 255)
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    stroke_width: int = 5
    highlight_color: tuple[int, int, int] = (255, 230, 0)  # karaoke active word
    position: str = "bottom"          # top / middle / bottom
    y_offset_pct: float = 0.80         # tâm text ở 80% chiều cao canvas
    max_width_pct: float = 0.90        # word-wrap tại 90% chiều rộng canvas
    line_spacing: int = 10             # px giữa các dòng wrap


SUBTITLE_PRESETS = {
    "TikTok": SubtitleStyle(
        font_name="Arial Bold", font_size=64, stroke_width=5, y_offset_pct=0.80
    ),
    "Cinematic": SubtitleStyle(
        font_name="Segoe UI", font_size=48, stroke_width=2, y_offset_pct=0.92
    ),
    "Bold Caption": SubtitleStyle(
        font_name="Arial Bold", font_size=80, color=(255, 230, 0),
        stroke_width=6, y_offset_pct=0.75,
    ),
    "Minimal": SubtitleStyle(
        font_name="Calibri", font_size=42, stroke_width=1, y_offset_pct=0.93
    ),
}

SUBTITLE_PRESET_NAMES = list(SUBTITLE_PRESETS.keys())


# ---------------------------------------------------------------------------
# Helpers


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    fname = _FONT_FILES.get(name, "arial.ttf")
    path = _WIN_FONTS_DIR / fname
    if not path.exists():
        path = _WIN_FONTS_DIR / "arial.ttf"
    return ImageFont.truetype(str(path), size)


def _text_size(text: str, font: ImageFont.FreeTypeFont, stroke_width: int = 0) -> tuple[int, int]:
    bbox = font.getbbox(text, stroke_width=stroke_width)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int, stroke_width: int = 0) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        candidate = " ".join([*current, w])
        cw, _ = _text_size(candidate, font, stroke_width)
        if cw <= max_w or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def _resolve_y_offset(style: SubtitleStyle) -> float:
    if style.position == "top":
        return 0.10
    if style.position == "middle":
        return 0.50
    if style.position == "bottom":
        return style.y_offset_pct
    return style.y_offset_pct


def _render_lines_to_rgba(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    style: SubtitleStyle,
    canvas_size: tuple[int, int],
    highlight_words: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Render text lines vào canvas RGBA trong suốt. Trả về np array h×w×4 uint8.

    Nếu `highlight_words` được set (chứa các tuple `(line_idx, word_idx)`),
    word đó sẽ được vẽ bằng `style.highlight_color` thay vì `style.color`.
    """
    cw, ch = canvas_size
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    line_widths = []
    line_heights = []
    for ln in lines:
        w, h = _text_size(ln, font, style.stroke_width)
        line_widths.append(w)
        line_heights.append(h)
    total_h = sum(line_heights) + style.line_spacing * max(0, len(lines) - 1)

    y_center = int(_resolve_y_offset(style) * ch)
    y = y_center - total_h // 2

    for li, (line, lw, lh) in enumerate(zip(lines, line_widths, line_heights)):
        x = (cw - lw) // 2

        if highlight_words and any(li == hi for hi, _ in highlight_words):
            # Vẽ từng word, đổi màu nếu là active word
            words = line.split(" ")
            space_w, _ = _text_size(" ", font, style.stroke_width)
            cur_x = x
            for wi, word in enumerate(words):
                ww, _ = _text_size(word, font, style.stroke_width)
                is_active = (li, wi) in highlight_words
                fill = style.highlight_color if is_active else style.color
                draw.text(
                    (cur_x, y),
                    word,
                    font=font,
                    fill=fill,
                    stroke_width=style.stroke_width,
                    stroke_fill=style.stroke_color,
                )
                cur_x += ww + space_w
        else:
            draw.text(
                (x, y),
                line,
                font=font,
                fill=style.color,
                stroke_width=style.stroke_width,
                stroke_fill=style.stroke_color,
            )
        y += lh + style.line_spacing

    return np.array(img)


def _split_phrases(text: str) -> list[str]:
    """Chia câu theo dấu phẩy/chấm/chấm hỏi. Giữ punctuation."""
    parts = re.split(r"(?<=[,.!?;:])\s+", text.strip())
    return [p for p in (p.strip() for p in parts) if p]


def _rgba_to_clip(make_rgba, duration: float, canvas_size: tuple[int, int]):
    """Wrap make_rgba(t) -> RGBA np array thành (color clip + mask clip).

    Có cache (last_t, last_img) trong closure để 2 lần gọi
    (color_frame và mask_frame) cùng t chỉ tính Pillow 1 lần.
    """
    cw, ch = canvas_size
    cache = {"t": None, "img": None}

    def _get(t):
        if cache["t"] != t:
            cache["t"] = t
            cache["img"] = make_rgba(t)
        return cache["img"]

    def color_frame(t):
        return _get(t)[:, :, :3]

    def mask_frame(t):
        return _get(t)[:, :, 3].astype(np.float32) / 255.0

    color = VideoClip(color_frame, duration=duration)
    color.size = (cw, ch)
    mask = VideoClip(mask_frame, duration=duration, ismask=True)
    mask.size = (cw, ch)
    return color.set_mask(mask)


# ---------------------------------------------------------------------------
# Builders


def _build_per_line(text: str, canvas_size: tuple[int, int], style: SubtitleStyle, duration: float):
    font = _load_font(style.font_name, style.font_size)
    cw, _ = canvas_size
    max_w = int(cw * style.max_width_pct)
    lines = _wrap_text(text, font, max_w, style.stroke_width)
    rgba = _render_lines_to_rgba(lines, font, style, canvas_size)
    return ImageClip(rgba, transparent=True).set_duration(duration)


def _build_typewriter(text: str, canvas_size: tuple[int, int], style: SubtitleStyle, duration: float):
    font = _load_font(style.font_name, style.font_size)
    cw, _ = canvas_size
    max_w = int(cw * style.max_width_pct)
    full_lines = _wrap_text(text, font, max_w, style.stroke_width)
    full_text = "\n".join(full_lines)
    n_chars = len(full_text)

    def make_rgba(t: float):
        progress = min(max(t / duration, 0.0), 1.0)
        n_show = max(1, int(round(n_chars * progress)))
        partial = full_text[:n_show]
        partial_lines = partial.split("\n")
        return _render_lines_to_rgba(partial_lines, font, style, canvas_size)

    return _rgba_to_clip(make_rgba, duration, canvas_size)


def _build_phrase(text: str, canvas_size: tuple[int, int], style: SubtitleStyle, duration: float):
    """Chia câu thành phrases, mỗi phrase hiện trong (duration / n) giây."""
    phrases = _split_phrases(text) or [text]
    n = len(phrases)
    font = _load_font(style.font_name, style.font_size)
    cw, _ = canvas_size
    max_w = int(cw * style.max_width_pct)
    rendered = [
        _render_lines_to_rgba(_wrap_text(p, font, max_w, style.stroke_width), font, style, canvas_size)
        for p in phrases
    ]

    def make_rgba(t: float):
        progress = min(max(t / duration, 0.0), 0.9999)
        idx = int(progress * n)
        return rendered[idx]

    return _rgba_to_clip(make_rgba, duration, canvas_size)


def _build_karaoke(text: str, canvas_size: tuple[int, int], style: SubtitleStyle, duration: float):
    """Highlight từng từ theo thời gian (chia đều)."""
    font = _load_font(style.font_name, style.font_size)
    cw, _ = canvas_size
    max_w = int(cw * style.max_width_pct)
    lines = _wrap_text(text, font, max_w, style.stroke_width)
    word_positions: list[tuple[int, int]] = []
    for li, ln in enumerate(lines):
        for wi, _ in enumerate(ln.split(" ")):
            word_positions.append((li, wi))
    n_words = max(1, len(word_positions))

    def make_rgba(t: float):
        progress = min(max(t / duration, 0.0), 0.9999)
        active_idx = min(int(progress * n_words), n_words - 1)
        highlight = {word_positions[active_idx]}
        return _render_lines_to_rgba(lines, font, style, canvas_size, highlight_words=highlight)

    return _rgba_to_clip(make_rgba, duration, canvas_size)


_BUILDERS = {
    "per_line":   _build_per_line,
    "typewriter": _build_typewriter,
    "phrase":     _build_phrase,
    "karaoke":    _build_karaoke,
}


# ---------------------------------------------------------------------------
# Top-level


def make_subtitle_clips(
    segments,
    canvas_size: tuple[int, int],
    sync_mode: str,
    style: SubtitleStyle,
    video_duration: float,
) -> list:
    """Return a list containing a single VideoClip that handles all subtitles.
    
    Approach: Thay vì trả về hàng trăm clips (làm CompositeVideoClip cực chậm), 
    ta trả về 1 VideoClip duy nhất quản lý toàn bộ segments.
    """
    builder = _BUILDERS.get(sync_mode)
    if builder is None:
        raise ValueError(f"Unknown sync_mode: {sync_mode!r}. Available: {sorted(_BUILDERS)}")

    # Lọc và chuẩn bị các segment clips trước (không render, chỉ build metadata)
    valid_segments = []
    for seg in segments:
        if seg.start >= video_duration:
            continue
        clip_end = min(seg.end, video_duration)
        clip_dur = clip_end - seg.start
        if clip_dur <= 0:
            continue
        text = (seg.text or "").strip()
        if not text:
            continue
        valid_segments.append({
            "start": seg.start,
            "end": clip_end,
            "text": text,
            "duration": clip_dur
        })

    if not valid_segments:
        return []

    # Cache các builders để tránh load font/wrap text lặp lại trong make_frame
    # Key: text + duration (vì duration ảnh hưởng sync modes như karaoke)
    _clip_cache = {}

    def get_rgba_at(t):
        # Tìm segment hiện tại
        active = None
        for s in valid_segments:
            if s["start"] <= t < s["end"]:
                active = s
                break
        
        if not active:
            return np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)

        # Build hoặc lấy từ cache
        cache_key = (active["text"], active["duration"])
        if cache_key not in _clip_cache:
            # Tạo clip tạm để lấy frame (builder trả về ImageClip hoặc VideoClip)
            _clip_cache[cache_key] = builder(active["text"], canvas_size, style, active["duration"])
        
        clip = _clip_cache[cache_key]
        rel_t = t - active["start"]
        
        # ImageClip.get_frame(t) trả về RGB, ta cần RGBA
        if hasattr(clip, "img"): # ImageClip
            # Thêm alpha channel từ mask nếu có
            rgb = clip.get_frame(rel_t)
            if clip.mask:
                mask = (clip.mask.get_frame(rel_t) * 255).astype(np.uint8)
                return np.dstack([rgb, mask])
            else:
                return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
        else: # VideoClip
            rgb = clip.get_frame(rel_t)
            mask = (clip.mask.get_frame(rel_t) * 255).astype(np.uint8)
            return np.dstack([rgb, mask])

    # Wrap thành 1 VideoClip duy nhất
    def color_frame(t):
        return get_rgba_at(t)[:, :, :3]

    def mask_frame(t):
        return get_rgba_at(t)[:, :, 3].astype(np.float32) / 255.0

    final_sub_clip = VideoClip(color_frame, duration=video_duration)
    final_sub_clip.size = canvas_size
    final_mask = VideoClip(mask_frame, duration=video_duration, ismask=True)
    final_mask.size = canvas_size
    final_sub_clip = final_sub_clip.set_mask(final_mask)
    
    return [final_sub_clip]
