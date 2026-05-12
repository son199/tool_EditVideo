"""Transition effects giữa các clip.

Convention: tham số `direction` = HƯỚNG CHUYỂN ĐỘNG. Ví dụ direction="left"
nghĩa là content trôi sang trái (B vào từ phải).

Mỗi transition function nhận clip_b (clip mới), duration của transition và
canvas_size, trả về clip_b đã modify. Khi đặt overlap `duration` giây với
clip_a trong CompositeVideoClip sẽ tạo hiệu ứng.
"""
import random

import numpy as np
from PIL import Image, ImageFilter
from moviepy.video.VideoClip import VideoClip


# ---------- Fade ----------
def _fade(clip_b, dur, canvas_size):
    return clip_b.crossfadein(dur)


# ---------- Slide (4 directions) ----------
def _slide(clip_b, dur, canvas_size, direction: str):
    w, h = canvas_size

    def pos_fn(t):
        if t >= dur:
            return (0, 0)
        p = t / dur
        if direction == "left":
            return (int(w * (1 - p)), 0)
        if direction == "right":
            return (int(-w * (1 - p)), 0)
        if direction == "up":
            return (0, int(h * (1 - p)))
        return (0, int(-h * (1 - p)))  # down

    return clip_b.set_position(pos_fn)


def _slide_left(clip_b, dur, canvas_size):
    return _slide(clip_b, dur, canvas_size, "left")


def _slide_right(clip_b, dur, canvas_size):
    return _slide(clip_b, dur, canvas_size, "right")


def _slide_up(clip_b, dur, canvas_size):
    return _slide(clip_b, dur, canvas_size, "up")


def _slide_down(clip_b, dur, canvas_size):
    return _slide(clip_b, dur, canvas_size, "down")


def _slide_random(clip_b, dur, canvas_size):
    return _slide(clip_b, dur, canvas_size, random.choice(["left", "right", "up", "down"]))


# ---------- Wipe (4 directions) ----------
def _wipe(clip_b, dur, canvas_size, direction: str):
    w, h = canvas_size

    def make_mask(t):
        if t >= dur:
            return np.ones((h, w), dtype=np.float32)
        p = t / dur
        mask = np.zeros((h, w), dtype=np.float32)
        if direction in ("left", "right"):
            edge = int(p * w)
            if direction == "left":
                mask[:, w - edge:] = 1.0   # B từ phải lan sang trái
            else:
                mask[:, :edge] = 1.0       # B từ trái lan sang phải
        else:
            edge = int(p * h)
            if direction == "up":
                mask[h - edge:, :] = 1.0   # B từ dưới lan lên trên
            else:
                mask[:edge, :] = 1.0       # B từ trên lan xuống dưới
        return mask

    mask_clip = VideoClip(make_mask, duration=clip_b.duration, ismask=True)
    return clip_b.set_mask(mask_clip)


def _wipe_left(clip_b, dur, canvas_size):
    return _wipe(clip_b, dur, canvas_size, "left")


def _wipe_right(clip_b, dur, canvas_size):
    return _wipe(clip_b, dur, canvas_size, "right")


def _wipe_up(clip_b, dur, canvas_size):
    return _wipe(clip_b, dur, canvas_size, "up")


def _wipe_down(clip_b, dur, canvas_size):
    return _wipe(clip_b, dur, canvas_size, "down")


def _wipe_random(clip_b, dur, canvas_size):
    return _wipe(clip_b, dur, canvas_size, random.choice(["left", "right", "up", "down"]))


# ---------- Wipe radial (iris from center) ----------
def _wipe_radial(clip_b, dur, canvas_size):
    w, h = canvas_size
    max_r = float(np.hypot(w, h) / 2.0)
    yy, xx = np.mgrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)

    def make_mask(t):
        if t >= dur:
            return np.ones((h, w), dtype=np.float32)
        r = (t / dur) * max_r
        return (dist <= r).astype(np.float32)

    mask_clip = VideoClip(make_mask, duration=clip_b.duration, ismask=True)
    return clip_b.set_mask(mask_clip)


# ---------- Zoom transitions ----------
def _zoom_in_trans(clip_b, dur, canvas_size):
    """B bắt đầu phóng to (1.3×) và co lại về 1.0× kết hợp fadein."""
    def filter_fn(get_frame, t):
        if t >= dur:
            return get_frame(t)
        progress = t / dur
        scale = 1.3 - 0.3 * progress
        frame = get_frame(t)
        fh, fw = frame.shape[:2]
        new_w = max(int(fw * scale), 1)
        new_h = max(int(fh * scale), 1)
        scaled = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - fw) // 2
        top = (new_h - fh) // 2
        return np.array(scaled.crop((left, top, left + fw, top + fh)))

    return clip_b.fl(filter_fn).crossfadein(dur)


def _zoom_out_trans(clip_b, dur, canvas_size):
    """B bắt đầu thu nhỏ (0.7× với viền đen), lớn dần về 1.0× kèm fadein."""
    def filter_fn(get_frame, t):
        if t >= dur:
            return get_frame(t)
        progress = t / dur
        scale = 0.7 + 0.3 * progress
        frame = get_frame(t)
        fh, fw = frame.shape[:2]
        new_w = max(int(fw * scale), 1)
        new_h = max(int(fh * scale), 1)
        scaled = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS))
        out = np.zeros_like(frame)
        ox = (fw - new_w) // 2
        oy = (fh - new_h) // 2
        out[oy:oy + new_h, ox:ox + new_w] = scaled
        return out

    return clip_b.fl(filter_fn).crossfadein(dur)


# ---------- Zoom blur (existing) ----------
def _zoom_blur(clip_b, dur, canvas_size):
    def filter_fn(get_frame, t):
        frame = get_frame(t)
        if t >= dur:
            return frame
        radius = (1.0 - t / dur) * 20.0
        if radius < 0.5:
            return frame
        img = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius))
        return np.array(img)

    return clip_b.fl(filter_fn).crossfadein(dur)


_REGISTRY = {
    "fade": _fade,
    "slide": _slide_random,
    "slide_left": _slide_left,
    "slide_right": _slide_right,
    "slide_up": _slide_up,
    "slide_down": _slide_down,
    "wipe": _wipe_random,
    "wipe_left": _wipe_left,
    "wipe_right": _wipe_right,
    "wipe_up": _wipe_up,
    "wipe_down": _wipe_down,
    "wipe_radial": _wipe_radial,
    "zoom_in": _zoom_in_trans,
    "zoom_out": _zoom_out_trans,
    "zoom_blur": _zoom_blur,
}


def apply_transition(clip_b, name: str, duration: float, canvas_size: tuple[int, int]):
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown transition: {name!r}. Available: {sorted(_REGISTRY)}")
    return fn(clip_b, duration, canvas_size)


def pick(transitions: list[str], mode: str = "mix", rng: random.Random | None = None) -> str:
    if not transitions:
        raise ValueError("transitions list empty")
    if mode == "fixed":
        return transitions[0]
    r = rng or random
    return r.choice(transitions)
