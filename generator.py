import argparse
import multiprocessing
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from moviepy.audio.AudioClip import AudioClip, concatenate_audioclips
from moviepy.editor import AudioFileClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips

from config import PAN_OPTIONS, ZOOM_MODES, CODEC_FORMATS, QUALITY_PRESETS, RenderConfig, resolve_encoder
from effects.subtitle import (
    SUBTITLE_PRESETS,
    SubtitleStyle,
    make_subtitle_clips,
    FONT_OPTIONS as SUB_FONT_OPTIONS,
    SYNC_MODES as SUB_SYNC_MODES,
)
from effects.transitions import apply_transition, pick
from effects.zoom import ken_burns
from utils.image_processor import list_images
from utils.parse_json import load_scenes
from utils.parse_srt import Segment, load_srt

_PAN_DIRECTIONS_NONRANDOM = [p for p in PAN_OPTIONS if p not in ("none", "random")]


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
    subtitle_enabled: bool = True             # E7: tắt sub cho scene cụ thể
    transition_in: Optional[str] = None       # E8: transition trước scene này; None = global mix


def _resolve_pan(scene_pan: Optional[str], global_pan: str) -> str:
    """Per-scene > global. 'random' được resolve thành 1 direction cố định."""
    val = scene_pan if scene_pan else global_pan
    if not val or val == "none":
        return "none"
    if val == "random":
        return random.choice(_PAN_DIRECTIONS_NONRANDOM)
    return val


def _resolve_zoom_kind(scene_idx: int, mode: str, override: Optional[str]) -> Literal["in", "out"]:
    """Per-scene override > global mode. mode quyết định pattern toàn cục."""
    if override in ("in", "out"):
        return override  # type: ignore[return-value]
    if mode == "alternate":
        return "in" if scene_idx % 2 == 0 else "out"
    if mode == "alternate_reverse":
        return "out" if scene_idx % 2 == 0 else "in"
    if mode == "all_in":
        return "in"
    if mode == "all_out":
        return "out"
    if mode == "random":
        return random.choice(["in", "out"])
    return "in"


def align_scenes(
    scenes_json: list[dict],
    segments: list[Segment],
    images: list[Path],
    zoom_mode: str = "alternate",
) -> list[Scene]:
    n = len(scenes_json)
    if len(segments) != n:
        raise ValueError(
            f"Mismatch: JSON có {n} scenes nhưng SRT có {len(segments)} dòng"
        )
    if len(images) != n:
        raise ValueError(
            f"Mismatch: JSON có {n} scenes nhưng tìm thấy {len(images)} ảnh"
        )

    scenes: list[Scene] = []
    for i, (s_data, seg, img) in enumerate(zip(scenes_json, segments, images), start=1):
        scene_idx = int(s_data.get("scene", i))
        zoom_kind = _resolve_zoom_kind(scene_idx, zoom_mode, s_data.get("zoom_kind"))

        zoom_amount_raw = s_data.get("zoom_amount")
        zoom_amount = float(zoom_amount_raw) if zoom_amount_raw is not None else None

        pan_raw = s_data.get("pan")
        pan = str(pan_raw) if pan_raw is not None else None

        sub_enabled_raw = s_data.get("subtitle_enabled", True)
        trans_in_raw = s_data.get("transition_in")
        trans_in = str(trans_in_raw) if trans_in_raw else None

        scenes.append(
            Scene(
                index=scene_idx,
                image_path=img,
                script=str(s_data.get("script", "")),
                start=seg.start,
                end=seg.end,
                duration=max(seg.end - seg.start, 0.1),
                zoom_kind=zoom_kind,
                zoom_amount=zoom_amount,
                pan=pan,
                subtitle_enabled=bool(sub_enabled_raw),
                transition_in=trans_in,
            )
        )
    return scenes


def _scene_zoom_range(scene: Scene, config: RenderConfig) -> tuple[float, float]:
    """Trả về (zoom_start, zoom_end) cho scene, ưu tiên per-scene zoom_amount."""
    if scene.zoom_amount is None:
        return config.zoom_start, config.zoom_end
    # Per-scene override: amount = mức zoom max; start/end = 1.0 và amount theo hướng
    if scene.zoom_kind == "in":
        return 1.0, scene.zoom_amount
    return scene.zoom_amount, 1.0


def _make_kb_clip(scene: Scene, config: RenderConfig):
    zs, ze = _scene_zoom_range(scene, config)
    pan = _resolve_pan(scene.pan, config.pan_direction)
    return ken_burns(
        image_path=scene.image_path,
        width=config.width,
        height=config.height,
        duration=scene.duration,
        zoom_kind=scene.zoom_kind,
        zoom_start=zs,
        zoom_end=ze,
        pan=pan,
        fps=config.fps,
    )


def _scene_to_temp_mp4(args):
    """Worker cho multiprocessing. Render 1 scene ra MP4 ultrafast tạm."""
    scene, config, temp_dir_str = args
    clip = _make_kb_clip(scene, config)
    out = Path(temp_dir_str) / f"scene_{scene.index:04d}.mp4"
    clip.write_videofile(
        str(out),
        codec="libx264",
        audio=False,
        fps=config.fps,
        threads=2,
        preset="ultrafast",
        ffmpeg_params=["-crf", "18"],   # gần lossless để khỏi mất chất lượng trung gian
        logger=None,
        verbose=False,
    )
    return str(out)


def _scene_mp4_filename(scene_index: int) -> str:
    return f"scene_{scene_index:04d}.mp4"


def _parse_scene_index_from_filename(path: Path) -> int:
    return int(path.stem.replace("scene_", ""))


def render_scenes(
    scenes: list[Scene],
    config: RenderConfig,
    workspace_dir: Path,
    scenes_to_render: Optional[list[int]] = None,
    progress_callback=None,
) -> dict[int, Path]:
    """Render scenes ra workspace_dir/scene_NNNN.mp4. Trả về {scene.index: path}.

    Nếu `scenes_to_render` được set: chỉ re-render những scene index đó (incremental
    edit mode). MP4 cũ của các scene khác giữ nguyên trên disk.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if scenes_to_render is None:
        targets = scenes
    else:
        target_set = set(scenes_to_render)
        targets = [s for s in scenes if s.index in target_set]

    if targets:
        if config.parallel and len(targets) > 1:
            _render_targets_parallel(targets, config, workspace_dir, progress_callback)
        else:
            _render_targets_sequential(targets, config, workspace_dir, progress_callback)

    # Quét tất cả MP4 hiện có trong workspace (gồm cả cái render trước đó chưa đổi)
    return {
        _parse_scene_index_from_filename(p): p
        for p in workspace_dir.glob("scene_*.mp4")
    }


def _render_targets_sequential(targets, config, workspace_dir, progress_callback):
    for i, scene in enumerate(targets, start=1):
        _scene_to_temp_mp4((scene, config, str(workspace_dir)))
        if progress_callback:
            progress_callback(i, len(targets))


def _render_targets_parallel(targets, config, workspace_dir, progress_callback):
    n_workers = config.parallel_workers if config.parallel_workers > 0 else (os.cpu_count() or 4)
    n_workers = min(n_workers, len(targets), 32)
    args_list = [(s, config, str(workspace_dir)) for s in targets]
    completed = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(_scene_to_temp_mp4, args_list):
            completed += 1
            if progress_callback:
                progress_callback(completed, len(targets))


def _pick_transition_for_index(
    i: int, scenes: Optional[list[Scene]], config: RenderConfig
) -> str:
    """Per-scene override > global mix > fallback fade."""
    if scenes and i < len(scenes) and scenes[i].transition_in:
        return scenes[i].transition_in
    if config.transitions:
        return pick(config.transitions, config.transition_mode)
    return "fade"


def _compose_clips(clips: list, config: RenderConfig, scenes: Optional[list[Scene]] = None):
    """Áp transitions giữa các clip → CompositeVideoClip.

    Nếu `scenes` truyền và scene.transition_in set, dùng transition đó (E8).
    """
    has_any_override = any(getattr(s, "transition_in", None) for s in (scenes or []))
    use_transitions = (
        (config.transitions or has_any_override)
        and config.transition_duration > 0
        and len(clips) > 1
    )
    if not use_transitions:
        return concatenate_videoclips(clips, method="compose")

    t_dur = config.transition_duration
    canvas = (config.width, config.height)
    placed = []
    cursor = 0.0
    for i, clip in enumerate(clips):
        if i > 0:
            name = _pick_transition_for_index(i, scenes, config)
            clip = apply_transition(clip, name, t_dur, canvas)
        placed.append(clip.set_start(cursor))
        cursor += clip.duration - (t_dur if i < len(clips) - 1 else 0)

    return CompositeVideoClip(placed, size=canvas).set_duration(cursor)


def build_video(
    scenes: list[Scene],
    config: RenderConfig,
    temp_dir: Optional[Path] = None,
    parallel_progress=None,
):
    """Build composite từ scenes. Path cũ (one-shot render)."""
    if config.parallel and len(scenes) > 1 and temp_dir is not None:
        scene_paths = render_scenes(scenes, config, temp_dir, progress_callback=parallel_progress)
        clips = [VideoFileClip(str(scene_paths[s.index])) for s in scenes]
    else:
        clips = [_make_kb_clip(s, config) for s in scenes]
    return _compose_clips(clips, config, scenes=scenes)


def build_final(
    scenes: list[Scene],
    scene_paths: dict[int, Path],
    segments: list[Segment],
    voice_path: Path,
    out_path: Path,
    config: RenderConfig,
    logger=None,
) -> Path:
    """Concat scene MP4 đã cache + transitions + subtitle + audio → final MP4.

    Dùng cho editor mode: scene MP4s đã render trước, chỉ cần ghép lại nhanh.
    Filter segments theo scene.subtitle_enabled (E7).
    """
    missing = [s.index for s in scenes if s.index not in scene_paths]
    if missing:
        raise ValueError(f"Thiếu scene MP4 cho index: {missing}. Chạy render_scenes() trước.")

    clips = [VideoFileClip(str(scene_paths[s.index])) for s in scenes]
    composite = _compose_clips(clips, config, scenes=scenes)

    # E7: filter subtitle segments theo scene.subtitle_enabled
    visible_segments = [
        seg for seg, scene in zip(segments, scenes)
        if getattr(scene, "subtitle_enabled", True)
    ]
    composite = add_subtitle(composite, visible_segments, config)
    composite = add_audio(composite, voice_path)
    export(composite, out_path, config, logger=logger)
    return out_path


def add_subtitle(video, segments: list[Segment], config: RenderConfig):
    """Overlay subtitle clips lên video chính. No-op nếu subtitle_enabled=False."""
    if not config.subtitle_enabled:
        return video
    preset = SUBTITLE_PRESETS.get(config.subtitle_preset)
    if preset is None:
        preset = next(iter(SUBTITLE_PRESETS.values()))
    style = replace(
        preset,
        font_name=config.subtitle_font,
        font_size=config.subtitle_font_size,
        color=tuple(config.subtitle_color),
        stroke_color=tuple(config.subtitle_stroke_color),
        stroke_width=config.subtitle_stroke_width,
        highlight_color=tuple(config.subtitle_highlight_color),
        position=config.subtitle_position,
        y_offset_pct=config.subtitle_y_offset_pct,
    )
    sub_clips = make_subtitle_clips(
        segments,
        (config.width, config.height),
        config.subtitle_sync_mode,
        style,
        video.duration,
    )
    if not sub_clips:
        return video
    composite = CompositeVideoClip(
        [video, *sub_clips], size=(config.width, config.height)
    ).set_duration(video.duration)
    if video.audio is not None:
        composite = composite.set_audio(video.audio)
    return composite


def _silence(duration: float, fps: int = 44100, channels: int = 2) -> AudioClip:
    def make_frame(t):
        if np.isscalar(t):
            return np.zeros(channels)
        return np.zeros((len(t), channels))

    return AudioClip(make_frame=make_frame, duration=duration, fps=fps)


def add_audio(video, voice_path: Path):
    audio = AudioFileClip(str(voice_path))
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    elif audio.duration < video.duration:
        gap = video.duration - audio.duration
        pad = _silence(gap, fps=audio.fps or 44100)
        audio = concatenate_audioclips([audio, pad])
    return video.set_audio(audio)


# Quality table: cho mỗi encoder, map quality_preset → (preset_name, quality_value)
# Quality value = CRF (libx*) | CQ (nvenc) | global_quality (qsv) | QP (amf). Số nhỏ = đẹp hơn.
_QUALITY_TABLE: dict[str, dict[str, tuple[str, int]]] = {
    "libx264":    {"speed": ("veryfast",  26), "balanced": ("medium",   23), "quality": ("slow",    20), "max": ("slower", 18)},
    "libx265":    {"speed": ("veryfast",  28), "balanced": ("medium",   24), "quality": ("slow",    22), "max": ("slower", 20)},
    "h264_nvenc": {"speed": ("p1",        28), "balanced": ("p4",       24), "quality": ("p6",      20), "max": ("p7",     18)},
    "hevc_nvenc": {"speed": ("p1",        30), "balanced": ("p4",       25), "quality": ("p6",      22), "max": ("p7",     20)},
    "h264_qsv":   {"speed": ("veryfast",  26), "balanced": ("medium",   23), "quality": ("slow",    20), "max": ("slower", 18)},
    "hevc_qsv":   {"speed": ("veryfast",  28), "balanced": ("medium",   24), "quality": ("slow",    22), "max": ("slower", 20)},
    "h264_amf":   {"speed": ("speed",     28), "balanced": ("balanced", 24), "quality": ("quality", 20), "max": ("quality", 18)},
    "hevc_amf":   {"speed": ("speed",     30), "balanced": ("balanced", 25), "quality": ("quality", 22), "max": ("quality", 20)},
}


def _encoder_extra_params(encoder: str, q_value: int) -> list[str]:
    if encoder.endswith("_nvenc"):
        return ["-rc", "vbr", "-cq", str(q_value)]
    if encoder.endswith("_qsv"):
        return ["-global_quality", str(q_value)]
    if encoder.endswith("_amf"):
        return ["-rc", "cqp", "-qp_i", str(q_value), "-qp_p", str(q_value)]
    if encoder in ("libx264", "libx265"):
        return ["-crf", str(q_value)]
    return []


def export(clip, out_path: Path, config: RenderConfig, logger=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = resolve_encoder(config.codec_format, config.engine)
    quality_map = _QUALITY_TABLE.get(encoder) or _QUALITY_TABLE["libx264"]
    preset, q_val = quality_map.get(config.quality_preset, quality_map["balanced"])
    extra = _encoder_extra_params(encoder, q_val)

    clip.write_videofile(
        str(out_path),
        codec=encoder,
        audio_codec="aac",
        fps=config.fps,
        threads=os.cpu_count() or 2,
        preset=preset,
        ffmpeg_params=extra or None,
        logger=logger if logger is not None else "bar",
    )


def render_to_project(
    project_files: dict,
    config: RenderConfig,
    logger=None,
    prep_callback=None,
    parallel_progress=None,
):
    """Pipeline render vào project workspace persistent (cho Scene Editor).

    Đọc inputs từ project_files dict (do project.init_workspace() trả về),
    render scenes ra project_files['scenes_dir']/scene_NNNN.mp4, build final
    ra project_files['final_mp4']. Trả về (scenes, segments, scene_paths).

    Caller (app.py) chịu trách nhiệm save_state() sau khi thành công.
    """
    if prep_callback:
        prep_callback("parse")
    scenes_json = load_scenes(project_files["scenes_json"])
    segments = load_srt(project_files["voice_srt"])
    images = list_images(project_files["images_dir"], len(scenes_json))
    scenes = align_scenes(scenes_json, segments, images, zoom_mode=config.zoom_mode)

    if prep_callback:
        use_par = config.parallel and len(scenes) > 1
        prep_callback("parallel" if use_par else "build")
    scene_paths = render_scenes(
        scenes, config,
        project_files["scenes_dir"],
        progress_callback=parallel_progress,
    )

    if prep_callback:
        prep_callback("encode")
    build_final(
        scenes, scene_paths, segments,
        project_files["voice"],
        project_files["final_mp4"],
        config,
        logger=logger,
    )

    return scenes, segments, scene_paths


def quick_render(
    json_path: str | Path,
    srt_path: str | Path,
    images_dir: str | Path,
    voice_path: str | Path,
    out_path: str | Path,
    config: RenderConfig | None = None,
    logger=None,
    prep_callback=None,
    parallel_progress=None,
) -> Path:
    config = config or RenderConfig()
    if prep_callback:
        prep_callback("parse")
    scenes_json = load_scenes(json_path)
    segments = load_srt(srt_path)
    images = list_images(images_dir, len(scenes_json))
    scenes = align_scenes(scenes_json, segments, images, zoom_mode=config.zoom_mode)

    use_parallel = config.parallel and len(scenes) > 1
    parallel_temp = Path(tempfile.mkdtemp(prefix="autoedit_par_")) if use_parallel else None

    try:
        if prep_callback:
            prep_callback("parallel" if use_parallel else "build")
        video = build_video(
            scenes,
            config,
            temp_dir=parallel_temp,
            parallel_progress=parallel_progress,
        )
        video = add_subtitle(video, segments, config)
        video = add_audio(video, Path(voice_path))
        if prep_callback:
            prep_callback("encode")
        out_path = Path(out_path)
        export(video, out_path, config, logger=logger)
    finally:
        if parallel_temp is not None:
            shutil.rmtree(parallel_temp, ignore_errors=True)
    return out_path


def _main():
    p = argparse.ArgumentParser(description="Auto Video Editor CLI")
    p.add_argument("--json", required=True)
    p.add_argument("--srt", required=True)
    p.add_argument("--images", required=True, help="Thư mục chứa 1.png, 2.png, ...")
    p.add_argument("--voice", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--zoom-mode", choices=ZOOM_MODES, default="alternate")
    p.add_argument("--zoom-start", type=float, default=1.0)
    p.add_argument("--zoom-end", type=float, default=1.2)
    p.add_argument("--pan", choices=PAN_OPTIONS, default="none")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--transitions",
        default="fade,slide",
        help="Comma list từ AVAILABLE_TRANSITIONS (empty = no transitions)",
    )
    p.add_argument("--transition-mode", choices=["fixed", "mix"], default="mix")
    p.add_argument("--transition-duration", type=float, default=0.5)
    p.add_argument("--subtitle", action="store_true", help="Bật subtitle overlay")
    p.add_argument("--subtitle-preset", default="TikTok",
                   choices=list(SUBTITLE_PRESETS.keys()))
    p.add_argument("--subtitle-sync", default="per_line", choices=SUB_SYNC_MODES)
    p.add_argument("--subtitle-font", default="Arial Bold", choices=SUB_FONT_OPTIONS)
    p.add_argument("--subtitle-size", type=int, default=64)
    p.add_argument("--subtitle-position", default="bottom", choices=["top", "middle", "bottom"])
    p.add_argument("--codec", default="h264", choices=CODEC_FORMATS,
                   help="h264 (compatible) | hevc (chất lượng cao)")
    p.add_argument("--engine", default="cpu", choices=["cpu", "nvidia", "amd", "intel"],
                   help="cpu = libx264/x265, hoặc GPU brand tương ứng")
    p.add_argument("--quality", default="balanced", choices=QUALITY_PRESETS,
                   help="speed | balanced | quality | max")
    p.add_argument("--parallel", action="store_true",
                   help="Render scene song song trên nhiều core")
    p.add_argument("--workers", type=int, default=0,
                   help="Số worker process (0 = auto theo cpu_count)")
    args = p.parse_args()

    transitions = [t.strip() for t in args.transitions.split(",") if t.strip()]
    cfg = RenderConfig(
        width=args.width,
        height=args.height,
        zoom_mode=args.zoom_mode,
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end,
        pan_direction=args.pan,
        fps=args.fps,
        transitions=transitions,
        transition_mode=args.transition_mode,
        transition_duration=args.transition_duration,
        subtitle_enabled=args.subtitle,
        subtitle_preset=args.subtitle_preset,
        subtitle_sync_mode=args.subtitle_sync,
        subtitle_font=args.subtitle_font,
        subtitle_font_size=args.subtitle_size,
        subtitle_position=args.subtitle_position,
        codec_format=args.codec,
        engine=args.engine,
        quality_preset=args.quality,
        parallel=args.parallel,
        parallel_workers=args.workers,
    )
    out = quick_render(args.json, args.srt, args.images, args.voice, args.out, cfg)
    print(f"OK -> {out}")


if __name__ == "__main__":
    _main()
