from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PanDirection = Literal["none", "left-right", "right-left", "up-down", "down-up", "random"]
ZoomMode = Literal["alternate", "alternate_reverse", "all_in", "all_out", "random"]
SubtitlePosition = Literal["top", "middle", "bottom"]
SubtitleSyncMode = Literal["per_line", "typewriter", "karaoke", "phrase"]
CodecFormat = Literal["h264", "hevc"]
EngineKind = Literal["cpu", "nvidia", "amd", "intel"]
QualityPreset = Literal["speed", "balanced", "quality", "max"]


ENCODER_MATRIX: dict[tuple[str, str], str] = {
    ("h264", "cpu"):    "libx264",
    ("h264", "nvidia"): "h264_nvenc",
    ("h264", "amd"):    "h264_amf",
    ("h264", "intel"):  "h264_qsv",
    ("hevc", "cpu"):    "libx265",
    ("hevc", "nvidia"): "hevc_nvenc",
    ("hevc", "amd"):    "hevc_amf",
    ("hevc", "intel"):  "hevc_qsv",
}


def resolve_encoder(codec_format: str, engine: str) -> str:
    return ENCODER_MATRIX.get((codec_format, engine), "libx264")


def _ffmpeg_has_encoder(encoder: str) -> bool:
    """Kiểm tra FFmpeg có hỗ trợ encoder này không (runtime check)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return encoder in out
    except Exception:
        return False


_nvenc_probe_result: bool | None = None


def probe_nvenc_works() -> bool:
    """Thử encode 1 frame bằng h264_nvenc để xác nhận GPU thực sự hoạt động.
    Kết quả được cache — chỉ probe một lần per process.
    """
    global _nvenc_probe_result
    if _nvenc_probe_result is not None:
        return _nvenc_probe_result

    import subprocess, tempfile, os
    # Tạo 1-frame black video bằng h264_nvenc
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp = f.name
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:size=128x72:rate=1:duration=1",
                "-vframes", "1", "-c:v", "h264_nvenc", "-y", tmp,
            ],
            capture_output=True, timeout=15,
        )
        _nvenc_probe_result = (result.returncode == 0)
    except Exception:
        _nvenc_probe_result = False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return _nvenc_probe_result


@dataclass
class RenderConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # Encoder
    codec_format: CodecFormat = "h264"
    engine: EngineKind = "cpu"
    quality_preset: QualityPreset = "balanced"
    # Parallelization
    parallel: bool = False
    parallel_workers: int = 0   # 0 = auto (cpu_count, capped at scenes count)
    zoom_mode: ZoomMode = "alternate"
    zoom_start: float = 1.0
    zoom_end: float = 1.2
    pan_direction: PanDirection = "none"
    transitions: list[str] = field(default_factory=lambda: ["fade", "slide"])
    transition_mode: Literal["fixed", "mix"] = "mix"
    transition_duration: float = 0.5
    # Subtitle
    subtitle_enabled: bool = False
    subtitle_preset: str = "TikTok"
    subtitle_sync_mode: SubtitleSyncMode = "per_line"
    subtitle_font: str = "Arial Bold"
    subtitle_font_size: int = 64
    subtitle_color: tuple = (255, 255, 255)
    subtitle_stroke_color: tuple = (0, 0, 0)
    subtitle_stroke_width: int = 5
    subtitle_highlight_color: tuple = (255, 230, 0)
    subtitle_position: SubtitlePosition = "bottom"
    subtitle_y_offset_pct: float = 0.80


ASPECT_PRESETS = {
    "9:16 (1080x1920)": (1080, 1920),
    "16:9 (1920x1080)": (1920, 1080),
    "1:1 (1080x1080)": (1080, 1080),
}

AVAILABLE_TRANSITIONS = [
    "fade",
    "slide", "slide_left", "slide_right", "slide_up", "slide_down",
    "wipe", "wipe_left", "wipe_right", "wipe_up", "wipe_down",
    "wipe_radial",
    "zoom_in", "zoom_out", "zoom_blur",
]

PAN_OPTIONS: list[PanDirection] = ["none", "left-right", "right-left", "up-down", "down-up", "random"]

ZOOM_MODES: list[ZoomMode] = ["alternate", "alternate_reverse", "all_in", "all_out", "random"]

ZOOM_MODE_LABELS = {
    "alternate":         "Xen kẽ (chẵn IN, lẻ OUT)",
    "alternate_reverse": "Xen kẽ ngược (chẵn OUT, lẻ IN)",
    "all_in":            "Tất cả zoom IN",
    "all_out":           "Tất cả zoom OUT",
    "random":            "Random mỗi scene",
}

CODEC_FORMATS: list[CodecFormat] = ["h264", "hevc"]
CODEC_LABELS = {
    "h264": "H.264 (tương thích nhất)",
    "hevc": "HEVC/H.265 (chất lượng cao, file nhỏ)",
}

QUALITY_PRESETS: list[QualityPreset] = ["speed", "balanced", "quality", "max"]
QUALITY_LABELS = {
    "speed":    "Speed (nhanh, file lớn hơn)",
    "balanced": "Balanced (cân bằng — mặc định)",
    "quality":  "Quality (chất lượng cao)",
    "max":      "Max Quality (đẹp nhất, chậm nhất)",
}

PROJECT_ROOT = Path(__file__).parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
