import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import pysrt
import streamlit as st
from proglog import ProgressBarLogger

from config import (
    ASPECT_PRESETS,
    AVAILABLE_TRANSITIONS,
    CODEC_FORMATS,
    CODEC_LABELS,
    OUTPUT_DIR,
    PAN_OPTIONS,
    QUALITY_LABELS,
    QUALITY_PRESETS,
    ZOOM_MODE_LABELS,
    ZOOM_MODES,
    RenderConfig,
    resolve_encoder,
)
from effects.subtitle import (
    FONT_OPTIONS as SUB_FONT_OPTIONS,
    SUBTITLE_PRESET_NAMES,
    SYNC_MODE_LABELS,
    SYNC_MODES,
)


from generator import (
    Scene,
    build_final,
    quick_render,
    render_scenes,
    render_to_project,
)
from project import (
    init_workspace,
    list_projects,
    load_state,
    new_project_id,
    project_files,
    project_path,
    save_state,
)
from utils.image_processor import _scene_num
from utils.parse_srt import Segment

SCENES_FILENAME = "scenes.json"
SRT_FILENAME = "voice.srt"

st.set_page_config(page_title="Auto Video Editor", layout="wide", page_icon="🎬")


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _list_gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return []
    return [g.strip() for g in out.splitlines() if g.strip()]


def _match_gpu_brand(gpus: list[str]) -> tuple[str | None, str | None]:
    """Trả về (brand, gpu_name) — brand ∈ nvidia | amd | intel | None."""
    for name in gpus:
        lower = name.lower()
        if any(k in lower for k in ("nvidia", "geforce", "rtx", "gtx")):
            return "nvidia", name
        if any(k in lower for k in ("radeon", " amd ")):
            return "amd", name
    for name in gpus:
        if "intel" in name.lower():
            return "intel", name
    return None, None


@st.cache_data(ttl=600)
def _ffmpeg_encoders() -> set[str]:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return set()
    available = set()
    for name in ("libx264", "libx265", "h264_nvenc", "hevc_nvenc",
                 "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf"):
        if name in out:
            available.add(name)
    return available


@st.cache_data(ttl=600)
def _detect_gpu() -> tuple[str | None, str | None]:
    """Returns (brand, gpu_name). brand ∈ nvidia | amd | intel | None."""
    gpus = _list_gpu_names()
    if not gpus:
        return None, None
    return _match_gpu_brand(gpus)


def _human_size(n: int) -> str:
    n_f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n_f < 1024:
            return f"{n_f:.1f} {unit}"
        n_f /= 1024
    return f"{n_f:.1f} TB"


def _format_render_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def _parse_fps(rate: str) -> float:
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


class StreamlitProgressLogger(ProgressBarLogger):
    """Bridge MoviePy/proglog → Streamlit st.progress widget.

    proglog gọi `bars_callback` mỗi khi bar's attribute thay đổi. Bar 't'
    là frame-write progress; 'chunk' là audio encoding chunk progress.
    Throttle update tối đa 10 lần/giây để tránh flood websocket.
    """

    _BAR_LABELS = {
        "t":     "Đang ghi video frames",
        "chunk": "Đang encode audio",
    }

    def __init__(self, widget, start_time: float):
        super().__init__()
        self.widget = widget
        self.start_time = start_time
        self._last_update = 0.0
        self._last_pct = -1
        self._current_bar = None

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr != "index":
            return
        bar_state = self.state.get("bars", {}).get(bar, {})
        total = bar_state.get("total")
        if not total:
            return

        pct = min(max(value / total, 0.0), 1.0)
        pct_int = int(pct * 100)
        now = time.monotonic()

        # Throttle: chỉ update khi đổi % nguyên hoặc đã 100ms từ lần trước
        if bar == self._current_bar and pct_int == self._last_pct and (now - self._last_update) < 0.1:
            return
        self._last_update = now
        self._last_pct = pct_int
        self._current_bar = bar

        elapsed = now - self.start_time
        eta = (elapsed / pct - elapsed) if pct > 0.02 else None
        label = self._BAR_LABELS.get(bar, bar)
        eta_text = f" • còn ~{eta:.0f}s" if eta is not None else ""
        text = f"{label}: {pct_int}% ({value}/{total}){eta_text}"

        try:
            self.widget.progress(pct, text=text)
        except Exception:
            pass


def _probe_video(path: Path) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
        return {
            "width": v.get("width"),
            "height": v.get("height"),
            "fps": _parse_fps(v.get("r_frame_rate", "0/1")),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "bitrate": int(data.get("format", {}).get("bit_rate", 0)),
            "v_codec": v.get("codec_name", "?"),
            "a_codec": a.get("codec_name", "?"),
        }
    except Exception:
        return {}


# Custom CSS — chỉ tinh chỉnh metric + video, không phá theme Streamlit
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.06);
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid rgba(128, 128, 128, 0.12);
    }
    [data-testid="stMetricLabel"] p { font-size: 0.78rem; opacity: 0.75; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; }
    [data-testid="stVideo"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    [data-testid="stVideo"] video {
        border-radius: 14px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);
        max-height: 460px;
        max-width: 100%;
        width: auto !important;
        height: auto !important;
        background: #000;
    }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Auto Video Editor")
st.caption("JSON + SRT + ảnh + voice → MP4 với Ken Burns + transitions + subtitle")


def _render_result_panel(rec: dict):
    out_path = Path(rec["path"])
    if not out_path.exists():
        return

    probe = _probe_video(out_path)
    file_size = out_path.stat().st_size
    duration = probe.get("duration") or 0.0
    elapsed = float(rec.get("elapsed", 0))
    speed_ratio = (elapsed / duration) if duration > 0 else 0.0

    with st.container(border=True):
        st.markdown("#### Kết quả render")
        col_v, col_info = st.columns([2, 3], gap="large")

        with col_v:
            st.video(str(out_path))
            st.caption(f"`{out_path}`")

        with col_info:
            st.markdown("##### Video metadata")
            m1, m2 = st.columns(2)
            m1.metric(
                "Độ phân giải",
                f"{probe.get('width', rec.get('w'))}×{probe.get('height', rec.get('h'))}",
            )
            m2.metric("FPS", f"{probe.get('fps', rec.get('fps', 0)):.0f}")
            m1.metric("Thời lượng", f"{duration:.2f}s")
            m2.metric("Dung lượng", _human_size(file_size))

            bitrate_mbps = (probe.get("bitrate", 0) or 0) / 1_000_000
            codecs = f"{probe.get('v_codec', '?')} / {probe.get('a_codec', '?')}"
            m1.metric("Bitrate", f"{bitrate_mbps:.2f} Mbps")
            m2.metric("Codec", codecs)

            st.metric(
                "Render time",
                _format_render_time(elapsed),
                delta=f"{speed_ratio:.1f}× realtime" if speed_ratio > 0 else None,
                delta_color="off",
            )

            st.write("")
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "Tải xuống MP4",
                video_bytes,
                file_name=rec["name"],
                mime="video/mp4",
                type="primary",
                use_container_width=True,
            )
            if st.button("Mở thư mục output", use_container_width=True, key="open_folder_btn"):
                try:
                    os.startfile(str(OUTPUT_DIR))
                except Exception as exc:
                    st.warning(f"Không mở được Explorer: {exc}")


def _peek_json_scene_count(uploaded) -> int | None:
    if uploaded is None:
        return None
    try:
        data = json.loads(uploaded.getvalue().decode("utf-8"))
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


def _peek_srt_count(uploaded) -> int | None:
    if uploaded is None:
        return None
    try:
        text = uploaded.getvalue().decode("utf-8-sig")
        return len(pysrt.from_string(text))
    except Exception:
        return None


# ============================================================================
# Scene Editor (E3-E5)
# ============================================================================

_PAN_OPTIONS_EDITOR = ["", "none", "left-right", "right-left", "up-down", "down-up", "random"]
_TRANSITION_OPTIONS_EDITOR = ["", *AVAILABLE_TRANSITIONS]


def _rgb_to_hex(rgb) -> str:
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        return f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}"
    return "#FFFFFF"


def _update_segment_text(pid: str, scene_idx: int, new_text: str):
    """Edit subtitle text cho scene_idx (segments[idx-1])."""
    state = load_state(pid)
    if not state:
        return None
    seg_pos = scene_idx - 1
    if 0 <= seg_pos < len(state["segments"]):
        state["segments"][seg_pos]["text"] = new_text
        _save_state_dict(pid, state)
    return state


def _swap_image(pid: str, scene_idx: int, image_bytes: bytes, original_name: str):
    """Ghi đè file ảnh hiện tại của scene. Trả về Path đã ghi."""
    state = load_state(pid)
    if not state:
        return None
    files = project_files(pid)
    for sd in state["scenes"]:
        if int(sd["index"]) == scene_idx:
            cur_path = Path(sd["image_path"])
            # Giữ tên file cũ (để Scene.image_path không đổi)
            cur_path.write_bytes(image_bytes)
            state.setdefault("scene_dirty", {})[str(scene_idx)] = True
            _save_state_dict(pid, state)
            return cur_path
    return None


def _scene_from_state_dict(d: dict) -> Scene:
    return Scene(
        index=int(d["index"]),
        image_path=Path(d["image_path"]),
        script=d.get("script", ""),
        start=float(d["start"]),
        end=float(d["end"]),
        duration=float(d["duration"]),
        zoom_kind=d.get("zoom_kind", "in"),
        zoom_amount=d.get("zoom_amount"),
        pan=d.get("pan"),
        subtitle_enabled=bool(d.get("subtitle_enabled", True)),
        transition_in=d.get("transition_in"),
    )


def _segments_from_state(items: list) -> list[Segment]:
    return [Segment(start=float(x["start"]), end=float(x["end"]), text=x.get("text", "")) for x in items]


def _save_state_dict(pid: str, state: dict):
    state["last_modified"] = time.time()
    sp = project_path(pid) / "state.json"
    sp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _update_scene_in_state(pid: str, scene_idx: int, updates: dict, mark_dirty: bool = True):
    state = load_state(pid)
    if not state:
        return None
    for sd in state["scenes"]:
        if int(sd["index"]) == scene_idx:
            sd.update(updates)
            break
    if mark_dirty:
        state.setdefault("scene_dirty", {})[str(scene_idx)] = True
    _save_state_dict(pid, state)
    return state


def _rerender_one_scene(pid: str, scene_idx: int):
    state = load_state(pid)
    if not state:
        return
    config = RenderConfig(**state["config"])
    all_scenes = [_scene_from_state_dict(d) for d in state["scenes"]]
    files = project_files(pid)
    render_scenes(
        all_scenes, config, files["scenes_dir"],
        scenes_to_render=[scene_idx],
    )
    state.setdefault("scene_dirty", {})[str(scene_idx)] = False
    _save_state_dict(pid, state)


def _do_build_final_from_state(pid: str):
    state = load_state(pid)
    if not state:
        st.error("Không load được state")
        return
    config = RenderConfig(**state["config"])
    all_scenes = [_scene_from_state_dict(d) for d in state["scenes"]]
    segments = _segments_from_state(state["segments"])
    files = project_files(pid)

    scene_paths = {}
    for s in all_scenes:
        p = files["scenes_dir"] / f"scene_{s.index:04d}.mp4"
        if p.exists():
            scene_paths[s.index] = p

    missing = [s.index for s in all_scenes if s.index not in scene_paths]
    if missing:
        st.error(f"Thiếu scene MP4: {missing}. Re-render scene đó trước.")
        return

    progress = st.progress(0.0, text="Build final video từ scenes cache...")
    t0 = time.perf_counter()
    logger = StreamlitProgressLogger(progress, start_time=t0)
    try:
        build_final(
            all_scenes, scene_paths, segments,
            files["voice"], files["final_mp4"], config,
            logger=logger,
        )
        elapsed = time.perf_counter() - t0
        st.session_state["last_render"] = {
            "path": str(files["final_mp4"]),
            "name": f"{pid}.mp4",
            "elapsed": elapsed,
            "w": config.width,
            "h": config.height,
            "fps": config.fps,
            "project_id": pid,
        }
        st.toast(f"Build final xong trong {_format_render_time(elapsed)}", icon="✅")
    except Exception as exc:
        st.error(f"Lỗi build final: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
    finally:
        try:
            progress.empty()
        except Exception:
            pass


def _scene_card(scene_data: dict, files: dict, is_dirty: bool, pid: str, segment_text: str):
    idx = int(scene_data["index"])
    mp4_path = files["scenes_dir"] / f"scene_{idx:04d}.mp4"
    image_path = Path(scene_data["image_path"])

    with st.container(border=True):
        c_thumb, c_form, c_video = st.columns([1, 2.5, 1.4], gap="medium")

        with c_thumb:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            st.markdown(f"**Scene {idx}**")
            if is_dirty:
                st.warning("Cần re-render", icon="⚠️")

        with c_form:
            st.caption(f"Script: {scene_data.get('script', '')}")
            with st.form(f"scn_form_{idx}", clear_on_submit=False, border=False):
                # E6a: Sub text + E7: sub toggle + E8: transition_in
                st.markdown("**Subtitle & transition** (build final áp dụng, không cần re-render scene)")
                new_text = st.text_area(
                    "Subtitle text (SRT line)",
                    value=segment_text,
                    height=68,
                    key=f"text_{idx}",
                )
                sc1, sc2 = st.columns(2)
                with sc1:
                    new_sub_on = st.checkbox(
                        "Hiện sub cho scene này",
                        value=bool(scene_data.get("subtitle_enabled", True)),
                        key=f"sub_{idx}",
                    )
                with sc2:
                    cur_trans = scene_data.get("transition_in") or ""
                    trans_idx = (
                        _TRANSITION_OPTIONS_EDITOR.index(cur_trans)
                        if cur_trans in _TRANSITION_OPTIONS_EDITOR
                        else 0
                    )
                    new_trans = st.selectbox(
                        "Transition vào scene này",
                        _TRANSITION_OPTIONS_EDITOR,
                        index=trans_idx,
                        format_func=lambda t: "(global mix)" if t == "" else t,
                        key=f"trans_{idx}",
                        disabled=(idx == 1),  # scene 1 không có transition vào
                        help="Override transition cụ thể. Để '(global mix)' = dùng setting global.",
                    )

                st.markdown("**Ken Burns** (đổi → cần re-render scene)")
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    new_dur = st.number_input(
                        "Duration (s)", min_value=0.5, max_value=60.0,
                        value=float(scene_data["duration"]), step=0.1,
                        key=f"dur_{idx}",
                    )
                with fc2:
                    new_kind = st.selectbox(
                        "Zoom", ["in", "out"],
                        index=0 if scene_data.get("zoom_kind") == "in" else 1,
                        key=f"zk_{idx}",
                    )
                with fc3:
                    new_amt = st.number_input(
                        "Amount (0=global)",
                        min_value=0.0, max_value=2.5,
                        value=float(scene_data.get("zoom_amount") or 0.0),
                        step=0.05,
                        key=f"za_{idx}",
                    )
                with fc4:
                    cur_pan = scene_data.get("pan") or ""
                    pan_idx = _PAN_OPTIONS_EDITOR.index(cur_pan) if cur_pan in _PAN_OPTIONS_EDITOR else 0
                    new_pan = st.selectbox(
                        "Pan", _PAN_OPTIONS_EDITOR,
                        index=pan_idx,
                        format_func=lambda p: "(global)" if p == "" else p,
                        key=f"pn_{idx}",
                    )

                # E9: image swap
                new_image = st.file_uploader(
                    "Đổi ảnh scene này (optional → mark dirty)",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"img_{idx}",
                )

                bc1, bc2 = st.columns(2)
                with bc1:
                    save_only = st.form_submit_button(
                        "Save (chỉ sub/transition)",
                        use_container_width=True,
                    )
                with bc2:
                    do_rerender = st.form_submit_button(
                        "Save + Re-render scene",
                        type="primary",
                        use_container_width=True,
                    )

            if save_only or do_rerender:
                # 1. Update segment text (E6a)
                if new_text != segment_text:
                    _update_segment_text(pid, idx, new_text)

                # 2. Image swap (E9): if file uploaded, write to disk + mark dirty
                if new_image is not None:
                    _swap_image(pid, idx, new_image.getvalue(), new_image.name)

                # 3. Update scene fields
                kb_changed = (
                    abs(new_dur - float(scene_data["duration"])) > 0.001
                    or new_kind != scene_data.get("zoom_kind")
                    or (new_amt if new_amt > 0 else None) != scene_data.get("zoom_amount")
                    or (new_pan if new_pan else None) != scene_data.get("pan")
                    or new_image is not None
                )
                scene_updates = {
                    "duration": new_dur,
                    "zoom_kind": new_kind,
                    "zoom_amount": new_amt if new_amt > 0 else None,
                    "pan": new_pan if new_pan else None,
                    "subtitle_enabled": new_sub_on,
                    "transition_in": new_trans if new_trans else None,
                }
                _update_scene_in_state(pid, idx, scene_updates, mark_dirty=kb_changed)

                if do_rerender:
                    with st.spinner(f"Re-rendering scene {idx}..."):
                        _rerender_one_scene(pid, idx)
                    st.success(f"Scene {idx} re-rendered xong. Bấm 'Build final' để áp dụng.")
                else:
                    if kb_changed:
                        st.info("Đã lưu. Vì có thay đổi Ken Burns/ảnh, cần bấm 'Save + Re-render' để cập nhật scene MP4.")
                    else:
                        st.success("Đã lưu sub/transition. Bấm 'Build final' để áp dụng (không cần re-render scene).")
                time.sleep(0.4)
                st.rerun()

        with c_video:
            if mp4_path.exists():
                with open(mp4_path, "rb") as f:
                    st.video(f.read())
                st.caption(f"{mp4_path.stat().st_size / 1024 / 1024:.1f} MB")
            else:
                st.warning("Chưa render", icon="📭")


def _render_subtitle_settings_form(pid: str, state: dict):
    """E6b: Editor sidebar panel — chỉnh global subtitle settings + apply qua build_final."""
    cfg = state.get("config", {})
    with st.expander("Subtitle settings (global)", expanded=False):
        with st.form(f"sub_settings_{pid}", clear_on_submit=False):
            sub_on = st.checkbox(
                "Hiện subtitle",
                value=bool(cfg.get("subtitle_enabled", False)),
                key=f"ed_sub_on_{pid}",
            )
            sub_preset = st.selectbox(
                "Preset",
                SUBTITLE_PRESET_NAMES,
                index=SUBTITLE_PRESET_NAMES.index(cfg.get("subtitle_preset", "TikTok"))
                if cfg.get("subtitle_preset") in SUBTITLE_PRESET_NAMES else 0,
                key=f"ed_sub_preset_{pid}",
            )
            sub_sync = st.selectbox(
                "Sync mode",
                SYNC_MODES,
                index=SYNC_MODES.index(cfg.get("subtitle_sync_mode", "per_line"))
                if cfg.get("subtitle_sync_mode") in SYNC_MODES else 0,
                format_func=lambda m: SYNC_MODE_LABELS.get(m, m),
                key=f"ed_sub_sync_{pid}",
            )
            sub_font = st.selectbox(
                "Font",
                SUB_FONT_OPTIONS,
                index=SUB_FONT_OPTIONS.index(cfg.get("subtitle_font", "Arial Bold"))
                if cfg.get("subtitle_font") in SUB_FONT_OPTIONS else 1,
                key=f"ed_sub_font_{pid}",
            )
            sub_size = st.slider(
                "Font size", 24, 120,
                value=int(cfg.get("subtitle_font_size", 64)), step=2,
                key=f"ed_sub_size_{pid}",
            )
            col_fg, col_bg = st.columns(2)
            with col_fg:
                sub_color_hex = st.color_picker(
                    "Màu chữ",
                    value=_rgb_to_hex(cfg.get("subtitle_color", [255, 255, 255])),
                    key=f"ed_sub_color_{pid}",
                )
            with col_bg:
                sub_stroke_hex = st.color_picker(
                    "Màu viền",
                    value=_rgb_to_hex(cfg.get("subtitle_stroke_color", [0, 0, 0])),
                    key=f"ed_sub_stroke_{pid}",
                )
            sub_stroke_w = st.slider(
                "Stroke width", 0, 10,
                value=int(cfg.get("subtitle_stroke_width", 5)),
                key=f"ed_sub_stroke_w_{pid}",
            )
            sub_highlight_hex = st.color_picker(
                "Highlight (karaoke)",
                value=_rgb_to_hex(cfg.get("subtitle_highlight_color", [255, 230, 0])),
                key=f"ed_sub_hl_{pid}",
            )
            sub_position = st.selectbox(
                "Vị trí", ["top", "middle", "bottom"],
                index=["top", "middle", "bottom"].index(cfg.get("subtitle_position", "bottom"))
                if cfg.get("subtitle_position") in ["top", "middle", "bottom"] else 2,
                key=f"ed_sub_pos_{pid}",
            )
            sub_y_off = st.slider(
                "Y offset", 0.05, 0.95,
                value=float(cfg.get("subtitle_y_offset_pct", 0.80)), step=0.01,
                key=f"ed_sub_yoff_{pid}",
            )

            if st.form_submit_button("Apply sub settings", use_container_width=True):
                cfg["subtitle_enabled"] = sub_on
                cfg["subtitle_preset"] = sub_preset
                cfg["subtitle_sync_mode"] = sub_sync
                cfg["subtitle_font"] = sub_font
                cfg["subtitle_font_size"] = sub_size
                cfg["subtitle_color"] = list(_hex_to_rgb(sub_color_hex))
                cfg["subtitle_stroke_color"] = list(_hex_to_rgb(sub_stroke_hex))
                cfg["subtitle_stroke_width"] = sub_stroke_w
                cfg["subtitle_highlight_color"] = list(_hex_to_rgb(sub_highlight_hex))
                cfg["subtitle_position"] = sub_position
                cfg["subtitle_y_offset_pct"] = sub_y_off
                state["config"] = cfg
                _save_state_dict(pid, state)
                st.success("Đã lưu sub settings. Bấm 'Build final' để áp dụng.")
                time.sleep(0.4)
                st.rerun()


def _show_editor():
    """Scene Editor page: load project, list scene cards, edit + re-render + build final."""
    with st.sidebar:
        st.header("Editor")
        projects = list_projects()
        if not projects:
            st.info("Chưa có project. Mở tab 'Render mới' để tạo project đầu tiên.")
        else:
            pids = [p["project_id"] for p in projects]
            cur = st.session_state.get("current_project_id")
            default_idx = pids.index(cur) if cur in pids else 0
            chosen_pid = st.selectbox(
                "Project",
                pids,
                index=default_idx,
                format_func=lambda p: f"{p} · {next(x['n_scenes'] for x in projects if x['project_id']==p)} scenes",
                key="editor_pid_select",
            )
            st.session_state["current_project_id"] = chosen_pid

            if st.button("🔄 Reload state", use_container_width=True, key="reload_state_btn"):
                st.rerun()

            # E6b: Global subtitle settings panel
            _state_for_sidebar = load_state(chosen_pid)
            if _state_for_sidebar:
                _render_subtitle_settings_form(chosen_pid, _state_for_sidebar)

    st.markdown("## Scene Editor")
    if not projects:
        return

    pid = st.session_state["current_project_id"]
    state = load_state(pid)
    if not state:
        st.error(f"Project `{pid}` không có state.json hợp lệ.")
        return

    files = project_files(pid)
    scenes_data = state["scenes"]
    segments_data = state.get("segments", [])
    dirty = state.get("scene_dirty", {})
    n_dirty = sum(1 for d in dirty.values() if d)

    # Header row
    h_info, h_action = st.columns([3, 1])
    with h_info:
        ago = max(0, time.time() - state.get("last_modified", time.time()))
        ago_text = f"{int(ago)}s" if ago < 60 else f"{int(ago/60)}m"
        st.caption(
            f"Project `{pid}` · **{len(scenes_data)} scenes** · "
            f"**{n_dirty}** cần re-render · last modified {ago_text} ago"
        )
    with h_action:
        if st.button(
            "Build final video",
            type="primary", use_container_width=True,
            key="build_final_top_btn",
            help="Concat scene MP4 đã cache → final.mp4 mới. Chỉ ~30-60s.",
        ):
            _do_build_final_from_state(pid)
            st.rerun()

    if n_dirty > 0:
        st.warning(f"Có {n_dirty} scene cần re-render trước khi build final hiển thị đúng.")

    st.divider()

    for i, sd in enumerate(scenes_data):
        seg_text = segments_data[i]["text"] if i < len(segments_data) else ""
        _scene_card(sd, files, dirty.get(str(sd["index"]), False), pid, seg_text)

    # Result panel nếu vừa build xong
    last = st.session_state.get("last_render")
    if last and last.get("project_id") == pid:
        st.divider()
        _render_result_panel(last)


# ============================================================================
# Mode selector: branch render vs editor
# ============================================================================

mode = st.radio(
    "App mode",
    ["render", "editor"],
    horizontal=True,
    label_visibility="collapsed",
    format_func=lambda m: {"render": "📥 Render mới", "editor": "✂️ Scene Editor"}[m],
    key="app_mode",
)

if mode == "editor":
    _show_editor()
    st.stop()


with st.sidebar:
    st.header("Settings")

    aspect_label = st.selectbox(
        "Aspect ratio",
        list(ASPECT_PRESETS.keys()) + ["Custom"],
        key="aspect_label",
    )
    if aspect_label == "Custom":
        w = int(st.number_input("Width", 240, 4096, value=1080, step=2, key="custom_w"))
        h = int(st.number_input("Height", 240, 4096, value=1920, step=2, key="custom_h"))
    else:
        w, h = ASPECT_PRESETS[aspect_label]
    st.caption(f"→ {w}×{h}")

    fps = int(st.selectbox("FPS", [24, 25, 30, 60], index=2, key="fps"))

    st.markdown("**Render engine**")
    gpu_brand, gpu_name = _detect_gpu()
    available_encoders = _ffmpeg_encoders()

    codec_format = st.selectbox(
        "Codec",
        CODEC_FORMATS,
        format_func=lambda c: CODEC_LABELS.get(c, c),
        help="HEVC nén tốt hơn ~30-40% so với H.264 cùng quality, encode nhanh hơn trên GPU.",
        key="codec_format",
    )

    engine_options = {"cpu": "CPU (software)"}
    if gpu_brand:
        derived = resolve_encoder(codec_format, gpu_brand)
        if derived in available_encoders:
            engine_options[gpu_brand] = f"GPU {gpu_brand.upper()} ({gpu_name})"
    engine = st.selectbox(
        "Engine",
        list(engine_options.keys()),
        format_func=lambda e: engine_options[e],
        help="CPU ổn định nhất. GPU nhanh hơn ở bước encode cuối — tổng cải thiện ~15-25%.",
        key="engine",
    )

    quality = st.selectbox(
        "Chất lượng",
        QUALITY_PRESETS,
        index=1,
        format_func=lambda q: QUALITY_LABELS.get(q, q),
        help="Số CRF/CQ thấp hơn = file lớn hơn, chất lượng cao hơn.",
        key="quality_preset",
    )
    final_encoder = resolve_encoder(codec_format, engine)
    if final_encoder not in available_encoders:
        st.warning(
            f"Encoder `{final_encoder}` không có trong FFmpeg → fallback libx264. "
            "Đổi codec/engine khác."
        )
    else:
        st.caption(f"Sẽ dùng: `{final_encoder}`")

    st.markdown("**Multiprocessing**")
    parallel = st.checkbox(
        "Render scene song song (xài nhiều CPU core)",
        value=False,
        key="parallel",
        help=(
            "Mỗi scene Ken Burns được render song song trên worker process riêng, "
            "sau đó concat lại. Tăng 4-8× trên máy nhiều core. "
            "Nên bật trên máy ≥8 core, không cần thiết trên laptop."
        ),
    )
    if parallel:
        cpu_count = os.cpu_count() or 4
        # Default an toàn hơn: cpu_count // 2 hoặc 4, tránh hết page file Windows
        safe_default = max(2, min(cpu_count // 2, 4))
        workers = st.slider(
            "Số worker", 2, max(cpu_count, 4), safe_default, 1, key="workers",
            help=(
                "Mỗi worker xài ~400 MB RAM. Trên Windows, > 6 worker dễ bị "
                "WinError 1455 (paging file too small). Tăng page file Windows "
                "lên 16-32 GB để chạy được nhiều worker hơn."
            ),
        )
        if workers > 6:
            st.caption(
                "⚠️ > 6 worker có thể gây lỗi paging file trên Windows. "
                "Tăng page file lên 16 GB trước khi chạy."
            )
    else:
        workers = 0

    st.markdown("**Ken Burns zoom**")
    zoom_mode = st.selectbox(
        "Zoom mode (pattern toàn cục)",
        ZOOM_MODES,
        format_func=lambda m: ZOOM_MODE_LABELS.get(m, m),
        help="Override từng scene qua JSON field 'zoom_kind' nếu cần.",
        key="zoom_mode",
    )
    col_zs, col_ze = st.columns(2)
    with col_zs:
        zoom_start_pct = st.number_input("Start (%)", 80, 200, 100, 5, key="zoom_start_pct")
    with col_ze:
        zoom_end_pct = st.number_input("End (%)", 80, 200, 120, 5, key="zoom_end_pct")

    pan_direction = st.selectbox(
        "Pan direction (camera motion)",
        PAN_OPTIONS,
        help="Hướng camera quét trong khung. 'random' = mỗi scene 1 hướng ngẫu nhiên.",
        key="pan_direction",
    )

    st.markdown("**Transitions**")
    transitions = st.multiselect(
        "Loại transition",
        AVAILABLE_TRANSITIONS,
        default=["fade", "slide"],
        help="'slide'/'wipe' không có hướng = random 1 trong 4 hướng. Chọn nhiều loại + mode 'mix' để mỗi cảnh có hiệu ứng khác.",
        key="transitions",
    )
    trans_mode = st.radio(
        "Mode",
        ["mix", "fixed"],
        horizontal=True,
        help="mix = random từ list; fixed = chỉ dùng loại đầu tiên",
        key="trans_mode",
    )
    trans_dur = st.slider("Duration (s)", 0.0, 1.5, 0.5, 0.1, key="trans_dur")

    st.markdown("**Subtitle**")
    sub_enabled = st.checkbox("Hiện subtitle (burn vào video)", value=False, key="sub_enabled")
    if sub_enabled:
        sub_preset = st.selectbox("Preset", SUBTITLE_PRESET_NAMES, key="sub_preset")
        sub_sync = st.selectbox(
            "Sync mode",
            SYNC_MODES,
            format_func=lambda m: SYNC_MODE_LABELS[m],
            key="sub_sync",
        )
        sub_font = st.selectbox("Font", SUB_FONT_OPTIONS, index=1, key="sub_font")
        sub_font_size = st.slider("Font size", 24, 120, 64, 2, key="sub_font_size")
        col_fg, col_bg = st.columns(2)
        with col_fg:
            sub_color_hex = st.color_picker("Màu chữ", "#FFFFFF", key="sub_color_hex")
        with col_bg:
            sub_stroke_hex = st.color_picker("Màu viền", "#000000", key="sub_stroke_hex")
        sub_stroke_w = st.slider("Độ rộng viền (px)", 0, 10, 5, 1, key="sub_stroke_w")
        sub_highlight_hex = st.color_picker(
            "Màu highlight (karaoke)", "#FFE600", key="sub_highlight_hex"
        )
        sub_position = st.selectbox(
            "Vị trí", ["top", "middle", "bottom"], index=2, key="sub_position"
        )
        sub_y_offset = st.slider(
            "Y offset (% chiều cao, chỉ khi position=bottom)",
            0.05, 0.95, 0.80, 0.01, key="sub_y_offset",
        )
    else:
        sub_preset = "TikTok"
        sub_sync = "per_line"
        sub_font = "Arial Bold"
        sub_font_size = 64
        sub_color_hex = "#FFFFFF"
        sub_stroke_hex = "#000000"
        sub_stroke_w = 5
        sub_highlight_hex = "#FFE600"
        sub_position = "bottom"
        sub_y_offset = 0.80

st.subheader("Inputs")
col1, col2 = st.columns(2)
with col1:
    json_file = st.file_uploader(SCENES_FILENAME, type=["json"], key="json_file")
    srt_file = st.file_uploader(SRT_FILENAME, type=["srt"], key="srt_file")
with col2:
    voice_file = st.file_uploader("Voice (mp3/wav)", type=["mp3", "wav"], key="voice_file")
    image_files = st.file_uploader(
        "Images (đặt tên có số, vd 1.png, 2.png, ...)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="image_files",
    )

n_scenes = _peek_json_scene_count(json_file)
n_srt = _peek_srt_count(srt_file)
n_img = len(image_files) if image_files else 0

if any(x is not None for x in (n_scenes, n_srt)) or n_img > 0:
    st.markdown("**Tổng quan input:**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Scenes (JSON)", n_scenes if n_scenes is not None else "—")
    m2.metric("Dòng SRT", n_srt if n_srt is not None else "—")
    m3.metric("Ảnh", n_img if n_img else "—")

    counts = [c for c in (n_scenes, n_srt, n_img) if c]
    mismatch = bool(counts) and len(set(counts)) > 1
    if mismatch:
        st.error(
            f"Số lượng không khớp! JSON={n_scenes}, SRT={n_srt}, Images={n_img}. "
            "Cả ba phải bằng nhau."
        )

    if image_files:
        with st.expander("Thứ tự ảnh (sort theo số trong tên)"):
            ordered = sorted(image_files, key=lambda f: _scene_num(Path(f.name)))
            st.write([f.name for f in ordered])

st.divider()

ready = all([json_file, srt_file, voice_file, image_files])
counts_match = n_scenes == n_srt == n_img if (n_scenes and n_srt and n_img) else False

if not transitions and trans_dur > 0:
    st.info("Chưa chọn transition nào — video sẽ không có hiệu ứng chuyển cảnh.")

can_render = ready and counts_match
if st.button("Generate Video", disabled=not can_render, type="primary"):
    cfg = RenderConfig(
        width=w,
        height=h,
        fps=fps,
        codec_format=codec_format,
        engine=engine,
        quality_preset=quality,
        parallel=parallel,
        parallel_workers=workers,
        zoom_mode=zoom_mode,
        zoom_start=zoom_start_pct / 100.0,
        zoom_end=zoom_end_pct / 100.0,
        pan_direction=pan_direction,
        transitions=transitions,
        transition_mode=trans_mode,
        transition_duration=trans_dur,
        subtitle_enabled=sub_enabled,
        subtitle_preset=sub_preset,
        subtitle_sync_mode=sub_sync,
        subtitle_font=sub_font,
        subtitle_font_size=sub_font_size,
        subtitle_color=_hex_to_rgb(sub_color_hex),
        subtitle_stroke_color=_hex_to_rgb(sub_stroke_hex),
        subtitle_stroke_width=sub_stroke_w,
        subtitle_highlight_color=_hex_to_rgb(sub_highlight_hex),
        subtitle_position=sub_position,
        subtitle_y_offset_pct=sub_y_offset,
    )

    project_id = new_project_id()
    files = init_workspace(project_id)
    work_dir = files["root"]

    try:
        with st.spinner("Đang chuẩn bị workspace..."):
            (work_dir / SCENES_FILENAME).write_bytes(json_file.getvalue())
            (work_dir / SRT_FILENAME).write_bytes(srt_file.getvalue())
            voice_suffix = Path(voice_file.name).suffix or ".mp3"
            voice_target = work_dir / f"voice{voice_suffix}"
            voice_target.write_bytes(voice_file.getvalue())
            for f in image_files:
                (files["images_dir"] / f.name).write_bytes(f.getvalue())
            # Refresh files dict để voice path hiện đúng
            files = project_files(project_id)

        out_name = f"{project_id}.mp4"
        out_path = files["final_mp4"]

        progress_widget = st.progress(0.0, text="Đang chuẩn bị pipeline...")

        _PREP_LABELS = {
            "parse":    "Đọc JSON + SRT + scan ảnh...",
            "build":    "Build timeline + Ken Burns + transitions...",
            "parallel": "Render scene song song trên nhiều core...",
            "encode":   "Bắt đầu encode video...",
        }

        def _on_prep(phase: str):
            label = _PREP_LABELS.get(phase, phase)
            try:
                progress_widget.progress(0.0, text=label)
            except Exception:
                pass

        def _on_parallel_progress(done: int, total: int):
            try:
                pct = min(done / total, 1.0)
                progress_widget.progress(
                    pct,
                    text=f"Parallel render scene: {done}/{total} ({int(pct*100)}%)",
                )
            except Exception:
                pass

        t0 = time.perf_counter()
        logger = StreamlitProgressLogger(progress_widget, start_time=t0)
        try:
            scenes, segments, scene_paths = render_to_project(
                files, cfg,
                logger=logger,
                prep_callback=_on_prep,
                parallel_progress=_on_parallel_progress,
            )
            save_state(project_id, scenes, segments, cfg, extra={
                "scene_mp4_files": {str(idx): str(p.name) for idx, p in scene_paths.items()},
            })
            st.session_state["current_project_id"] = project_id
        finally:
            elapsed = time.perf_counter() - t0
            try:
                progress_widget.empty()
            except Exception:
                pass

        st.session_state["last_render"] = {
            "path": str(out_path),
            "name": out_name,
            "elapsed": elapsed,
            "w": cfg.width,
            "h": cfg.height,
            "fps": cfg.fps,
            "project_id": project_id,
        }
        st.toast(f"Render xong trong {_format_render_time(elapsed)}", icon="✅")
    except Exception as e:
        msg = str(e)
        if "1455" in msg or "paging file" in msg.lower():
            st.error(
                "Windows hết page file (WinError 1455). "
                "**Fix nhanh**: Giảm số worker trong sidebar (Multiprocessing) xuống 4, "
                "hoặc tắt checkbox 'Render scene song song'. "
                "**Fix triệt để**: Tăng Windows page file lên 16-32 GB "
                "(System Properties → Advanced → Performance → Virtual memory)."
            )
        else:
            st.error(f"Lỗi: {e}")
        with st.expander("Traceback chi tiết"):
            st.code(traceback.format_exc())
        # Workspace giữ lại cho debug, không xóa


# Result panel — persist qua reruns nhờ session_state
last_render = st.session_state.get("last_render")
if last_render:
    _render_result_panel(last_render)
