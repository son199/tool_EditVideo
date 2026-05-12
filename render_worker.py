import os
from pathlib import Path
from moviepy.editor import VideoFileClip
from dataclasses import replace
from config import RenderConfig
from models import Scene, Segment

def _render_chunk_worker(args):
    """Worker render 1 đoạn video final. Tách ra file riêng để tránh lỗi Pickle trên Windows."""
    idx, start, end, scenes_raw, scene_paths, segments_raw, voice_path, config, out_tmp, progress_array = args
    
    from models import Scene, Segment
    from dataclasses import fields
    
    # Khôi phục Object từ Dictionary để tránh lỗi Pickle Class identity
    def dict_to_obj(cls, data):
        if isinstance(data, cls): return data
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    scenes = [dict_to_obj(Scene, s) for s in scenes_raw]
    segments = [dict_to_obj(Segment, s) for s in segments_raw]
    
    # Custom logger để bắn % về progress_array
    from proglog import ProgressBarLogger
    class WorkerProgressLogger(ProgressBarLogger):
        def callback(self, **kwargs):
            if 'bars' in kwargs and kwargs['bars']:
                # Lấy bar đầu tiên (thường là 't')
                bar = list(kwargs['bars'].values())[0]
                if bar['total'] > 0:
                    pct = int(bar['index'] / bar['total'] * 100)
                    progress_array[idx] = pct
        def bars_callback(self, bar, attr, value, old_value=None):
            if attr == 'index':
                pct = int(value / self.bars[bar]['total'] * 100)
                progress_array[idx] = pct

    # Import generator bên trong để tránh circular import khi pickling
    import generator
    
    # RAM OPTIMIZATION: Only load relevant clips for this 5s chunk
    t_dur = config.transition_duration
    cursor = 0.0
    relevant_scenes = []
    relevant_paths = []
    
    for i, s in enumerate(scenes):
        s_start = cursor
        s_end = cursor + s.duration
        if s_start < end and s_end > start:
            relevant_scenes.append(s)
            relevant_paths.append(scene_paths[s.index])
        cursor += s.duration - (t_dur if i < len(scenes) - 1 else 0)

    # Calculate offset
    first_s_start = 0.0
    temp_cursor = 0.0
    for i, s in enumerate(scenes):
        if s.index == relevant_scenes[0].index:
            first_s_start = temp_cursor
            break
        temp_cursor += s.duration - (t_dur if i < len(scenes) - 1 else 0)

    clips = [VideoFileClip(str(p)) for p in relevant_paths]
    composite = generator._compose_clips(clips, config, scenes=relevant_scenes)
    
    # Filter and SHIFT segments to local composite timeline
    visible_segments = [
        replace(seg, start=seg.start - first_s_start, end=seg.end - first_s_start)
        for seg in segments if seg.start < (first_s_start + composite.duration) and seg.end > first_s_start
    ]
    composite = generator.add_subtitle(composite, visible_segments, config)
    
    sub = composite.subclip(start - first_s_start, end - first_s_start)
    
    worker_logger = WorkerProgressLogger()
    _single_export(sub, out_tmp, config, logger=worker_logger)
    
    # Đánh dấu xong 100%
    progress_array[idx] = 100
    
    for c in clips: c.close()
    if composite.audio: composite.audio.close()
    return idx, str(out_tmp)


def _single_export(clip, out_path: Path, config: RenderConfig, logger=None):
    from config import resolve_encoder, probe_nvenc_works
    from generator import _QUALITY_TABLE, _encoder_extra_params

    encoder = resolve_encoder(config.codec_format, config.engine)
    if encoder in ("hevc_nvenc", "h264_nvenc") and not probe_nvenc_works():
        encoder = "libx264"

    quality_map = _QUALITY_TABLE.get(encoder) or _QUALITY_TABLE["libx264"]
    preset, q_val = quality_map.get(config.quality_preset, quality_map["balanced"])
    extra = _encoder_extra_params(encoder, q_val)

    clip.write_videofile(
        str(out_path),
        codec=encoder,
        audio_codec="aac",
        fps=config.fps,
        threads=2, 
        preset=preset,
        ffmpeg_params=extra or None,
        logger=logger if logger is not None else "bar",
    )
