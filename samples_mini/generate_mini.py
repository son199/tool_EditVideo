"""Sinh sample mini 3 scene để test pipeline nhanh.

Chạy: `python samples_mini/generate_mini.py`
"""
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
PARENT_SAMPLES = HERE.parent / "samples"
IMG_OUT = HERE / "images"
IMG_OUT.mkdir(exist_ok=True)


def main():
    # Đọc 3 scene đầu từ samples chính
    full_scenes = json.loads((PARENT_SAMPLES / "scenes.json").read_text(encoding="utf-8"))
    mini_scenes = full_scenes[:3]

    # Lưu scenes.json mini
    (HERE / "scenes.json").write_text(
        json.dumps(mini_scenes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy 3 ảnh đầu
    for s in mini_scenes:
        idx = s["scene"]
        shutil.copy(PARENT_SAMPLES / "images" / f"{idx}.png", IMG_OUT / f"{idx}.png")

    # Trim SRT đến scene 3 — đọc full, lấy 3 block đầu
    full_srt = (PARENT_SAMPLES / "voice.srt").read_text(encoding="utf-8")
    blocks = [b for b in full_srt.strip().split("\n\n") if b.strip()][:3]
    mini_srt = "\n\n".join(blocks) + "\n"
    (HERE / "voice.srt").write_text(mini_srt, encoding="utf-8")

    # Sinh voice.mp3 silence khớp tổng duration 3 scene
    # Đọc end timestamp của scene 3 từ SRT
    import re
    last_block = blocks[-1]
    m = re.search(r"-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})", last_block)
    if m:
        h, mm, s, ms = (int(x) for x in m.groups())
        total = h * 3600 + mm * 60 + s + ms / 1000.0
    else:
        total = 13.0

    voice_out = HERE / "voice.mp3"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-t", f"{total:.3f}",
         "-ac", "2", "-ar", "44100", "-b:a", "128k",
         str(voice_out)],
        check=True, capture_output=True,
    )

    print(f"OK: 3 scenes, total {total:.1f}s")
    print(f"Path: {HERE}")


if __name__ == "__main__":
    main()
