"""Sinh sample inputs cho test: 40 scene "Du lịch Việt Nam".

Chạy: `python samples/generate_test_assets.py`

Output:
- samples/images/1.png ... 40.png  (1080×1920, color + scene title)
- samples/scenes.json              (40 scene)
- samples/voice.srt                (40 dòng SRT timing)
- samples/voice.mp3                (silence trùng tổng duration; thay bằng voice thật khi dùng)
"""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
IMG_DIR = HERE / "images"
IMG_DIR.mkdir(exist_ok=True)

WORDS_PER_SEC = 2.5
MIN_DUR = 3.0
SIZE = (1080, 1920)

# 40 scene: (color_rgb, script)
SCENES: list[tuple[tuple[int, int, int], str]] = [
    ((255, 140,  80), "Bình minh trên đỉnh Fansipan, nóc nhà Đông Dương."),
    ((220,  80,  40), "Phố cổ Hội An lung linh đèn lồng mỗi đêm rằm."),
    (( 40, 120, 180), "Vịnh Hạ Long với hàng nghìn đảo đá kỳ vĩ giữa biển xanh."),
    ((180, 180,  60), "Ruộng bậc thang Mù Cang Chải mùa lúa chín vàng rực."),
    ((120, 100,  80), "Sông Hương trầm mặc bên kinh thành Huế cổ kính."),
    (( 90, 130, 110), "Chợ nổi Cái Răng tấp nập từ tinh mơ sáng."),
    ((200,  50,  60), "Hồ Gươm với cầu Thê Húc đỏ rực giữa lòng Hà Nội."),
    ((255, 180,  90), "Đồi cát Mũi Né nhuộm vàng dưới ánh hoàng hôn."),
    (( 50,  60,  90), "Hang Sơn Đoòng, hang động tự nhiên lớn nhất thế giới."),
    ((180,  80, 130), "Phố đi bộ Nguyễn Huệ rực rỡ ánh đèn về đêm."),
    (( 70, 130, 180), "Đèo Hải Vân uốn lượn nhìn ra biển xanh thẳm."),
    ((180,  30,  30), "Lăng Bác trang nghiêm giữa quảng trường Ba Đình lịch sử."),
    (( 90,  70,  50), "Núi Bà Đen cao nhất Nam Bộ, linh thiêng và hùng vĩ."),
    (( 40, 180, 200), "Đảo Phú Quốc với biển ngọc bích và cát trắng mịn."),
    ((130, 150, 110), "Tam Cốc Bích Động, vịnh Hạ Long trên cạn của Ninh Bình."),
    ((210, 150,  80), "Chùa Một Cột độc đáo giữa lòng thủ đô Hà Nội."),
    ((110,  90,  80), "Cao nguyên đá Đồng Văn kỳ vĩ tựa miền cổ tích."),
    ((220, 200,  80), "Cánh đồng hoa cải vàng Mộc Châu rực rỡ mùa đông."),
    ((160, 130, 110), "Nhà thờ Đức Bà Sài Gòn cổ kính hơn trăm năm tuổi."),
    ((140,  80,  80), "Đại Nội Huế thâm trầm với dấu ấn triều Nguyễn."),
    ((130, 140,  70), "Suối Yến chùa Hương rộn ràng mùa lễ hội đầu xuân."),
    ((100, 130, 190), "Đỉnh Lảo Thẩn nơi mây luồn qua thung lũng nắng."),
    (( 60, 150, 120), "Đồng bằng sông Cửu Long mênh mông phù sa và sông nước."),
    ((220,  70,  30), "Núi lửa Chư Đăng Ya rực đỏ mùa hoa dã quỳ."),
    (( 50, 140, 170), "Hồ Ba Bể mặt nước phẳng lặng giữa núi rừng Bắc Kạn."),
    ((130, 100,  70), "Cố đô Hoa Lư ngàn năm văn hiến của đất Việt."),
    (( 80,  80, 100), "Đèo Mã Pí Lèng, vua của bốn đại đèo Tây Bắc."),
    (( 60, 170, 200), "Biển Mỹ Khê, một trong những bãi biển đẹp nhất hành tinh."),
    ((130, 170,  80), "Thung lũng Mai Châu ngút ngàn hoa và lúa xanh."),
    (( 40, 130,  70), "Vườn quốc gia Cúc Phương xanh thẳm và đa dạng sinh học."),
    ((200, 100,  60), "Phố cổ Hà Nội với ba mươi sáu phố phường lâu đời."),
    ((170, 130,  90), "Tháp Chăm Mỹ Sơn lưu dấu xưa của vương quốc Champa."),
    ((210, 170,  60), "Cầu Vàng Đà Nẵng nâng đỡ bởi đôi bàn tay khổng lồ."),
    (( 90, 160,  90), "Đồi chè Mộc Châu xanh ngút mắt trong sương sớm."),
    (( 40, 100, 150), "Vịnh Lan Hạ, viên ngọc ẩn của quần đảo Cát Bà."),
    ((250, 130,  30), "Phố lồng đèn Hội An đêm rằm tháng tám lung linh."),
    ((190, 130, 170), "Cánh đồng tam giác mạch Hà Giang nở rộ giữa cao nguyên."),
    (( 60, 110, 150), "Cảng Tuần Châu rộn ràng đón du khách lên tàu Hạ Long."),
    (( 50,  80, 140), "Hồ Tà Đùng, Vịnh Hạ Long thu nhỏ giữa Tây Nguyên."),
    ((200,  50,  50), "Tạm biệt Việt Nam, đất nước hình chữ S yêu thương."),
]

assert len(SCENES) == 40, f"Need 40 scenes, got {len(SCENES)}"


def _script_duration(script: str) -> float:
    words = len(script.split())
    return max(MIN_DUR, round(words / WORDS_PER_SEC, 1))


def _format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _load_fonts():
    try:
        return (
            ImageFont.truetype("arialbd.ttf", 80),
            ImageFont.truetype("arial.ttf", 48),
        )
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def _wrap_for_image(draw, text, font, max_w):
    lines, current = [], []
    for word in text.split():
        cand = " ".join([*current, word])
        if draw.textlength(cand, font=font) <= max_w or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _make_image(idx: int, color: tuple[int, int, int], text: str):
    font_title, font_small = _load_fonts()
    img = Image.new("RGB", SIZE, color)
    draw = ImageDraw.Draw(img)

    draw.text((40, 40), f"Scene {idx}", fill="white", font=font_title,
              stroke_width=3, stroke_fill="black")

    lines = _wrap_for_image(draw, text, font_title, SIZE[0] - 120)
    line_h = font_title.size + 20
    total_h = line_h * len(lines)
    y = (SIZE[1] - total_h) // 2
    for line in lines:
        lw = draw.textlength(line, font=font_title)
        x = (SIZE[0] - lw) // 2
        draw.text((x, y), line, fill="white", font=font_title,
                  stroke_width=4, stroke_fill="black")
        y += line_h

    # Watermark footer
    draw.text((40, SIZE[1] - 80), "samples/test asset", fill=(255, 255, 255, 180),
              font=font_small, stroke_width=2, stroke_fill="black")

    img.save(IMG_DIR / f"{idx}.png")


def _make_voice(total_duration: float):
    out = HERE / "voice.mp3"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{total_duration:.3f}",
        "-ac", "2", "-ar", "44100", "-b:a", "128k",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    scenes_json: list[dict] = []
    srt_blocks: list[str] = []
    cursor = 0.0

    for i, (color, script) in enumerate(SCENES, start=1):
        _make_image(i, color, script)
        dur = _script_duration(script)
        end = cursor + dur
        scenes_json.append({"scene": i, "script": script})
        srt_blocks.append(
            f"{i}\n{_format_srt_time(cursor)} --> {_format_srt_time(end)}\n{script}\n"
        )
        cursor = end

    (HERE / "scenes.json").write_text(
        json.dumps(scenes_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (HERE / "voice.srt").write_text("\n".join(srt_blocks), encoding="utf-8")
    _make_voice(cursor)

    print(f"OK: {len(SCENES)} scenes")
    print(f"Total duration: {cursor:.1f}s ({cursor/60:.2f} phút)")
    print("Files: samples/images/1..40.png, scenes.json, voice.srt, voice.mp3")


if __name__ == "__main__":
    main()
