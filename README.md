# Auto Video Editor

Tool tự động edit video từ JSON scene + SRT + ảnh + voice. Render MP4 với
Ken Burns (zoom + pan) và 15 loại transition tùy chọn.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Yêu cầu `ffmpeg` trong PATH (test: `ffmpeg -version`).

## Chạy

```powershell
streamlit run app.py
```

Mở http://localhost:8501.

## CLI

```powershell
python generator.py `
  --json samples/scenes.json --srt samples/voice.srt `
  --images samples/images --voice samples/voice.mp3 `
  --out output/test.mp4 `
  --width 1080 --height 1920 `
  --zoom-mode random --zoom-end 1.25 --pan left-right `
  --transitions "fade,slide,wipe_radial,zoom_in" `
  --transition-mode mix --transition-duration 0.6
```

## Cấu trúc input

### `scenes.json`

```json
[
  {"scene": 1, "script": "prompt mô tả ảnh"},
  {"scene": 2, "script": "..."}
]
```

**Optional per-scene override** (sẽ thay default global):

```json
[
  {
    "scene": 1,
    "script": "...",
    "zoom_kind": "in",        // "in" hoặc "out", override parity default
    "zoom_amount": 1.3,        // mức zoom max cho scene này (vd 1.3 = 130%)
    "pan": "left-right"        // 1 trong: none|left-right|right-left|up-down|down-up|random
  },
  {"scene": 2, "script": "..."}
]
```

Các field optional có thể bỏ → fallback về setting global trong UI/CLI.

### `voice.srt`

SRT chuẩn, dòng thứ N khớp scene N (cùng số lượng).

### `images/`

Đặt tên có số: `1.png`, `2.png`, ... Tool sort theo số trong tên file.
Hỗ trợ `.png`, `.jpg`, `.jpeg`, `.webp`.

### `voice.mp3` hoặc `voice.wav`

Nếu voice ngắn hơn tổng duration video → tự pad silence.
Nếu voice dài hơn → cắt theo độ dài video.

## Tham số

### Ken Burns

**Zoom mode** (pattern toàn cục, có thể override per-scene qua JSON):

| Mode | Hành vi |
|------|---------|
| `alternate` (default) | Chẵn IN, lẻ OUT |
| `alternate_reverse` | Chẵn OUT, lẻ IN |
| `all_in` | Mọi scene zoom IN |
| `all_out` | Mọi scene zoom OUT |
| `random` | Random IN/OUT mỗi scene |

- **zoom_start/zoom_end (%)**: 80-200, default 100→120
- **pan**: hướng camera quét trong khung
  - `none`: chỉ zoom center, không pan
  - `left-right`: camera quét từ trái sang phải
  - `right-left`: ngược lại
  - `up-down` / `down-up`: dọc
  - `random`: mỗi scene tự pick 1 hướng

### Transitions (15 loại)

| Loại | Mô tả |
|------|-------|
| `fade` | Crossfade |
| `slide` | Random 1 trong 4 hướng |
| `slide_left` / `slide_right` / `slide_up` / `slide_down` | B trượt theo hướng motion |
| `wipe` | Random 1 trong 4 hướng |
| `wipe_left` / `wipe_right` / `wipe_up` / `wipe_down` | Wipe mask theo hướng |
| `wipe_radial` | Iris từ tâm lan ra |
| `zoom_in` | B phình to (1.3×) co lại 1.0× + fade |
| `zoom_out` | B thu nhỏ (0.7×) lớn dần 1.0× + fade |
| `zoom_blur` | B fade-in kèm blur giảm dần |

**Mode**:
- `mix`: random từ list đã chọn cho mỗi cảnh chuyển
- `fixed`: dùng option đầu tiên cho mọi cảnh chuyển

**Duration**: 0.0-1.5s (default 0.5s)

### Subtitle (toggle)

Bật trong UI hoặc CLI flag `--subtitle`. Text lấy từ file SRT (cùng file đã
dùng làm timing scene).

**4 sync modes**:

| Mode | Hành vi |
|------|---------|
| `per_line` (default) | Mỗi dòng SRT hiện nguyên câu trong [start, end] |
| `typewriter` | Ký tự xuất hiện dần kiểu TikTok |
| `karaoke` | Highlight từng từ (màu vàng `highlight_color`) khi voice đọc đến (chia đều theo độ dài segment) |
| `phrase` | Chia câu theo dấu `, . ! ? ; :`, mỗi cụm hiện lần lượt |

**4 style presets** (override được trong UI):

| Preset | Đặc điểm |
|--------|----------|
| `TikTok` | Arial Bold 64px, viền dày 5px, vị trí 80% |
| `Cinematic` | Segoe UI 48px, viền 2px, vị trí 92% (gần cuối) |
| `Bold Caption` | Arial Bold 80px màu vàng, viền 6px, vị trí 75% |
| `Minimal` | Calibri 42px, viền 1px, vị trí 93% |

**Custom controls** (UI):
- Font dropdown: Arial, Arial Bold, Tahoma, Segoe UI, Calibri, Verdana (tất cả hỗ trợ tiếng Việt có dấu)
- Font size: 24-120px
- Màu chữ + màu viền + màu highlight (karaoke): color picker
- Độ rộng viền: 0-10px
- Vị trí: top / middle / bottom
- Y offset (% chiều cao, chỉ áp dụng khi position=bottom)

**CLI flags**:
```
--subtitle                   # bật subtitle
--subtitle-preset TikTok     # 1 trong 4 preset
--subtitle-sync per_line     # per_line | typewriter | karaoke | phrase
--subtitle-font "Arial Bold"
--subtitle-size 64
--subtitle-position bottom   # top | middle | bottom
```

## Cấu trúc thư mục

```
e:\toolEditVideo\
├─ app.py               # Streamlit UI
├─ generator.py         # Core pipeline + CLI
├─ config.py            # RenderConfig, presets
├─ utils/               # Parse JSON / SRT / list images
├─ effects/
│  ├─ zoom.py           # Ken Burns (zoom + pan)
│  ├─ transitions.py    # 15 transitions
│  └─ subtitle.py       # 4 sync modes + 4 presets
├─ samples/             # Dữ liệu test
└─ output/              # MP4 ra
```

## Roadmap

- [x] M0 — Scaffold
- [x] M1 — Pipeline CLI (parse + Ken Burns + render MP4)
- [x] M2 — Alternating zoom + 4 transitions cơ bản
- [x] M3 — Streamlit UI
- [x] M4 — Polish (validation, audio pad silence, session_state)
- [x] **Mở rộng** — Pan support + per-scene JSON override + 15 transitions
- [x] **Zoom mode** — 5 preset (alternate/all_in/all_out/random/...)
- [x] **Subtitle** — 4 sync modes (per_line/typewriter/karaoke/phrase) + 4 style presets + custom controls
- [ ] M5 (optional) — FFmpeg subprocess cho performance (5-10× faster)
- [ ] Auto generate ảnh từ prompt
