import re
from dataclasses import dataclass
from pathlib import Path

import pysrt


from models import Segment


def _to_seconds(t) -> float:
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0


def load_srt(path: Path | str) -> list[Segment]:
    subs = pysrt.open(str(path), encoding="utf-8")
    segments = [
        Segment(
            start=_to_seconds(s.start),
            end=_to_seconds(s.end),
            text=s.text.replace("\n", " ").strip(),
        )
        for s in subs
    ]
    segments.sort(key=lambda x: x.start)
    return segments


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and extra spaces for fuzzy matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def group_segments_by_script(
    segments: list[Segment],
    scripts: list[str],
) -> list[Segment]:
    """Gộp nhiều SRT segments thành N segments tương ứng với N script câu.

    Thuật toán: duyệt qua các SRT segment và tích lũy text cho đến khi
    text tích lũy khớp (fuzzy) với script câu hiện tại. Nếu hết segments
    trước khi hết scripts, các script còn lại sẽ nhận segment cuối cùng.

    Trả về list[Segment] có độ dài = len(scripts).
    """
    if len(segments) == len(scripts):
        return list(segments)

    # Chuẩn hóa scripts để so sánh
    norm_scripts = [_normalize(s) for s in scripts]

    grouped: list[Segment] = []
    seg_idx = 0
    n_segs = len(segments)

    for script_idx, norm_script in enumerate(norm_scripts):
        if seg_idx >= n_segs:
            # Hết segment, dùng lại segment cuối
            grouped.append(segments[-1])
            continue

        # Tích lũy segments cho đến khi text khớp với script
        acc_text = ""
        group_start_idx = seg_idx

        while seg_idx < n_segs:
            current_seg_text = segments[seg_idx].text
            acc_text = (acc_text + " " + current_seg_text).strip()
            seg_idx += 1

            norm_acc = _normalize(acc_text)

            # Kiểm tra nếu đây là script cuối cùng
            is_last_script = script_idx == len(norm_scripts) - 1
            if is_last_script:
                # Script cuối: gom hết segments còn lại
                if seg_idx == n_segs:
                    break
                continue

            # Script giữa: dừng khi norm_acc đạt độ dài tương đương script.
            # Nếu thêm segment tiếp theo mà vượt quá xa độ dài script thì dừng sớm.
            target_len = len(norm_script)
            if target_len == 0:
                break # Script trống: lấy 1 segment rồi dừng
            
            curr_len = len(norm_acc)
            
            # Nếu đã đạt 90% độ dài target, xem xét dừng
            if curr_len >= target_len * 0.9:
                # Nếu còn segment tiếp theo, thử xem nó có làm khớp hơn không
                if seg_idx < n_segs:
                    next_seg_norm = _normalize(segments[seg_idx].text)
                    # Nếu thêm segment tiếp theo mà tổng độ dài vượt quá 120% target, thì dừng ở đây
                    if (curr_len + len(next_seg_norm)) > target_len * 1.2:
                        break
                else:
                    break

        grouped.append(
            Segment(
                start=segments[group_start_idx].start,
                end=segments[seg_idx - 1].end,
                text=acc_text,
            )
        )

    return grouped
