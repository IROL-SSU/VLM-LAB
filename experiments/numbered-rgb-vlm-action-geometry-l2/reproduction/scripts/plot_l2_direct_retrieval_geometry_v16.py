#!/usr/bin/env python3
"""Create publication-ready PNG plots for the v16 L2 retrieval experiment.

The script intentionally depends only on the Python standard library and Pillow so
that the result page can be regenerated on the experiment machine.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16/results"
OUT = RESULTS / "plots"
OUT.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MEDIUM = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

BG = "#F7F8FA"
PANEL = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
GRID = "#DDE2EA"
BLUE = "#3B82F6"
BLUE_DARK = "#1D4ED8"
ORANGE = "#F59E0B"
RED = "#E45858"
GREEN = "#2FA36B"
PURPLE = "#7C5CE7"
TEAL = "#159EAA"
LIGHT_BLUE = "#DCEBFF"
LIGHT_RED = "#FADDDD"


def font(size: int, bold: bool = False, medium: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_MEDIUM if medium else FONT_REGULAR
    return ImageFont.truetype(path, size)


def canvas(width: int = 1800, height: int = 1080):
    im = Image.new("RGB", (width, height), BG)
    return im, ImageDraw.Draw(im)


def text(draw, xy, value, size=28, fill=INK, bold=False, medium=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold=bold, medium=medium), fill=fill, anchor=anchor)


def text_width(draw, value, size=28, bold=False, medium=False):
    box = draw.textbbox((0, 0), str(value), font=font(size, bold=bold, medium=medium))
    return box[2] - box[0]


def header(draw, title, subtitle, width=1800, kicker="L2 DIRECT RETRIEVAL · V16"):
    text(draw, (90, 62), kicker, 22, BLUE_DARK, bold=True)
    text(draw, (90, 108), title, 48, INK, bold=True)
    text(draw, (90, 175), subtitle, 25, MUTED)
    draw.line((90, 222, width - 90, 222), fill=GRID, width=2)


def footer(draw, width=1800, height=1080, note=None):
    note = note or "일치율은 장면 설계 의도와 응답의 일치이며, 실제 회수 성공률이 아님"
    draw.line((90, height - 72, width - 90, height - 72), fill=GRID, width=2)
    text(draw, (90, height - 51), note, 19, MUTED)
    text(draw, (width - 90, height - 51), "25 scenes · 12 conditions · 5 seeds · N=1,500", 19, MUTED, anchor="ra")


def rounded_panel(draw, box, radius=24, fill=PANEL, outline="#E8EBF0"):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def save(im, name):
    path = OUT / name
    im.save(path, format="PNG", optimize=True)
    print(path)


def short_label(key: str) -> str:
    names = {
        "C0": "C0 · RGB only",
        "C1": "C1 · R 수치형",
        "C2_D4": "C2 D4 · R 정성",
        "C2_D8": "C2 D8 · R 정성",
        "C3_D4": "C3 D4 · F 수치",
        "C3_D8": "C3 D8 · F 수치",
        "C4_D4": "C4 D4 · F 정성",
        "C4_D8": "C4 D8 · F 정성",
        "C5_D4": "C5 D4 · M 수치",
        "C5_D8": "C5 D8 · M 수치",
        "C6_D4": "C6 D4 · M 정성",
        "C6_D8": "C6 D8 · M 정성",
    }
    return names[key]


def family_of(scene_id: str) -> str:
    if "fc_clear" in scene_id:
        return "FC_CLEAR"
    if "fc_blocked" in scene_id:
        return "FC_BLOCKED"
    if "translate" in scene_id:
        return "TRANSLATE"
    if "rotate" in scene_id:
        return "ROTATE"
    if "lift_and_relocate" in scene_id:
        return "LIFT"
    raise ValueError(scene_id)


def lerp_color(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    av = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
    bv = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
    cv = tuple(round(x + (y - x) * t) for x, y in zip(av, bv))
    return "#%02X%02X%02X" % cv


with (RESULTS / "condition_results.csv").open(newline="", encoding="utf-8") as f:
    conditions = list(csv.DictReader(f))
for row in conditions:
    for key in (
        "RETRIEVE_NOW", "REARRANGE_FIRST", "INVALID", "unanimous_scenes",
        "changed_vs_c0", "changes_toward_scene_design", "changes_away_from_scene_design",
        "scene_design_agreement_count",
    ):
        row[key] = int(row[key])
    row["agreement_rate"] = row["scene_design_agreement_count"] / 125 * 100

with (RESULTS / "retrieve_counts_by_scene.csv").open(newline="", encoding="utf-8") as f:
    scenes = list(csv.DictReader(f))
condition_keys = [row["condition_key"] for row in conditions]
for row in scenes:
    row["target_object_id"] = int(row["target_object_id"])
    row["family"] = family_of(row["scene_id"])
    for key in condition_keys:
        row[key] = int(row[key])

condition_map = {row["condition_key"]: row for row in conditions}
family_order = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT"]
family_label = {
    "FC_CLEAR": "FC_CLEAR",
    "FC_BLOCKED": "FC_BLOCKED",
    "TRANSLATE": "TRANSLATE",
    "ROTATE": "ROTATE",
    "LIFT": "LIFT",
}


def correct_count(scene, key):
    return scene[key] if scene["family"] == "FC_CLEAR" else 5 - scene[key]


family_correct = {
    key: {
        fam: sum(correct_count(s, key) for s in scenes if s["family"] == fam)
        for fam in family_order
    }
    for key in condition_keys
}


# 01 — Agreement ranking
im, d = canvas()
header(d, "전체 조건 성능 순위", "125회 기준 장면 설계 일치율 · 높은 순")
ranked = sorted(conditions, key=lambda r: r["agreement_rate"], reverse=True)
x0, x1 = 440, 1660
y0, row_h = 270, 59
for tick in range(60, 101, 10):
    x = x0 + (tick - 60) / 40 * (x1 - x0)
    d.line((x, y0 - 24, x, y0 + row_h * len(ranked) - 12), fill=GRID, width=2)
    text(d, (x, y0 - 36), f"{tick}%", 18, MUTED, anchor="ms")
for i, row in enumerate(ranked):
    y = y0 + i * row_h
    key = row["condition_key"]
    rate = row["agreement_rate"]
    text(d, (110, y + 18), f"{i + 1:02d}", 20, MUTED, bold=True, anchor="lm")
    text(d, (155, y + 18), short_label(key), 23, INK, medium=True, anchor="lm")
    bar_x = x0 + max(0, rate - 60) / 40 * (x1 - x0)
    color = BLUE_DARK if key == "C2_D8" else (BLUE if key.startswith("C2") else "#9AA9BF")
    d.rounded_rectangle((x0, y, bar_x, y + 36), radius=12, fill=color)
    text(d, (bar_x + 14, y + 18), f"{rate:.1f}%  ({row['scene_design_agreement_count']}/125)", 20, color, bold=True, anchor="lm")
footer(d)
save(im, "01_overall_agreement_ranking.png")


# 02 — Decision composition
im, d = canvas()
header(d, "응답 구성", "각 조건의 RETRIEVE_NOW / REARRANGE_FIRST 선택 비율")
x0, x1 = 430, 1655
y0, row_h = 278, 58
for i, row in enumerate(conditions):
    y = y0 + i * row_h
    total = row["RETRIEVE_NOW"] + row["REARRANGE_FIRST"]
    split = x0 + row["RETRIEVE_NOW"] / total * (x1 - x0)
    text(d, (115, y + 18), short_label(row["condition_key"]), 22, INK, medium=True, anchor="lm")
    d.rounded_rectangle((x0, y, x1, y + 36), radius=12, fill=ORANGE)
    d.rounded_rectangle((x0, y, split, y + 36), radius=12, fill=BLUE)
    text(d, (x0 + 10, y + 18), str(row["RETRIEVE_NOW"]), 18, "white", bold=True, anchor="lm")
    text(d, (x1 - 10, y + 18), str(row["REARRANGE_FIRST"]), 18, "white", bold=True, anchor="rm")
text(d, (1170, 250), "■ RETRIEVE_NOW", 19, BLUE, bold=True)
text(d, (1420, 250), "■ REARRANGE_FIRST", 19, ORANGE, bold=True)
footer(d)
save(im, "02_decision_composition.png")


# 03 — Delta vs C0
im, d = canvas()
header(d, "RGB only 대비 순효과", "C0=82.4%를 기준으로 한 장면 설계 일치율 변화(%p)")
rows = [r for r in conditions if r["condition_key"] != "C0"]
xmid, scale = 1040, 42
y0, row_h = 280, 66
d.line((xmid, y0 - 32, xmid, y0 + row_h * len(rows) - 20), fill=INK, width=3)
for tick in range(-15, 16, 5):
    x = xmid + tick * scale
    d.line((x, y0 - 28, x, y0 + row_h * len(rows) - 20), fill=GRID, width=2)
    text(d, (x, y0 - 40), f"{tick:+d}", 19, MUTED, anchor="ms")
for i, row in enumerate(rows):
    y = y0 + i * row_h
    delta = row["agreement_rate"] - condition_map["C0"]["agreement_rate"]
    x = xmid + delta * scale
    color = GREEN if delta > 0 else RED
    text(d, (140, y + 20), short_label(row["condition_key"]), 23, INK, medium=True, anchor="lm")
    d.rounded_rectangle((min(xmid, x), y, max(xmid, x), y + 40), radius=10, fill=color)
    text(d, (x + (13 if delta >= 0 else -13), y + 20), f"{delta:+.1f}%p", 21, color, bold=True, anchor="lm" if delta >= 0 else "rm")
footer(d)
save(im, "03_delta_vs_rgb_only.png")


# 04 — Paired response changes toward/away design
im, d = canvas()
header(d, "C0 대비 바뀐 응답의 방향", "동일 scene·seed에서 바뀐 응답 중 설계 의도 쪽 / 반대쪽 횟수")
xmid, scale = 980, 32
y0, row_h = 280, 66
d.line((xmid, y0 - 32, xmid, y0 + row_h * len(rows) - 20), fill=INK, width=3)
for tick in range(-20, 21, 5):
    x = xmid + tick * scale
    d.line((x, y0 - 28, x, y0 + row_h * len(rows) - 20), fill=GRID, width=2)
    text(d, (x, y0 - 40), str(abs(tick)), 18, MUTED, anchor="ms")
for i, row in enumerate(rows):
    y = y0 + i * row_h
    toward = row["changes_toward_scene_design"]
    away = row["changes_away_from_scene_design"]
    text(d, (120, y + 20), short_label(row["condition_key"]), 23, INK, medium=True, anchor="lm")
    d.rounded_rectangle((xmid - away * scale, y, xmid, y + 38), radius=9, fill=RED)
    d.rounded_rectangle((xmid, y, xmid + toward * scale, y + 38), radius=9, fill=GREEN)
    text(d, (xmid - away * scale - 10, y + 19), str(away), 20, RED, bold=True, anchor="rm")
    text(d, (xmid + toward * scale + 10, y + 19), str(toward), 20, GREEN, bold=True, anchor="lm")
text(d, (335, 245), "← 설계 의도에서 멀어짐", 19, RED, bold=True)
text(d, (1010, 245), "설계 의도에 가까워짐 →", 19, GREEN, bold=True)
footer(d)
save(im, "04_paired_changes_direction.png")


# 05 — Unanimity / seed stability
im, d = canvas()
header(d, "Seed 안정성", "5회 모두 같은 결정을 낸 장면 수 / 25")
x0, x1 = 430, 1630
y0, row_h = 278, 58
for tick in (0, 5, 10, 15, 20, 25):
    x = x0 + tick / 25 * (x1 - x0)
    d.line((x, y0 - 24, x, y0 + row_h * len(conditions) - 12), fill=GRID, width=2)
    text(d, (x, y0 - 36), str(tick), 18, MUTED, anchor="ms")
for i, row in enumerate(conditions):
    y = y0 + i * row_h
    value = row["unanimous_scenes"]
    bar_x = x0 + value / 25 * (x1 - x0)
    color = BLUE_DARK if row["condition_key"] == "C2_D8" else (TEAL if value >= 20 else "#91A3B8")
    text(d, (115, y + 18), short_label(row["condition_key"]), 22, INK, medium=True, anchor="lm")
    d.rounded_rectangle((x0, y, bar_x, y + 36), radius=12, fill=color)
    text(d, (bar_x + 12, y + 18), f"{value}/25", 20, color, bold=True, anchor="lm")
footer(d, note="만장일치는 정답 여부가 아니라 seed 간 결정 일관성을 나타냄")
save(im, "05_seed_unanimity.png")


# 06 — Family heatmap
im, d = canvas(1800, 1100)
header(d, "Family별 설계 일치 heatmap", "셀 값은 각 family 25회 중 기대 응답 횟수", height if False else 1800)
left, top = 410, 290
cw, ch = 246, 58
for j, fam in enumerate(family_order):
    text(d, (left + j * cw + cw / 2, top - 38), family_label[fam], 19, INK, bold=True, anchor="mm")
for i, key in enumerate(condition_keys):
    y = top + i * ch
    text(d, (115, y + ch / 2), short_label(key), 22, INK, medium=True, anchor="lm")
    for j, fam in enumerate(family_order):
        value = family_correct[key][fam]
        t = value / 25
        color = lerp_color("#F7C9C9", "#2474D2", t)
        x = left + j * cw
        d.rounded_rectangle((x + 4, y + 4, x + cw - 4, y + ch - 4), radius=9, fill=color)
        text(d, (x + cw / 2, y + ch / 2), f"{value}/25", 22, "white" if t >= .55 else INK, bold=True, anchor="mm")
for j in range(6):
    x = 610 + j * 150
    val = j * 5
    d.rectangle((x, 1000, x + 150, 1024), fill=lerp_color("#F7C9C9", "#2474D2", val / 25))
    text(d, (x, 1048), str(val), 16, MUTED, anchor="ms")
footer(d, width=1800, height=1100)
save(im, "06_family_condition_heatmap.png")


# 07 — Scene × condition heatmap
im, d = canvas(2100, 1500)
header(d, "장면별 일치 패턴", "셀 값은 5개 seed 중 장면 설계와 일치한 횟수", width=2100)
ordered_scenes = sorted(scenes, key=lambda s: (family_order.index(s["family"]), s["scene_id"]))
left, top = 550, 310
cw, ch = 121, 42
for j, key in enumerate(condition_keys):
    text(d, (left + j * cw + cw / 2, top - 55), key.replace("_", " "), 16, INK, bold=True, anchor="mm")
for i, scene in enumerate(ordered_scenes):
    y = top + i * ch
    if i and ordered_scenes[i - 1]["family"] != scene["family"]:
        d.line((95, y, 2020, y), fill=INK, width=3)
    label = scene["scene_id"].replace("scene_", "")
    text(d, (105, y + ch / 2), label, 19, INK, medium=True, anchor="lm")
    text(d, (485, y + ch / 2), scene["family"], 14, MUTED, anchor="rm")
    for j, key in enumerate(condition_keys):
        value = correct_count(scene, key)
        x = left + j * cw
        color = lerp_color("#F4C4C4", "#256FD1", value / 5)
        d.rectangle((x + 2, y + 2, x + cw - 2, y + ch - 2), fill=color)
        text(d, (x + cw / 2, y + ch / 2), str(value), 17, "white" if value >= 3 else INK, bold=True, anchor="mm")
footer(d, width=2100, height=1500)
save(im, "07_scene_condition_heatmap.png")


# 08 — D4 vs D8 paired comparison
im, d = canvas()
header(d, "D4 ↔ D8 비교", "동일 정보 계열에서 방향 분해 수준이 결과와 안정성에 미친 변화")
pairs = [
    ("R 정성", "C2_D4", "C2_D8"),
    ("F 수치", "C3_D4", "C3_D8"),
    ("F 정성", "C4_D4", "C4_D8"),
    ("M 수치", "C5_D4", "C5_D8"),
    ("M 정성", "C6_D4", "C6_D8"),
]
rounded_panel(d, (90, 260, 885, 955))
rounded_panel(d, (915, 260, 1710, 955))
text(d, (145, 305), "장면 설계 일치율 (%)", 27, INK, bold=True)
text(d, (970, 305), "5회 만장일치 장면 (/25)", 27, INK, bold=True)
for panel_x, metric, lo, hi, suffix in [
    (145, "agreement_rate", 65, 100, "%"),
    (970, "unanimous_scenes", 12, 25, ""),
]:
    line_x0, line_x1 = panel_x + 215, panel_x + 680
    for t in range(lo, hi + 1, 5 if metric == "agreement_rate" else 1):
        if metric == "unanimous_scenes" and t % 2:
            continue
        x = line_x0 + (t - lo) / (hi - lo) * (line_x1 - line_x0)
        d.line((x, 360, x, 900), fill=GRID, width=1)
        text(d, (x, 925), f"{t}{suffix}", 16, MUTED, anchor="ms")
    for i, (label, d4, d8) in enumerate(pairs):
        y = 405 + i * 99
        v4, v8 = condition_map[d4][metric], condition_map[d8][metric]
        x4 = line_x0 + (v4 - lo) / (hi - lo) * (line_x1 - line_x0)
        x8 = line_x0 + (v8 - lo) / (hi - lo) * (line_x1 - line_x0)
        text(d, (panel_x, y), label, 22, INK, medium=True, anchor="lm")
        d.line((x4, y, x8, y), fill="#AAB5C3", width=6)
        d.ellipse((x4 - 10, y - 10, x4 + 10, y + 10), fill=ORANGE)
        d.ellipse((x8 - 11, y - 11, x8 + 11, y + 11), fill=BLUE_DARK)
        text(d, (x4, y - 28), f"{v4:.1f}" if isinstance(v4, float) else str(v4), 17, ORANGE, bold=True, anchor="mm")
        text(d, (x8, y + 29), f"{v8:.1f}" if isinstance(v8, float) else str(v8), 17, BLUE_DARK, bold=True, anchor="mm")
text(d, (660, 970), "● D4", 20, ORANGE, bold=True)
text(d, (780, 970), "● D8", 20, BLUE_DARK, bold=True)
footer(d)
save(im, "08_d4_vs_d8.png")


# 09 — Focused family profile
im, d = canvas()
header(d, "R 정보가 바꾼 family별 성능", "C0, R 수치형(C1), R 정성형 D4/D8의 기대 응답 횟수 비교")
focus = ["C0", "C1", "C2_D4", "C2_D8"]
colors = ["#9AA9BF", PURPLE, TEAL, BLUE_DARK]
left, top, chart_w, chart_h = 180, 300, 1460, 560
for tick in (0, 5, 10, 15, 20, 25):
    y = top + chart_h - tick / 25 * chart_h
    d.line((left, y, left + chart_w, y), fill=GRID, width=2)
    text(d, (left - 20, y), str(tick), 18, MUTED, anchor="rm")
group_w = chart_w / len(family_order)
bar_w = 54
for j, fam in enumerate(family_order):
    cx = left + group_w * (j + .5)
    text(d, (cx, top + chart_h + 44), family_label[fam], 20, INK, bold=True, anchor="mm")
    for k, key in enumerate(focus):
        val = family_correct[key][fam]
        x = cx + (k - 1.5) * (bar_w + 10)
        y = top + chart_h - val / 25 * chart_h
        d.rounded_rectangle((x - bar_w / 2, y, x + bar_w / 2, top + chart_h), radius=9, fill=colors[k])
        text(d, (x, y - 17), str(val), 17, colors[k], bold=True, anchor="mm")
for k, key in enumerate(focus):
    x = 490 + k * 270
    d.rounded_rectangle((x, 934, x + 30, 964), radius=7, fill=colors[k])
    text(d, (x + 42, 949), short_label(key), 18, INK, medium=True, anchor="lm")
footer(d)
save(im, "09_r_information_family_profile.png")


# 10 — C2 D8 residual errors
im, d = canvas()
header(d, "최고 조건 C2 D8의 잔여 오차", "장면별 5개 seed 중 설계 불일치 횟수 · 22개 장면은 0회")
error_rows = []
for scene in ordered_scenes:
    errors = 5 - correct_count(scene, "C2_D8")
    error_rows.append((scene["scene_id"], scene["family"], errors))
error_rows.sort(key=lambda x: (-x[2], family_order.index(x[1]), x[0]))
left, right, top = 690, 1630, 280
row_h = 29
for tick in range(6):
    x = left + tick / 5 * (right - left)
    d.line((x, top - 25, x, top + row_h * 25), fill=GRID, width=2)
    text(d, (x, top - 38), str(tick), 18, MUTED, anchor="ms")
for i, (scene_id, fam, errors) in enumerate(error_rows):
    y = top + i * row_h
    label = scene_id.replace("scene_", "")
    text(d, (115, y + 10), label, 17, INK, medium=errors > 0, anchor="lm")
    text(d, (615, y + 10), fam, 13, MUTED, anchor="rm")
    x = left + errors / 5 * (right - left)
    d.line((left, y + 10, x, y + 10), fill=RED if errors else "#C9D1DC", width=8)
    d.ellipse((x - 8, y + 2, x + 8, y + 18), fill=RED if errors else "#B8C2CE")
    if errors:
        text(d, (x + 16, y + 10), str(errors), 18, RED, bold=True, anchor="lm")
rounded_panel(d, (1210, 870, 1650, 965), fill="#EFF6FF", outline="#C8DCF9")
text(d, (1240, 898), "총 8/125회 불일치", 24, BLUE_DARK, bold=True)
text(d, (1240, 937), "3개 장면에 집중", 21, MUTED, medium=True)
footer(d)
save(im, "10_c2d8_residual_errors.png")


# 11 — Clear-vs-blocked operating point
im, d = canvas()
header(d, "‘바로 회수’ 보존과 ‘먼저 정리’ 검출의 균형", "가로: FC_CLEAR 기대 응답률 · 세로: 나머지 4 family 기대 응답률")
left, top, right, bottom = 220, 290, 1590, 875
for tick in range(0, 101, 10):
    x = left + tick / 100 * (right - left)
    y = bottom - tick / 100 * (bottom - top)
    d.line((x, top, x, bottom), fill=GRID, width=2)
    d.line((left, y, right, y), fill=GRID, width=2)
    text(d, (x, bottom + 28), str(tick), 17, MUTED, anchor="ms")
    text(d, (left - 18, y), str(tick), 17, MUTED, anchor="rm")
text(d, ((left + right) / 2, 940), "FC_CLEAR: RETRIEVE_NOW 기대 응답률 (%)", 22, INK, bold=True, anchor="mm")
text(d, (left + 16, top + 18), "세로: 나머지 4 family의 REARRANGE_FIRST 기대 응답률 (%)", 17, INK, bold=True, anchor="la")
offsets = [(12, -18), (12, 18), (12, -16), (12, 16), (12, -18), (12, 18), (12, -16), (12, 16), (12, -18), (12, 18), (12, -16), (12, 16)]
for idx, row in enumerate(conditions):
    key = row["condition_key"]
    clear = family_correct[key]["FC_CLEAR"] / 25 * 100
    blocked = sum(family_correct[key][f] for f in family_order[1:]) / 100 * 100
    x = left + clear / 100 * (right - left)
    y = bottom - blocked / 100 * (bottom - top)
    color = BLUE_DARK if key == "C2_D8" else (BLUE if key.startswith("C2") else "#7B8CA4")
    radius = 16 if key == "C2_D8" else 11
    d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)
    dx, dy = offsets[idx]
    text(d, (x + dx, y + dy), key.replace("_", " "), 16, color, bold=True, anchor="lm")
footer(d)
save(im, "11_clear_vs_blocked_operating_point.png")


# 12 — Accuracy vs stability scatter
im, d = canvas()
header(d, "성능–안정성 지도", "가로: 장면 설계 일치율 · 세로: 5회 만장일치 장면 수")
left, top, right, bottom = 220, 290, 1590, 875
for tick in range(65, 101, 5):
    x = left + (tick - 65) / 35 * (right - left)
    d.line((x, top, x, bottom), fill=GRID, width=2)
    text(d, (x, bottom + 28), str(tick), 17, MUTED, anchor="ms")
for tick in range(12, 26, 2):
    y = bottom - (tick - 12) / 13 * (bottom - top)
    d.line((left, y, right, y), fill=GRID, width=2)
    text(d, (left - 18, y), str(tick), 17, MUTED, anchor="rm")
text(d, ((left + right) / 2, 940), "장면 설계 일치율 (%)", 22, INK, bold=True, anchor="mm")
text(d, (left + 16, top + 18), "세로: 5회 만장일치 장면 (/25)", 18, INK, bold=True, anchor="la")
for idx, row in enumerate(conditions):
    key = row["condition_key"]
    x = left + (row["agreement_rate"] - 65) / 35 * (right - left)
    y = bottom - (row["unanimous_scenes"] - 12) / 13 * (bottom - top)
    color = BLUE_DARK if key == "C2_D8" else (BLUE if key.startswith("C2") else "#7B8CA4")
    radius = 17 if key == "C2_D8" else 11
    d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)
    dy = -22 if idx % 2 == 0 else 22
    text(d, (x + 14, y + dy), key.replace("_", " "), 17, color, bold=True, anchor="lm")
rounded_panel(d, (1190, 320, 1555, 410), fill="#EFF6FF", outline="#C8DCF9")
text(d, (1220, 346), "우상단일수록", 21, BLUE_DARK, bold=True)
text(d, (1220, 379), "높은 일치 + 높은 안정성", 19, MUTED, medium=True)
footer(d)
save(im, "12_agreement_vs_stability.png")
