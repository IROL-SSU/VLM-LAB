#!/usr/bin/env python3
"""Plot the 4°/16°/24° N3 camera-view comparison used in Notion.

The figure intentionally excludes full-order Exact and the Explicit all-pairs
prompt.  Every displayed cell is All-pairs accuracy over 40 scenes x 5 seeds.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FRONT4 = ROOT / "experiments/n3_frontview_input_ablation_3conditions_20260901/results/summary.json"
TILTED16 = ROOT / "experiments/n3_depth_input_ablation_3conditions_full40_20260907/results/summary.json"
TILTED24 = ROOT / "experiments/n3_tilted24_height90_input_ablation_3conditions_20260907/results/summary.json"
OUT = ROOT / "experiments/n3_camera_view_effect_4_16_24_filtered_20260907"

FAMILIES = [
    ("separated_nonoverlap", "분리 비가림"),
    ("packed_nonoverlap", "밀집 비가림"),
    ("structured_occlusion", "구조적 가림"),
]
INPUTS = [
    ("numbered_rgb", "Numbered RGB"),
    ("rgb_plus_gray", "RGB + Gray"),
    ("rgb_plus_numbered_gray", "RGB + Numbered Gray"),
]
METHODS = [
    ("detailed_direct", "Detailed direct"),
    ("minimal_direct", "Minimal direct"),
    ("contact_point", "Contact point"),
]
ANGLES = ["4°", "16°", "24°"]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_rows() -> list[dict]:
    front = load(FRONT4)["front_order"]
    tilted16 = load(TILTED16)["full_order_matrix"]
    tilted24 = load(TILTED24)["matrix"]

    front_lookup = {(row["scene_type"], row["method"]): row for row in front}
    tilted16_lookup = {(row["scene_type"], row["method"]): row for row in tilted16}
    family_ko_to_id = {ko: family_id for family_id, ko in FAMILIES}
    method_label_to_id = {label: method_id for method_id, label in METHODS}
    tilted24_lookup = {
        (family_ko_to_id[row["scene_family"]], method_label_to_id[row["method"]]): row
        for row in tilted24
        if row["method"] in method_label_to_id
    }

    rows: list[dict] = []
    for family_id, family_label in FAMILIES:
        for input_id, input_label in INPUTS:
            for method_id, method_label in METHODS:
                f4 = front_lookup[(family_id, method_id)]
                f16 = tilted16_lookup[(family_id, method_id)]
                f24 = tilted24_lookup[(family_id, method_id)]

                front_field = {
                    "numbered_rgb": "rgb_only_pairwise_accuracy_pct",
                    "rgb_plus_gray": "gray_pairwise_accuracy_pct",
                    "rgb_plus_numbered_gray": "numbered_gray_pairwise_accuracy_pct",
                }[input_id]
                tilted16_field = f"{input_id}_all_pairs_accuracy_pct"
                tilted24_field = {
                    "numbered_rgb": "rgb_only_all_pairs_pct",
                    "rgb_plus_gray": "rgb_plus_gray_all_pairs_pct",
                    "rgb_plus_numbered_gray": "rgb_plus_numbered_gray_all_pairs_pct",
                }[input_id]

                if f4["scenes"] != 40 or f4["calls_per_condition"] != 200:
                    raise ValueError(f"4° sample mismatch: {family_id}/{input_id}/{method_id}")
                if f16[f"{input_id}_scenes"] != 40 or f16[f"{input_id}_calls"] != 200:
                    raise ValueError(f"16° sample mismatch: {family_id}/{input_id}/{method_id}")

                rows.append(
                    {
                        "scene_family": family_label,
                        "input": input_label,
                        "prompt": method_label,
                        "4deg_all_pairs_pct": float(f4[front_field]),
                        "16deg_all_pairs_pct": float(f16[tilted16_field]),
                        "24deg_all_pairs_pct": float(f24[tilted24_field]),
                        "scenes_per_angle": 40,
                        "seeds": 5,
                        "calls_per_angle_cell": 200,
                    }
                )
    if len(rows) != 27:
        raise ValueError(f"Expected 27 filtered rows, got {len(rows)}")
    return rows


def write_csv(rows: list[dict]) -> Path:
    result_dir = OUT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "camera_view_all_pairs_direct3.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _color(value: float) -> str:
    """Interpolate a light-to-dark sequential color for 55–100%."""
    low = (224, 242, 241)
    high = (15, 76, 129)
    ratio = max(0.0, min(1.0, (value - 55.0) / 45.0))
    rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(low, high))
    return "#%02x%02x%02x" % rgb


def plot(rows: list[dict]) -> Path:
    lookup = {(r["scene_family"], r["input"], r["prompt"]): r for r in rows}
    width, height = 1740, 1230
    left, top = 105, 180
    panel_w, panel_h = 535, 315
    label_w, cell_w, cell_h = 140, 118, 64
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Noto Sans CJK KR","Noto Sans KR",sans-serif;fill:#111827}.title{font-size:30px;font-weight:700}.subtitle{font-size:17px;fill:#4b5563}.input{font-size:20px;font-weight:700}.family{font-size:20px;font-weight:700}.angle{font-size:16px;font-weight:600;fill:#374151}.method{font-size:15px;fill:#374151}.value{font-size:17px;font-weight:700}.foot{font-size:15px;fill:#6b7280}</style>',
        f'<text x="{width/2}" y="48" text-anchor="middle" class="title">카메라 시점별 All-pairs accuracy (%)</text>',
        f'<text x="{width/2}" y="80" text-anchor="middle" class="subtitle">Exact 지표 및 Explicit all pairs 프롬프트 제외 · 각 cell: 40 scenes × 5 seeds = 200 calls</text>',
    ]

    for col_idx, (_, input_label) in enumerate(INPUTS):
        center_x = left + col_idx * panel_w + panel_w / 2
        parts.append(f'<text x="{center_x:.1f}" y="132" text-anchor="middle" class="input">{html.escape(input_label)}</text>')

    for row_idx, (_, family_label) in enumerate(FAMILIES):
        row_y = top + row_idx * panel_h
        parts.append(
            f'<text x="28" y="{row_y + 125}" text-anchor="middle" class="family" '
            f'transform="rotate(-90 28 {row_y + 125})">{html.escape(family_label)}</text>'
        )
        for col_idx, (_, input_label) in enumerate(INPUTS):
            panel_x = left + col_idx * panel_w
            for angle_idx, angle_label in enumerate(ANGLES):
                x = panel_x + label_w + angle_idx * cell_w + cell_w / 2
                parts.append(f'<text x="{x:.1f}" y="{row_y - 13}" text-anchor="middle" class="angle">{angle_label}</text>')

            for method_idx, (_, method_label) in enumerate(METHODS):
                y = row_y + method_idx * cell_h
                short_label = method_label.replace(" direct", "").replace(" point", "")
                parts.append(
                    f'<text x="{panel_x + label_w - 14}" y="{y + cell_h/2 + 5:.1f}" '
                    f'text-anchor="end" class="method">{html.escape(short_label)}</text>'
                )
                for angle_idx, angle in enumerate((4, 16, 24)):
                    value = lookup[(family_label, input_label, method_label)][f"{angle}deg_all_pairs_pct"]
                    x = panel_x + label_w + angle_idx * cell_w
                    fill = _color(value)
                    text_fill = "white" if value >= 78 else "#111827"
                    parts.extend(
                        [
                            f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="8" fill="{fill}"/>',
                            f'<text x="{x + (cell_w - 4)/2:.1f}" y="{y + (cell_h - 4)/2 + 6:.1f}" '
                            f'text-anchor="middle" class="value" style="fill:{text_fill}">{value:.1f}</text>',
                        ]
                    )

            panel_bottom = row_y + 3 * cell_h
            parts.append(
                f'<line x1="{panel_x + label_w}" y1="{panel_bottom + 12}" '
                f'x2="{panel_x + label_w + 3 * cell_w - 4}" y2="{panel_bottom + 12}" stroke="#e5e7eb"/>'
            )

    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    svg_path = figure_dir / "camera_view_all_pairs_direct3.svg"
    footer = "행: scene family · 열: 입력 조건 · 각 패널의 행: prompt · 수치는 All-pairs accuracy (%)"
    parts.append(f'<text x="{width/2}" y="{height - 28}" text-anchor="middle" class="foot">{html.escape(footer)}</text>')
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return svg_path


def plot_png(rows: list[dict]) -> Path:
    lookup = {(r["scene_family"], r["input"], r["prompt"]): r for r in rows}
    width, height = 1740, 1230
    left, top = 105, 180
    panel_w, panel_h = 535, 315
    label_w, cell_w, cell_h = 140, 118, 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    fonts = {
        "title": ImageFont.truetype(bold, 30),
        "subtitle": ImageFont.truetype(regular, 17),
        "input": ImageFont.truetype(bold, 20),
        "family": ImageFont.truetype(bold, 20),
        "angle": ImageFont.truetype(bold, 16),
        "method": ImageFont.truetype(regular, 15),
        "value": ImageFont.truetype(bold, 17),
        "foot": ImageFont.truetype(regular, 15),
    }
    draw.text((width / 2, 45), "카메라 시점별 All-pairs accuracy (%)", font=fonts["title"], anchor="mm", fill="#111827")
    draw.text(
        (width / 2, 80),
        "Exact 지표 및 Explicit all pairs 프롬프트 제외 · 각 cell: 40 scenes × 5 seeds = 200 calls",
        font=fonts["subtitle"],
        anchor="mm",
        fill="#4b5563",
    )

    for col_idx, (_, input_label) in enumerate(INPUTS):
        center_x = left + col_idx * panel_w + panel_w / 2
        draw.text((center_x, 132), input_label, font=fonts["input"], anchor="mm", fill="#111827")

    for row_idx, (_, family_label) in enumerate(FAMILIES):
        row_y = top + row_idx * panel_h
        family_layer = Image.new("RGBA", (240, 44), (255, 255, 255, 0))
        family_draw = ImageDraw.Draw(family_layer)
        family_draw.text((120, 22), family_label, font=fonts["family"], anchor="mm", fill="#111827")
        family_layer = family_layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(family_layer, (8, row_y - 8), family_layer)

        for col_idx, (_, input_label) in enumerate(INPUTS):
            panel_x = left + col_idx * panel_w
            for angle_idx, angle_label in enumerate(ANGLES):
                x = panel_x + label_w + angle_idx * cell_w + cell_w / 2
                draw.text((x, row_y - 18), angle_label, font=fonts["angle"], anchor="ms", fill="#374151")

            for method_idx, (_, method_label) in enumerate(METHODS):
                y = row_y + method_idx * cell_h
                short_label = method_label.replace(" direct", "").replace(" point", "")
                draw.text(
                    (panel_x + label_w - 14, y + (cell_h - 4) / 2),
                    short_label,
                    font=fonts["method"],
                    anchor="rm",
                    fill="#374151",
                )
                for angle_idx, angle in enumerate((4, 16, 24)):
                    value = lookup[(family_label, input_label, method_label)][f"{angle}deg_all_pairs_pct"]
                    x = panel_x + label_w + angle_idx * cell_w
                    fill = _color(value)
                    text_fill = "white" if value >= 78 else "#111827"
                    draw.rounded_rectangle(
                        (x, y, x + cell_w - 4, y + cell_h - 4),
                        radius=8,
                        fill=fill,
                    )
                    draw.text(
                        (x + (cell_w - 4) / 2, y + (cell_h - 4) / 2),
                        f"{value:.1f}",
                        font=fonts["value"],
                        anchor="mm",
                        fill=text_fill,
                    )
            panel_bottom = row_y + 3 * cell_h
            draw.line(
                (panel_x + label_w, panel_bottom + 12, panel_x + label_w + 3 * cell_w - 4, panel_bottom + 12),
                fill="#e5e7eb",
                width=1,
            )

    footer = "행: scene family · 열: 입력 조건 · 각 패널의 행: prompt · 수치는 All-pairs accuracy (%)"
    draw.text((width / 2, height - 30), footer, font=fonts["foot"], anchor="mm", fill="#6b7280")
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "camera_view_all_pairs_direct3.png"
    image.save(png_path, optimize=True)
    return png_path


def main() -> None:
    rows = build_rows()
    csv_path = write_csv(rows)
    svg = plot(rows)
    png = plot_png(rows)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "svg": str(svg), "png": str(png)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
