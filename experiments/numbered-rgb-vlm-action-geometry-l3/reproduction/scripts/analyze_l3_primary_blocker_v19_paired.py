#!/usr/bin/env python3
"""Paired v17/v19 analysis and lightweight SVG visualizations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V17 = ROOT / "experiments/vlm_action_geometry_single_info_l3_maskfix_v17"
V19 = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
OUT = V19 / "results"
VIS = OUT / "visualizations"
CONDITIONS = [
    "C0", "C1", "C2_D4", "C2_D8", "C3_D4", "C3_D8",
    "C4_D4", "C4_D8", "C5_D4", "C5_D8", "C6_D4", "C6_D8",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> tuple:
    return (
        row["scene_id"], row["geometry_condition"], row["direction_resolution"],
        row["repeat_index"], row["seed"],
    )


def condition(row: dict) -> str:
    c = row["geometry_condition"]
    return c if c in {"C0", "C1"} else f"{c}_{row['direction_resolution']}"


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 3) if d else 0.0


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:24px;font-weight:700}.label{font-size:13px}.value{font-size:13px;font-weight:700}.axis{stroke:#c8ccd0;stroke-width:1}.grid{stroke:#e7e9ec;stroke-width:1}</style>',
        f'<text class="title" x="40" y="38">{title}</text>',
    ]


def overall_svg(v17: dict, v19: dict, path: Path) -> None:
    width, height = 900, 470
    left, top, bottom, plot_h = 90, 80, 390, 300
    rows = [("Reason", "reason_correct"), ("Blocker ID", "blocker_correct"), ("Joint", "joint_correct")]
    out = svg_start(width, height, "Prompt change: v17 vs v19")
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * plot_h
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="850" y2="{y}"/>', f'<text class="label" x="50" y="{y+5}">{tick}%</text>']
    for i, (label, field) in enumerate(rows):
        x = 180 + i * 240
        for j, (name, data, color) in enumerate((("v17", v17, "#9aa0a6"), ("v19", v19, "#1a73e8"))):
            value = 100 * data[field] / data["n"]
            bx = x + j * 62
            bh = value / 100 * plot_h
            out += [
                f'<rect x="{bx}" y="{bottom-bh}" width="48" height="{bh}" rx="4" fill="{color}"/>',
                f'<text class="value" x="{bx+24}" y="{bottom-bh-8}" text-anchor="middle">{value:.1f}%</text>',
                f'<text class="label" x="{bx+24}" y="{bottom+22}" text-anchor="middle">{name}</text>',
            ]
        out.append(f'<text class="label" x="{x+55}" y="{bottom+48}" text-anchor="middle">{label}</text>')
    out += [
        '<rect x="620" y="20" width="14" height="14" fill="#9aa0a6"/><text class="label" x="642" y="32">v17: any valid blocker</text>',
        '<rect x="620" y="44" width="14" height="14" fill="#1a73e8"/><text class="label" x="642" y="56">v19: strongest / remove first</text>',
        '</svg>',
    ]
    path.write_text("\n".join(out))


def condition_svg(rows: list[dict], path: Path) -> None:
    width, height = 1320, 620
    left, bottom, plot_h = 80, 500, 390
    out = svg_start(width, height, "Blocker ID accuracy by input condition")
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * plot_h
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="1280" y2="{y}"/>', f'<text class="label" x="42" y="{y+5}">{tick}%</text>']
    by_condition = {r["condition"]: r for r in rows}
    for i, c in enumerate(CONDITIONS):
        r = by_condition[c]
        x = 105 + i * 97
        for j, (name, field, color) in enumerate((("v17", "v17_blocker_pct", "#9aa0a6"), ("v19", "v19_blocker_pct", "#1a73e8"))):
            value = r[field]
            bx = x + j * 33
            bh = value / 100 * plot_h
            out.append(f'<rect x="{bx}" y="{bottom-bh}" width="27" height="{bh}" rx="3" fill="{color}"/>')
        delta = r["delta_blocker_pp"]
        dcolor = "#137333" if delta > 0 else "#c5221f" if delta < 0 else "#5f6368"
        out += [
            f'<text class="label" x="{x+30}" y="{bottom+24}" text-anchor="middle">{c.replace("_", " ")}</text>',
            f'<text class="value" x="{x+30}" y="{bottom+45}" text-anchor="middle" fill="{dcolor}">{delta:+.1f}p</text>',
        ]
    out += [
        '<rect x="1020" y="28" width="14" height="14" fill="#9aa0a6"/><text class="label" x="1042" y="40">v17</text>',
        '<rect x="1090" y="28" width="14" height="14" fill="#1a73e8"/><text class="label" x="1112" y="40">v19</text>',
        '<text class="label" x="80" y="575">Numbers below conditions are v19 - v17 percentage-point changes.</text>',
        '</svg>',
    ]
    path.write_text("\n".join(out))


def transition_svg(transitions: dict, path: Path) -> None:
    width, height = 960, 390
    labels = [
        ("Correct → Correct", "correct_to_correct", "#188038"),
        ("Wrong → Correct", "wrong_to_correct", "#34a853"),
        ("Correct → Wrong", "correct_to_wrong", "#ea4335"),
        ("Wrong → Wrong", "wrong_to_wrong", "#9aa0a6"),
    ]
    max_n = max(transitions.values())
    out = svg_start(width, height, "Paired blocker-ID result transitions (1,500 matched calls)")
    for i, (label, k, color) in enumerate(labels):
        y = 82 + i * 70
        n = transitions[k]
        w = 650 * n / max_n
        out += [
            f'<text class="label" x="40" y="{y+22}">{label}</text>',
            f'<rect x="190" y="{y}" width="{w}" height="34" rx="5" fill="{color}"/>',
            f'<text class="value" x="{200+w}" y="{y+22}">{n} ({100*n/1500:.1f}%)</text>',
        ]
    out.append('</svg>')
    path.write_text("\n".join(out))


def main() -> None:
    v17_rows = read_jsonl(V17 / "logs/runs.jsonl")
    v19_rows = read_jsonl(V19 / "logs/runs.jsonl")
    old = {key(r): r for r in v17_rows}
    new = {key(r): r for r in v19_rows}
    if old.keys() != new.keys() or len(old) != 1500:
        raise SystemExit("paired run keys do not match")

    transitions = Counter()
    output_changes = 0
    blocker_changes = 0
    reason_changes = 0
    examples = defaultdict(list)
    condition_counts = defaultdict(lambda: Counter(n=0, v17_blocker=0, v19_blocker=0, v17_reason=0, v19_reason=0, v17_joint=0, v19_joint=0))
    family_counts = defaultdict(lambda: Counter(n=0, v17_blocker=0, v19_blocker=0, v17_reason=0, v19_reason=0, v17_joint=0, v19_joint=0))

    for k in sorted(old):
        a, b = old[k], new[k]
        ac = bool(a["scoring"]["blocker_valid_set_member"])
        bc = bool(b["scoring"]["blocker_valid_set_member"])
        tkey = ("correct" if ac else "wrong") + "_to_" + ("correct" if bc else "wrong")
        transitions[tkey] += 1
        pa, pb = a["parsed_response"], b["parsed_response"]
        output_changes += pa != pb
        blocker_changes += pa["blocker_id"] != pb["blocker_id"]
        reason_changes += pa["blocking_reason"] != pb["blocking_reason"]
        c = condition(b)
        f = b["scene_family"]
        for dst in (condition_counts[c], family_counts[f]):
            dst["n"] += 1
            for prefix, row in (("v17", a), ("v19", b)):
                dst[f"{prefix}_blocker"] += bool(row["scoring"]["blocker_valid_set_member"])
                dst[f"{prefix}_reason"] += bool(row["scoring"]["blocking_reason_correct"])
                dst[f"{prefix}_joint"] += bool(row["scoring"]["task_correct"])
        if len(examples[tkey]) < 8:
            gt_path = V19 / b["ground_truth"]
            gt = json.loads(gt_path.read_text())["L3"]
            examples[tkey].append({
                "scene_id": b["scene_id"], "scene_family": f, "condition": c,
                "seed": b["seed"], "target_object_id": b["target_object_id"],
                "ground_truth_reason": gt["blocking_reason"],
                "valid_blocker_ids": gt["valid_blocker_ids"],
                "v17_output": pa, "v19_output": pb,
                "image": str(V19 / b["numbered_rgb"]),
            })

    def overall(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "reason_correct": sum(bool(r["scoring"]["blocking_reason_correct"]) for r in rows),
            "blocker_correct": sum(bool(r["scoring"]["blocker_valid_set_member"]) for r in rows),
            "joint_correct": sum(bool(r["scoring"]["task_correct"]) for r in rows),
        }

    s17 = overall(v17_rows)
    s19 = overall(v19_rows)

    def summarize(grouped: dict) -> list[dict]:
        result = []
        for name, x in grouped.items():
            row = {"name": name, "n": x["n"]}
            for metric in ("blocker", "reason", "joint"):
                for version in ("v17", "v19"):
                    row[f"{version}_{metric}_correct"] = x[f"{version}_{metric}"]
                    row[f"{version}_{metric}_pct"] = pct(x[f"{version}_{metric}"], x["n"])
                row[f"delta_{metric}_pp"] = round(row[f"v19_{metric}_pct"] - row[f"v17_{metric}_pct"], 3)
            result.append(row)
        return result

    by_condition = summarize(condition_counts)
    for row in by_condition:
        row["condition"] = row.pop("name")
    by_condition.sort(key=lambda r: CONDITIONS.index(r["condition"]))
    by_family = summarize(family_counts)
    by_family.sort(key=lambda r: r["name"])

    result = {
        "paired_n": 1500,
        "v17": {**s17, "reason_pct": pct(s17["reason_correct"], 1500), "blocker_pct": pct(s17["blocker_correct"], 1500), "joint_pct": pct(s17["joint_correct"], 1500)},
        "v19": {**s19, "reason_pct": pct(s19["reason_correct"], 1500), "blocker_pct": pct(s19["blocker_correct"], 1500), "joint_pct": pct(s19["joint_correct"], 1500)},
        "delta_percentage_points": {
            "reason": round(pct(s19["reason_correct"], 1500) - pct(s17["reason_correct"], 1500), 3),
            "blocker": round(pct(s19["blocker_correct"], 1500) - pct(s17["blocker_correct"], 1500), 3),
            "joint": round(pct(s19["joint_correct"], 1500) - pct(s17["joint_correct"], 1500), 3),
        },
        "blocker_transitions": dict(transitions),
        "changed_outputs": output_changes,
        "changed_blocker_ids": blocker_changes,
        "changed_reasons": reason_changes,
        "by_condition": by_condition,
        "by_family": by_family,
        "transition_examples": dict(examples),
        "dataset_limitation": "No scene has more than one visible valid blocker ID (21 scenes have one; 4 have zero).",
    }
    OUT.mkdir(exist_ok=True)
    VIS.mkdir(exist_ok=True)
    (OUT / "paired_comparison_vs_v17.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    overall_svg(s17, s19, VIS / "v17_vs_v19_overall.svg")
    condition_svg(by_condition, VIS / "v17_vs_v19_blocker_by_condition.svg")
    transition_svg(dict(transitions), VIS / "v17_vs_v19_blocker_transitions.svg")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
