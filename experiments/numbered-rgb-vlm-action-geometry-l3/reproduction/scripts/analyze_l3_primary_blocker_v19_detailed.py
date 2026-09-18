#!/usr/bin/env python3
"""Build a detailed, v19-only analysis package and SVG figures."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
RESULTS = EXP / "results"
VIS = RESULTS / "v19_visualizations"
CONDITIONS = [
    "C0", "C1", "C2_D4", "C2_D8", "C3_D4", "C3_D8",
    "C4_D4", "C4_D8", "C5_D4", "C5_D8", "C6_D4", "C6_D8",
]
FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
REASONS = ["OCCLUSION", "CLEARANCE_OVERLAP", "BOTH", "NONE"]
COLORS = {"reason": "#f9ab00", "blocker": "#1a73e8", "joint": "#9334e6"}


def read_rows() -> list[dict]:
    return [json.loads(x) for x in (EXP / "logs/runs.jsonl").read_text().splitlines() if x.strip()]


def cond(row: dict) -> str:
    c = row["geometry_condition"]
    return c if c in {"C0", "C1"} else f"{c}_{row['direction_resolution']}"


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 3) if d else 0.0


def start_svg(w: int, h: int, title: str, subtitle: str = "") -> list[str]:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#5f6368}.label{font-size:12px}.small{font-size:11px}.value{font-size:12px;font-weight:700}.grid{stroke:#e6e8eb;stroke-width:1}</style>',
        f'<text class="title" x="38" y="36">{escape(title)}</text>',
    ]
    if subtitle:
        out.append(f'<text class="sub" x="38" y="58">{escape(subtitle)}</text>')
    return out


def bar_chart(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float], str]], width: int = 1280) -> None:
    height, left, right, top, bottom = 590, 75, 25, 90, 475
    plot_w, plot_h = width - left - right, bottom - top
    out = start_svg(width, height, title)
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * plot_h
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="{width-right}" y2="{y}"/>', f'<text class="label" x="38" y="{y+4}">{tick}%</text>']
    group_w = plot_w / len(labels)
    bar_w = min(24, group_w / (len(series) + 1))
    for i, label in enumerate(labels):
        center = left + (i + 0.5) * group_w
        for j, (_, values, color) in enumerate(series):
            val = values[i]
            x = center + (j - (len(series)-1)/2) * (bar_w+4) - bar_w/2
            bh = val / 100 * plot_h
            out.append(f'<rect x="{x:.1f}" y="{bottom-bh:.1f}" width="{bar_w}" height="{bh:.1f}" rx="3" fill="{color}"/>')
        out.append(f'<text class="small" x="{center:.1f}" y="{bottom+21}" text-anchor="middle">{escape(label.replace("_", " "))}</text>')
    lx = width - 330
    for j, (name, _, color) in enumerate(series):
        x = lx + j * 105
        out += [f'<rect x="{x}" y="28" width="13" height="13" fill="{color}"/>', f'<text class="label" x="{x+19}" y="39">{escape(name)}</text>']
    out.append('</svg>')
    path.write_text("\n".join(out))


def overview_svg(path: Path, overall: dict, categories: dict) -> None:
    width, height, left, bottom, plot_h = 900, 470, 85, 380, 285
    metrics = [("Reason", overall["reason_pct"], COLORS["reason"]), ("Blocker ID", overall["blocker_pct"], COLORS["blocker"]), ("Joint", overall["joint_pct"], COLORS["joint"])]
    out = start_svg(width, height, "L3 primary-blocker v19: overall accuracy", "25 scenes × 12 conditions × 5 seeds = 1,500 calls")
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * plot_h
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="850" y2="{y}"/>', f'<text class="label" x="48" y="{y+4}">{tick}%</text>']
    for i, (label, value, color) in enumerate(metrics):
        x = 175 + i * 240
        bh = value / 100 * plot_h
        out += [f'<rect x="{x}" y="{bottom-bh}" width="92" height="{bh}" rx="6" fill="{color}"/>', f'<text class="value" x="{x+46}" y="{bottom-bh-10}" text-anchor="middle">{value:.1f}%</text>', f'<text class="label" x="{x+46}" y="{bottom+26}" text-anchor="middle">{label}</text>']
    out.append(f'<text class="sub" x="85" y="445">Joint outcomes: both correct {categories["both_correct"]}, reason-only {categories["reason_only"]}, blocker-only {categories["blocker_only"]}, both wrong {categories["both_wrong"]}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out))


def heatmap_svg(path: Path, matrix: dict[str, dict[str, float]]) -> None:
    width, height, left, top, cw, ch = 1260, 470, 210, 90, 82, 55
    out = start_svg(width, height, "Blocker-ID accuracy heatmap", "Rows: scene family · Columns: input condition")
    for j, c in enumerate(CONDITIONS):
        x = left + j*cw + cw/2
        out.append(f'<text class="small" x="{x}" y="{top-15}" text-anchor="middle">{escape(c.replace("_", " "))}</text>')
    for i, f in enumerate(FAMILIES):
        y = top + i*ch
        out.append(f'<text class="label" x="{left-12}" y="{y+34}" text-anchor="end">{escape(f.replace("_", " "))}</text>')
        for j, c in enumerate(CONDITIONS):
            v = matrix[f][c]
            red = int(245 - 120*v/100)
            green = int(245 - 35*v/100)
            blue = int(250 - 175*v/100)
            color = f'rgb({red},{green},{blue})'
            x = left + j*cw
            out += [f'<rect x="{x}" y="{y}" width="{cw-3}" height="{ch-3}" rx="3" fill="{color}"/>', f'<text class="value" x="{x+(cw-3)/2}" y="{y+32}" text-anchor="middle">{v:.0f}%</text>']
    out.append('</svg>')
    path.write_text("\n".join(out))


def confusion_svg(path: Path, confusion: dict[str, dict[str, int]]) -> None:
    width, height, left, top, cell = 820, 610, 250, 120, 105
    max_v = max(v for row in confusion.values() for v in row.values())
    out = start_svg(width, height, "Blocking-reason confusion matrix", "Rows: ground truth · Columns: model prediction")
    for j, p in enumerate(REASONS):
        x = left+j*cell+cell/2
        out.append(f'<text class="small" x="{x}" y="{top-18}" text-anchor="middle" transform="rotate(-25 {x} {top-18})">{escape(p)}</text>')
    for i, gt in enumerate(REASONS):
        y = top+i*cell
        out.append(f'<text class="small" x="{left-15}" y="{y+58}" text-anchor="end">{escape(gt)}</text>')
        row_total = sum(confusion[gt].values())
        for j, pred in enumerate(REASONS):
            v = confusion[gt][pred]
            intensity = v/max_v if max_v else 0
            color = f'rgb({int(245-100*intensity)},{int(248-85*intensity)},{int(252-25*intensity)})'
            x = left+j*cell
            stroke = '#188038' if gt == pred else '#ffffff'
            sw = 3 if gt == pred else 1
            out += [f'<rect x="{x}" y="{y}" width="{cell-5}" height="{cell-5}" rx="5" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>', f'<text class="value" x="{x+50}" y="{y+45}" text-anchor="middle">{v}</text>', f'<text class="small" x="{x+50}" y="{y+66}" text-anchor="middle">{pct(v,row_total):.1f}%</text>']
    out.append('</svg>')
    path.write_text("\n".join(out))


def breakdown_svg(path: Path, cats: dict, blocker_types: dict) -> None:
    width, height = 1080, 520
    out = start_svg(width, height, "Where the 1,500 calls land", "Joint decomposition and blocker-ID error types")
    sections = [
        ("Joint outcomes", [("Both correct", cats["both_correct"], "#188038"), ("Blocker only", cats["blocker_only"], "#1a73e8"), ("Reason only", cats["reason_only"], "#f9ab00"), ("Both wrong", cats["both_wrong"], "#d93025")], 95),
        ("377 blocker-ID errors", [("False positive (GT none)", blocker_types["false_positive"], "#f9ab00"), ("Wrong non-null ID", blocker_types["wrong_id"], "#d93025"), ("Null miss", blocker_types["null_miss"], "#9aa0a6")], 315),
    ]
    for title, items, y in sections:
        out.append(f'<text class="value" x="45" y="{y-18}">{escape(title)}</text>')
        total = sum(v for _,v,_ in items)
        x = 45
        for label, value, color in items:
            w = 950*value/total
            out += [f'<rect x="{x}" y="{y}" width="{w}" height="58" fill="{color}"/>', f'<text class="value" x="{x+w/2}" y="{y+25}" text-anchor="middle" fill="#fff">{value}</text>', f'<text class="small" x="{x+w/2}" y="{y+44}" text-anchor="middle" fill="#fff">{100*value/total:.1f}%</text>']
            x += w
        lx = 45
        for label, value, color in items:
            out += [f'<rect x="{lx}" y="{y+78}" width="12" height="12" fill="{color}"/>', f'<text class="small" x="{lx+18}" y="{y+89}">{escape(label)}</text>']
            lx += max(190, len(label)*7+45)
    out.append('</svg>')
    path.write_text("\n".join(out))


def seed_svg(path: Path, distribution: dict[int, int], stable: dict) -> None:
    width, height, left, bottom, plot_h = 900, 490, 90, 380, 280
    max_v = max(distribution.values())
    out = start_svg(width, height, "Blocker correctness across five seeds", "300 scene × condition groups; bar = number of groups")
    for i in range(6):
        v = distribution.get(i,0)
        x = 145+i*120
        bh = v/max_v*plot_h
        color = "#188038" if i==5 else "#1a73e8" if i>=3 else "#f9ab00" if i>=1 else "#d93025"
        out += [f'<rect x="{x}" y="{bottom-bh}" width="70" height="{bh}" rx="5" fill="{color}"/>', f'<text class="value" x="{x+35}" y="{bottom-bh-8}" text-anchor="middle">{v}</text>', f'<text class="label" x="{x+35}" y="{bottom+24}" text-anchor="middle">{i}/5 correct</text>']
    out.append(f'<text class="sub" x="90" y="455">Exact output identical across all seeds: {stable["exact_output_unanimous"]}/300 groups · blocker ID identical: {stable["blocker_unanimous"]}/300</text>')
    out.append('</svg>')
    path.write_text("\n".join(out))


def scene_svg(path: Path, scenes: list[dict]) -> None:
    width, height, left, right, top, row_h = 1050, 1030, 245, 55, 65, 36
    out = start_svg(width, height, "Blocker-ID accuracy by scene", "Each scene has 60 calls (12 conditions × 5 seeds)")
    for i, s in enumerate(sorted(scenes, key=lambda x:(-x["blocker_pct"], x["scene_id"]))):
        y = top+i*row_h
        out.append(f'<text class="small" x="{left-10}" y="{y+18}" text-anchor="end">{escape(s["scene_id"])}</text>')
        w = (width-left-right)*s["blocker_pct"]/100
        color = "#188038" if s["blocker_pct"]>=90 else "#1a73e8" if s["blocker_pct"]>=70 else "#f9ab00" if s["blocker_pct"]>=50 else "#d93025"
        out += [f'<rect x="{left}" y="{y}" width="{w}" height="24" rx="4" fill="{color}"/>', f'<text class="value" x="{left+w+8}" y="{y+17}">{s["blocker_correct"]}/60 · {s["blocker_pct"]:.1f}%</text>']
    out.append('</svg>')
    path.write_text("\n".join(out))


def main() -> None:
    rows = read_rows()
    gt_by_scene = {p.stem: json.loads(p.read_text())["L3"] for p in (EXP/"geometry/ground_truth").glob("*.json")}
    overall = Counter(n=len(rows))
    reason_confusion = {g:{p:0 for p in REASONS} for g in REASONS}
    joint_categories = Counter()
    blocker_error_types = Counter()
    grouped = defaultdict(list)
    by_scene = defaultdict(Counter)
    by_cf = defaultdict(Counter)
    examples = defaultdict(list)
    failures = []
    c2_failures = []

    for r in rows:
        sc = r["scoring"]
        rc, bc = bool(sc["blocking_reason_correct"]), bool(sc["blocker_valid_set_member"])
        overall["reason_correct"] += rc
        overall["blocker_correct"] += bc
        overall["joint_correct"] += rc and bc
        cat = "both_correct" if rc and bc else "reason_only" if rc else "blocker_only" if bc else "both_wrong"
        joint_categories[cat] += 1
        gt = gt_by_scene[r["scene_id"]]
        pred = r["parsed_response"]
        reason_confusion[gt["blocking_reason"]][pred["blocking_reason"]] += 1
        c = cond(r)
        grouped[(r["scene_id"],c)].append(r)
        for bucket in (by_scene[r["scene_id"]], by_cf[(r["scene_family"],c)]):
            bucket["n"] += 1; bucket["reason"] += rc; bucket["blocker"] += bc; bucket["joint"] += rc and bc
        if not bc:
            if not gt["valid_blocker_ids"] and pred["blocker_id"] is not None: typ="false_positive"
            elif gt["valid_blocker_ids"] and pred["blocker_id"] is None: typ="null_miss"
            else: typ="wrong_id"
            blocker_error_types[typ] += 1
            item = {"run_id":r["run_id"],"scene_id":r["scene_id"],"family":r["scene_family"],"condition":c,"seed":r["seed"],"target_object_id":r["target_object_id"],"ground_truth":gt,"output":pred,"blocker_error_type":typ,"reason_correct":rc,"image":str(EXP/r["numbered_rgb"])}
            failures.append(item)
            if c in {"C2_D4","C2_D8"}: c2_failures.append(item)
            if len(examples[typ])<5: examples[typ].append(item)
        elif not rc and len(examples["blocker_correct_reason_wrong"])<5:
            examples["blocker_correct_reason_wrong"].append({"run_id":r["run_id"],"scene_id":r["scene_id"],"condition":c,"seed":r["seed"],"target_object_id":r["target_object_id"],"ground_truth":gt,"output":pred,"image":str(EXP/r["numbered_rgb"])})

    overall.update({"reason_pct":pct(overall["reason_correct"],overall["n"]),"blocker_pct":pct(overall["blocker_correct"],overall["n"]),"joint_pct":pct(overall["joint_correct"],overall["n"])})
    seed_dist = Counter()
    exact_unanimous = blocker_unanimous = 0
    for group in grouped.values():
        seed_dist[sum(bool(r["scoring"]["blocker_valid_set_member"]) for r in group)] += 1
        exact_unanimous += len({json.dumps(r["parsed_response"],sort_keys=True) for r in group}) == 1
        blocker_unanimous += len({r["parsed_response"]["blocker_id"] for r in group}) == 1
    stable={"groups":len(grouped),"exact_output_unanimous":exact_unanimous,"blocker_unanimous":blocker_unanimous,"blocker_correct_count_distribution":dict(sorted(seed_dist.items()))}

    scene_rows=[]
    for scene,x in by_scene.items():
        scene_rows.append({"scene_id":scene,"family":scene.rsplit("_v",1)[0].replace("scene_","").upper(),"n":x["n"],"reason_correct":x["reason"],"reason_pct":pct(x["reason"],x["n"]),"blocker_correct":x["blocker"],"blocker_pct":pct(x["blocker"],x["n"]),"joint_correct":x["joint"],"joint_pct":pct(x["joint"],x["n"])})
    matrix={f:{c:pct(by_cf[(f,c)]["blocker"],by_cf[(f,c)]["n"]) for c in CONDITIONS} for f in FAMILIES}

    result={"experiment_version":"l3_primary_blocker_v19","overall":dict(overall),"joint_outcome_categories":dict(joint_categories),"blocker_error_types":dict(blocker_error_types),"reason_confusion":reason_confusion,"seed_stability":stable,"by_scene":sorted(scene_rows,key=lambda x:x["scene_id"]),"condition_family_blocker_accuracy":matrix,"representative_examples":dict(examples),"c2_blocker_failures":c2_failures,"dataset_limitation":{"zero_visible_valid_blockers":4,"one_visible_valid_blocker":21,"multiple_visible_valid_blockers":0}}
    RESULTS.mkdir(exist_ok=True); VIS.mkdir(exist_ok=True)
    (RESULTS/"v19_detailed_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (RESULTS/"v19_all_blocker_failures.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2)+"\n")
    (RESULTS/"v19_c2_blocker_failures.json").write_text(json.dumps(c2_failures,ensure_ascii=False,indent=2)+"\n")

    summary=json.loads((RESULTS/"summary.json").read_text())
    cr=summary["condition_results"]
    by_cond={x["condition_key"]:x for x in cr}
    bar_chart(VIS/"01_condition_metrics.svg","Accuracy by input condition",CONDITIONS,[("Reason",[pct(by_cond[c]["reason_correct"],by_cond[c]["n"]) for c in CONDITIONS],COLORS["reason"]),("Blocker",[pct(by_cond[c]["blocker_correct"],by_cond[c]["n"]) for c in CONDITIONS],COLORS["blocker"]),("Joint",[pct(by_cond[c]["joint_correct"],by_cond[c]["n"]) for c in CONDITIONS],COLORS["joint"])])
    fam=summary["by_family"]
    bar_chart(VIS/"02_family_metrics.svg","Accuracy by scene family",[x.replace("_"," ") for x in FAMILIES],[("Reason",[pct(fam[f]["reason_correct"],fam[f]["n"]) for f in FAMILIES],COLORS["reason"]),("Blocker",[pct(fam[f]["blocker_correct"],fam[f]["n"]) for f in FAMILIES],COLORS["blocker"]),("Joint",[pct(fam[f]["joint_correct"],fam[f]["n"]) for f in FAMILIES],COLORS["joint"])],1000)
    overview_svg(VIS/"00_overall_metrics.svg",dict(overall),dict(joint_categories))
    heatmap_svg(VIS/"03_condition_family_blocker_heatmap.svg",matrix)
    confusion_svg(VIS/"04_reason_confusion_matrix.svg",reason_confusion)
    breakdown_svg(VIS/"05_error_breakdown.svg",dict(joint_categories),dict(blocker_error_types))
    seed_svg(VIS/"06_seed_stability.svg",dict(seed_dist),stable)
    scene_svg(VIS/"07_scene_blocker_accuracy.svg",scene_rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
