#!/usr/bin/env python3
"""Analyze and visualize every C2-D8 failure in the v25 L4 experiment."""

from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l4_blocker_conditioned_v25"
OUT = EXP / "results/c2_d8_failure_analysis"
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
PRIMARY_ORDER = [
    "PRECHECK_ERROR",
    "ACTION_WRONG",
    "OBJECT_WRONG",
    "DIRECTION_WRONG",
    "DISTANCE_WRONG",
    "PHYSICAL_FAILURE_AFTER_GT_MATCH",
]
SHORT_CODE = {
    "SUCCESS": "S",
    "PRECHECK_ERROR": "P",
    "ACTION_WRONG": "A",
    "OBJECT_WRONG": "O",
    "DIRECTION_WRONG": "R",
    "DISTANCE_WRONG": "M",
    "PHYSICAL_FAILURE_AFTER_GT_MATCH": "X",
}
COLORS = {
    "SUCCESS": "#59A14F",
    "PRECHECK_ERROR": "#B07AA1",
    "ACTION_WRONG": "#E15759",
    "OBJECT_WRONG": "#9C755F",
    "DIRECTION_WRONG": "#F28E2B",
    "DISTANCE_WRONG": "#EDC948",
    "PHYSICAL_FAILURE_AFTER_GT_MATCH": "#4E79A7",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def distance_verdict(pred: dict, gt_action: str, gt_params: dict):
    if gt_action != "TRANSLATE" or pred.get("action") != "TRANSLATE":
        return None, None, None
    predicted = pred.get("distance_cm")
    if predicted is None:
        return False, "missing", None
    if "distance_cm" in gt_params:
        target = float(gt_params["distance_cm"])
        return float(predicted) == target, "exact", target
    if "minimum_distance_cm" in gt_params:
        target = float(gt_params["minimum_distance_cm"])
        return float(predicted) >= target, "minimum", target
    return None, None, None


def classify(run: dict, replay: dict, gt: dict) -> dict:
    pred = run["parsed_response"]
    gt_action = gt["L4"]["manifest_focus_action_advisory_only"]
    gt_params = gt["L4"]["manifest_action_parameters_advisory_only"]
    gt_blockers = gt["L3"]["valid_blocker_ids"]
    action_match = pred.get("action") == gt_action
    if gt_action == "RETRIEVE":
        object_match = pred.get("object_id") == gt["target_object_id"]
    else:
        object_match = pred.get("object_id") in gt_blockers
    frozen_id = run["l3_source_response"].get("blocker_id")
    if frozen_id is None:
        object_matches_frozen_l3 = pred.get("object_id") == run["target_object_id"]
    else:
        object_matches_frozen_l3 = pred.get("object_id") == frozen_id
    direction_match = None
    if gt_action == "TRANSLATE" and pred.get("action") == "TRANSLATE":
        direction_match = pred.get("direction") == gt_params.get("direction")
    distance_match, distance_rule, gt_distance = distance_verdict(pred, gt_action, gt_params)

    if replay["action_valid"]:
        primary = "SUCCESS"
    elif not run["post_validation_pass"]:
        primary = "PRECHECK_ERROR"
    elif not action_match:
        primary = "ACTION_WRONG"
    elif not object_match:
        primary = "OBJECT_WRONG"
    elif direction_match is False:
        primary = "DIRECTION_WRONG"
    elif distance_match is False:
        primary = "DISTANCE_WRONG"
    else:
        primary = "PHYSICAL_FAILURE_AFTER_GT_MATCH"

    artifact = None
    if replay.get("cache_artifact"):
        artifact = read_json(EXP / replay["cache_artifact"])

    return {
        "run_id": run["run_id"],
        "scene_id": run["scene_id"],
        "scene_family": run["scene_family"],
        "seed": run["seed"],
        "repeat_index": run["repeat_index"],
        "numbered_rgb": run["numbered_rgb"],
        "target_object_id": run["target_object_id"],
        "l3_output": run["l3_source_response"],
        "l3_reason_correct": run["l3_source_reason_correct"],
        "l3_blocker_correct": run["l3_source_blocker_correct"],
        "l3_joint_correct": run["l3_source_joint_correct"],
        "model_raw_response": run["raw_response"],
        "model_output": pred,
        "post_validation_pass": run["post_validation_pass"],
        "post_validation_errors": run["post_validation_errors"],
        "gt": {
            "target_object_id": gt["target_object_id"],
            "blocking_reason": gt["L3"]["blocking_reason"],
            "valid_blocker_ids": gt_blockers,
            "advisory_action": gt_action,
            "advisory_parameters": gt_params,
        },
        "comparison": {
            "action_match_advisory": action_match,
            "object_matches_scene_gt": object_match,
            "object_matches_frozen_l3": object_matches_frozen_l3,
            "direction_match_advisory": direction_match,
            "distance_match_advisory": distance_match,
            "distance_rule": distance_rule,
            "gt_distance_cm": gt_distance,
        },
        "primary_class": primary,
        "replay": {
            "action_valid": replay["action_valid"],
            "simulator_executed": replay["simulator_executed"],
            "failure_reason": replay.get("failure_reason"),
            "collision": replay.get("collision"),
            "boundary_violation": replay.get("boundary_violation"),
            "post_visibility": replay.get("post_visibility"),
            "post_minimum_surface_gap_cm": replay.get("post_minimum_surface_gap_cm"),
            "post_direct_graspable": replay.get("post_direct_graspable"),
            "post_target_mask": artifact.get("post_target_mask") if artifact else None,
        },
    }


def make_matrix(records: list[dict]) -> None:
    family_rank = {name: i for i, name in enumerate(FAMILY_ORDER)}
    scenes = sorted(
        {r["scene_id"] for r in records},
        key=lambda scene: (family_rank[next(r["scene_family"] for r in records if r["scene_id"] == scene)], scene),
    )
    codes = ["SUCCESS"] + PRIMARY_ORDER
    code_to_value = {name: i for i, name in enumerate(codes)}
    matrix = np.zeros((len(scenes), 5), dtype=int)
    lookup = {(r["scene_id"], r["repeat_index"]): r for r in records}
    for i, scene in enumerate(scenes):
        for repeat in range(1, 6):
            matrix[i, repeat - 1] = code_to_value[lookup[(scene, repeat)]["primary_class"]]

    fig, ax = plt.subplots(figsize=(10.2, 11.8))
    cmap = ListedColormap([COLORS[name] for name in codes])
    ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=len(codes) - 0.5, aspect="auto")
    ax.set_xticks(range(5), ["28101", "28102", "28103", "28104", "28105"])
    ax.set_yticks(range(len(scenes)), [scene.replace("scene_", "") for scene in scenes])
    ax.set_xlabel("Seed")
    ax.set_title("C2 D8 — All 125 Runs (41 success, 84 failure)\nFailure class for every scene × seed")
    for i, scene in enumerate(scenes):
        for repeat in range(1, 6):
            rec = lookup[(scene, repeat)]
            ax.text(repeat - 1, i, SHORT_CODE[rec["primary_class"]], ha="center", va="center",
                    color="white" if rec["primary_class"] not in {"DISTANCE_WRONG"} else "#243447",
                    fontweight="bold", fontsize=9)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scenes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[Patch(color=COLORS[name], label=f"{SHORT_CODE[name]} · {name}") for name in codes],
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9,
    )
    fig.savefig(OUT / "01_all_scene_seed_matrix.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_taxonomy(failures: list[dict]) -> None:
    primary = Counter(r["primary_class"] for r in failures)
    labels = PRIMARY_ORDER
    values = [primary[name] for name in labels]
    pretty = ["Pre-check", "Action", "Object", "Direction", "Distance", "Physical\nafter match"]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar(pretty, values, color=[COLORS[name] for name in labels])
    ax.set_ylabel("Failed runs")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title("C2 D8 — Primary Failure Classification (84 failures)\nCompared with scene GT advisory, then Isaac Sim outcome")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7,
                f"{value}\n({100 * value / len(failures):.1f}%)",
                ha="center", fontweight="bold", color="#243447")
    fig.savefig(OUT / "02_primary_failure_taxonomy.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    family_counts = {family: Counter() for family in FAMILY_ORDER}
    for rec in failures:
        family_counts[rec["scene_family"]][rec["primary_class"]] += 1
    x = np.arange(len(FAMILY_ORDER))
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    bottom = np.zeros(len(FAMILY_ORDER))
    for name in PRIMARY_ORDER:
        vals = np.array([family_counts[family][name] for family in FAMILY_ORDER])
        ax.bar(x, vals, bottom=bottom, color=COLORS[name], label=name)
        bottom += vals
    ax.set_xticks(x, ["FC clear", "FC blocked", "Translate", "Rotate", "Lift & relocate"])
    ax.set_ylabel("Failed runs")
    ax.set_title("C2 D8 — Failure Composition by Scene Family")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    for i, total in enumerate(bottom.astype(int)):
        ax.text(i, total + 0.5, str(total), ha="center", fontweight="bold")
    fig.savefig(OUT / "03_family_failure_composition.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_failure_scene_contact_sheet(failures: list[dict]) -> None:
    family_rank = {name: i for i, name in enumerate(FAMILY_ORDER)}
    by_scene = defaultdict(list)
    for rec in failures:
        by_scene[rec["scene_id"]].append(rec)
    scene_ids = sorted(
        by_scene,
        key=lambda scene: (family_rank[by_scene[scene][0]["scene_family"]], scene),
    )
    fonts = load_fonts()
    cols = 4
    tile_w, tile_h = 500, 335
    rows = (len(scene_ids) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, 90 + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), f"C2 D8 — {len(scene_ids)} unique scenes containing 84 failures",
              fill="#243447", font=fonts["title"])
    for index, scene_id in enumerate(scene_ids):
        recs = by_scene[scene_id]
        row, col = divmod(index, cols)
        x, y = col * tile_w, 90 + row * tile_h
        rgb = Image.open(EXP / recs[0]["numbered_rgb"]).convert("RGB")
        rgb.thumbnail((460, 255), Image.Resampling.LANCZOS)
        canvas.paste(rgb, (x + 20 + (460 - rgb.width) // 2, y + 5))
        title = scene_id.replace("scene_", "").replace("lift_and_relocate", "lift")
        gt_action = recs[0]["gt"]["advisory_action"]
        gt_short = {
            "RETRIEVE": "RETR",
            "TRANSLATE": "TRANS",
            "ROTATE": "ROT",
            "LIFT_AND_RELOCATE": "LIFT",
        }[gt_action]
        classes = Counter(r["primary_class"] for r in recs)
        class_text = ", ".join(f"{SHORT_CODE[k]}:{v}" for k, v in classes.items())
        draw.text((x + 20, y + 270), f"{title} · fail {len(recs)}/5 · GT {gt_short}",
                  fill="#243447", font=fonts["body"])
        draw.text((x + 20, y + 301), class_text, fill="#E15759", font=fonts["body"])
    canvas.save(OUT / "04_unique_failure_scenes.png", quality=95)


def load_fonts():
    regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    try:
        return {
            "title": ImageFont.truetype(bold, 36),
            "head": ImageFont.truetype(bold, 25),
            "body": ImageFont.truetype(regular, 21),
            "mono": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18),
        }
    except OSError:
        fallback = ImageFont.load_default()
        return {"title": fallback, "head": fallback, "body": fallback, "mono": fallback}


def wrap_lines(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def make_case_panel(rec: dict, filename: str) -> None:
    fonts = load_fonts()
    canvas = Image.new("RGB", (2100, 1200), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 2100, 85), fill=COLORS[rec["primary_class"]])
    draw.text((45, 20), f"{rec['primary_class']} — {rec['scene_id']} / seed {rec['seed']}",
              fill="white", font=fonts["title"])

    rgb = Image.open(EXP / rec["numbered_rgb"]).convert("RGB")
    rgb.thumbnail((900, 660), Image.Resampling.LANCZOS)
    canvas.paste(rgb, (35 + (900 - rgb.width) // 2, 115))
    draw.text((45, 785), "Input numbered RGB", fill="#243447", font=fonts["head"])

    mask_path = rec["replay"].get("post_target_mask")
    if mask_path:
        mask = Image.open(EXP / mask_path).convert("RGB")
        mask.thumbnail((420, 270), Image.Resampling.NEAREST)
        canvas.paste(mask, (45 + (420 - mask.width) // 2, 840))
        draw.text((80, 1120), "Post target mask", fill="#243447", font=fonts["body"])

    gt = rec["gt"]
    pred = rec["model_output"]
    comp = rec["comparison"]
    replay = rec["replay"]
    right_x = 1000
    draw.text((right_x, 120), "Scene GT advisory", fill="#4E79A7", font=fonts["head"])
    gt_text = json.dumps(
        {
            "target_id": gt["target_object_id"],
            "valid_blocker_ids": gt["valid_blocker_ids"],
            "action": gt["advisory_action"],
            "parameters": gt["advisory_parameters"],
        }, ensure_ascii=False, indent=2,
    )
    draw.multiline_text((right_x, 160), gt_text, fill="#243447", font=fonts["mono"], spacing=4)

    draw.text((right_x, 500), "Model JSON", fill="#E15759", font=fonts["head"])
    pred_text = json.dumps(pred, ensure_ascii=False, indent=2)
    draw.multiline_text((right_x, 540), pred_text, fill="#243447", font=fonts["mono"], spacing=4)

    verdict = (
        f"action={comp['action_match_advisory']}  object={comp['object_matches_scene_gt']}  "
        f"direction={comp['direction_match_advisory']}  distance={comp['distance_match_advisory']}\n"
        f"sim={replay['simulator_executed']}  collision={replay['collision']}  boundary={replay['boundary_violation']}  "
        f"visibility={replay['post_visibility']}  gap={replay['post_minimum_surface_gap_cm']}  "
        f"graspable={replay['post_direct_graspable']}"
    )
    draw.text((520, 840), "Verdict", fill="#243447", font=fonts["head"])
    draw.multiline_text((520, 885), wrap_lines(verdict, 115), fill="#243447", font=fonts["body"], spacing=7)
    canvas.save(OUT / filename, quality=95)


def choose_representatives(failures: list[dict]) -> list[dict]:
    wanted = [
        ("PRECHECK_ERROR", None),
        ("ACTION_WRONG", "RETRIEVE"),
        ("ACTION_WRONG", "ROTATE"),
        ("ACTION_WRONG", "LIFT_AND_RELOCATE"),
        ("OBJECT_WRONG", None),
        ("DIRECTION_WRONG", None),
        ("DISTANCE_WRONG", None),
        ("PHYSICAL_FAILURE_AFTER_GT_MATCH", None),
    ]
    selected = []
    used = set()
    for primary, gt_action in wanted:
        candidates = [
            r for r in failures
            if r["primary_class"] == primary
            and (gt_action is None or r["gt"]["advisory_action"] == gt_action)
            and r["run_id"] not in used
        ]
        if candidates:
            rec = sorted(candidates, key=lambda r: (r["scene_id"], r["seed"]))[0]
            selected.append(rec)
            used.add(rec["run_id"])
    return selected


def write_tables(records: list[dict], failures: list[dict]) -> None:
    detailed_path = OUT / "c2_d8_failures_detailed.json"
    detailed_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "run_id", "scene_id", "scene_family", "seed", "target_id", "gt_blocker_ids", "l3_output",
        "gt_action_advisory", "gt_parameters_advisory", "model_output", "primary_class",
        "action_match", "object_match_scene_gt", "object_match_frozen_l3", "direction_match",
        "distance_match", "post_validation_pass", "post_validation_errors", "simulator_executed",
        "collision", "boundary_violation", "post_visibility", "post_gap_cm", "post_direct_graspable",
    ]
    with (OUT / "c2_d8_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in failures:
            writer.writerow({
                "run_id": rec["run_id"],
                "scene_id": rec["scene_id"],
                "scene_family": rec["scene_family"],
                "seed": rec["seed"],
                "target_id": rec["target_object_id"],
                "gt_blocker_ids": json.dumps(rec["gt"]["valid_blocker_ids"]),
                "l3_output": json.dumps(rec["l3_output"]),
                "gt_action_advisory": rec["gt"]["advisory_action"],
                "gt_parameters_advisory": json.dumps(rec["gt"]["advisory_parameters"]),
                "model_output": json.dumps(rec["model_output"]),
                "primary_class": rec["primary_class"],
                "action_match": rec["comparison"]["action_match_advisory"],
                "object_match_scene_gt": rec["comparison"]["object_matches_scene_gt"],
                "object_match_frozen_l3": rec["comparison"]["object_matches_frozen_l3"],
                "direction_match": rec["comparison"]["direction_match_advisory"],
                "distance_match": rec["comparison"]["distance_match_advisory"],
                "post_validation_pass": rec["post_validation_pass"],
                "post_validation_errors": json.dumps(rec["post_validation_errors"]),
                "simulator_executed": rec["replay"]["simulator_executed"],
                "collision": rec["replay"]["collision"],
                "boundary_violation": rec["replay"]["boundary_violation"],
                "post_visibility": rec["replay"]["post_visibility"],
                "post_gap_cm": rec["replay"]["post_minimum_surface_gap_cm"],
                "post_direct_graspable": rec["replay"]["post_direct_graspable"],
            })

    primary_counts = Counter(r["primary_class"] for r in failures)
    independent = {
        "precheck_error": sum(not r["post_validation_pass"] for r in failures),
        "action_mismatch_advisory": sum(not r["comparison"]["action_match_advisory"] for r in failures),
        "object_mismatch_scene_gt": sum(not r["comparison"]["object_matches_scene_gt"] for r in failures),
        "object_mismatch_frozen_l3": sum(not r["comparison"]["object_matches_frozen_l3"] for r in failures),
        "direction_mismatch_when_comparable": sum(r["comparison"]["direction_match_advisory"] is False for r in failures),
        "distance_mismatch_when_comparable": sum(r["comparison"]["distance_match_advisory"] is False for r in failures),
    }
    by_family = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
    for rec in records:
        by_family[rec["scene_family"]]["total"] += 1
        if rec["replay"]["action_valid"]:
            by_family[rec["scene_family"]]["success"] += 1
        else:
            by_family[rec["scene_family"]]["failure"] += 1
    summary = {
        "scope": "v25 C2 D8",
        "total_runs": len(records),
        "success": sum(r["replay"]["action_valid"] for r in records),
        "failure": len(failures),
        "primary_failure_counts": dict(primary_counts),
        "independent_mismatch_counts_among_failures": independent,
        "by_family": dict(by_family),
        "gt_note": "L4 manifest action and parameters are advisory_only; official correctness is direct Isaac Sim replay.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    runs = read_jsonl(EXP / "logs/runs.jsonl")
    replays = {row["run_id"]: row for row in read_jsonl(EXP / "logs/replay.jsonl")}
    selected = [r for r in runs if r["geometry_condition"] == "C2" and r["direction_resolution"] == "D8"]
    records = []
    for run in selected:
        gt = read_json(EXP / run["ground_truth"])
        records.append(classify(run, replays[run["run_id"]], gt))
    family_rank = {name: i for i, name in enumerate(FAMILY_ORDER)}
    records.sort(key=lambda r: (family_rank[r["scene_family"]], r["scene_id"], r["seed"]))
    failures = [r for r in records if not r["replay"]["action_valid"]]
    write_tables(records, failures)
    make_matrix(records)
    make_taxonomy(failures)
    make_failure_scene_contact_sheet(failures)
    representatives = choose_representatives(failures)
    representative_index = []
    for i, rec in enumerate(representatives, start=1):
        filename = f"case_{i:02d}_{rec['primary_class'].lower()}.png"
        make_case_panel(rec, filename)
        representative_index.append({"filename": filename, **rec})
    (OUT / "representative_cases.json").write_text(
        json.dumps(representative_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print((OUT / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps([
        {"filename": r["filename"], "run_id": r["run_id"], "class": r["primary_class"],
         "gt_action": r["gt"]["advisory_action"], "model_output": r["model_output"]}
        for r in representative_index
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
