#!/usr/bin/env python3
"""Aggregate the matched 24-degree/90-cm N3 full-order experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import analyze_n3_rgbd_prompt_methods as common


ROOT = Path("/home/ssu/ShelfScene")
EXPERIMENTS = {
    "rgb_only": ROOT
    / "experiments/n3_tilted24_height90_rgb_3scene_40each_4methods_5seeds_20260907",
    "rgb_plus_gray": ROOT
    / "experiments/n3_tilted24_height90_gray_3scene_40each_4methods_5seeds_20260907",
    "rgb_plus_numbered_gray": ROOT
    / "experiments/n3_tilted24_height90_numbered_gray_3scene_40each_4methods_5seeds_20260907",
}
OUTPUT = ROOT / "experiments/n3_tilted24_height90_input_ablation_3conditions_20260907"
METHODS = ("minimal_direct", "detailed_direct", "contact_point", "explicit_pairs")
SCENES = ("separated_nonoverlap", "packed_nonoverlap", "structured_occlusion")
INPUT_LABELS = {
    "rgb_only": "Numbered RGB",
    "rgb_plus_gray": "RGB + Gray",
    "rgb_plus_numbered_gray": "RGB + Numbered Gray",
}
METHOD_LABELS = {
    "minimal_direct": "Minimal direct",
    "detailed_direct": "Detailed direct",
    "contact_point": "Contact point",
    "explicit_pairs": "Explicit all pairs",
}
SCENE_LABELS = {
    "separated_nonoverlap": "분리 비가림",
    "packed_nonoverlap": "밀집 비가림",
    "structured_occlusion": "구조적 가림",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(correct: int, total: int) -> float:
    return round(100.0 * correct / total, 3) if total else 0.0


def scene_signature(rows) -> list:
    return sorted(
        (
            row["scene_uid"],
            tuple(row["visible_ids"]),
            tuple(row["expected_order"]),
            int(row["target_id"]),
            row["source_geometry_signature"],
        )
        for row in rows
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results").mkdir(exist_ok=True)
    (OUTPUT / "reports").mkdir(exist_ok=True)
    (OUTPUT / "audit").mkdir(exist_ok=True)
    (OUTPUT / "config").mkdir(exist_ok=True)

    evaluated_by_condition = {}
    frozen_by_condition = {}
    configs = {}
    condition_audits = {}
    summaries = {}
    for condition, experiment in EXPERIMENTS.items():
        config = read_json(experiment / "config/experiment_config.json")
        frozen = read_json(experiment / "config/frozen_scenes.json")
        raw = read_jsonl(experiment / "logs/runs.jsonl")
        if len(raw) != 2400 or len({row["run_id"] for row in raw}) != 2400:
            raise ValueError(f"{condition}: incomplete or duplicate calls")
        if any(row.get("finish_reason") != "stop" for row in raw):
            raise ValueError(f"{condition}: non-stop completion")
        if config["methods"] != list(METHODS):
            raise ValueError(f"{condition}: method mismatch")
        if config["main_seeds"] != [28101, 28102, 28103, 28104, 28105]:
            raise ValueError(f"{condition}: seed mismatch")
        if config["camera"]["position_cm"] != [25.0, -95.0, 90.0]:
            raise ValueError(f"{condition}: camera position mismatch")
        if config["camera"]["downward_from_horizontal_deg"] != 24.0:
            raise ValueError(f"{condition}: camera angle mismatch")
        evaluated = common.evaluate_rows(raw)
        write_jsonl(OUTPUT / "results" / f"evaluated_{condition}.jsonl", evaluated)
        evaluated_by_condition[condition] = evaluated
        frozen_by_condition[condition] = frozen
        configs[condition] = config

        grouped = defaultdict(list)
        for row in evaluated:
            grouped[(row["scene_type"], row["method"])].append(row)
        summaries[condition] = {
            scene: {
                method: common.order_summary(grouped[(scene, method)], method)
                for method in METHODS
            }
            for scene in SCENES
        }
        condition_audits[condition] = {
            "calls": len(raw),
            "unique_run_ids": len({row["run_id"] for row in raw}),
            "finish_stop": sum(row.get("finish_reason") == "stop" for row in raw),
            "scene_type_counts": dict(Counter(row["scene_type"] for row in raw)),
            "method_counts": dict(Counter(row["method"] for row in raw)),
            "seed_counts": dict(Counter(str(row["seed"]) for row in raw)),
            "format_failures": sum(not row.get("format_valid") for row in evaluated),
            "setup_geometry_audit": read_json(experiment / "audit/setup_audit.json")[
                "all_geometry_signatures_match_source"
            ],
        }

    reference = scene_signature(frozen_by_condition["rgb_only"])
    cross_condition_audit = {
        "scene_gt_geometry_signatures_identical": {
            condition: scene_signature(rows) == reference
            for condition, rows in frozen_by_condition.items()
        },
        "rgb_hashes_identical": {
            condition: sorted(row["rgb_sha256"] for row in rows)
            == sorted(row["rgb_sha256"] for row in frozen_by_condition["rgb_only"])
            for condition, rows in frozen_by_condition.items()
        },
    }
    if not all(cross_condition_audit["scene_gt_geometry_signatures_identical"].values()):
        raise ValueError("cross-condition scene/GT/geometry mismatch")
    if not all(cross_condition_audit["rgb_hashes_identical"].values()):
        raise ValueError("cross-condition numbered RGB mismatch")

    matrix_rows = []
    for scene in SCENES:
        for method in METHODS:
            row = {
                "scene_family": SCENE_LABELS[scene],
                "method": METHOD_LABELS[method],
            }
            for condition in INPUT_LABELS:
                value = summaries[condition][scene][method]
                if value["scenes"] != 40 or value["calls"] != 200:
                    raise ValueError(f"{condition}/{scene}/{method}: expected 40 scenes and 200 calls")
                row[f"{condition}_exact_pct"] = value["full_order_exact_pct"]
                row[f"{condition}_all_pairs_pct"] = value["all_object_pairs_accuracy_pct"]
            matrix_rows.append(row)
    write_csv(OUTPUT / "results/full_order_matrix.csv", matrix_rows)

    pooled = []
    for condition in INPUT_LABELS:
        exact = calls = pair_correct = pair_total = 0
        for scene in SCENES:
            for method in METHODS:
                value = summaries[condition][scene][method]
                exact += value["full_order_exact_count"]
                calls += value["calls"]
                pair_correct += value["all_object_pairs_correct"]
                pair_total += value["all_object_pairs_total"]
        pooled.append(
            {
                "condition": condition,
                "input": INPUT_LABELS[condition],
                "calls": calls,
                "full_order_exact_pct": pct(exact, calls),
                "all_pairs_accuracy_pct": pct(pair_correct, pair_total),
            }
        )
    write_csv(OUTPUT / "results/pooled.csv", pooled)

    audit = {
        "camera": {
            "position_cm": [25.0, -95.0, 90.0],
            "downward_from_horizontal_deg": 24.0,
            "focal_length_mm": 32.0,
            "render_resolution": [1280, 960],
        },
        "condition_audits": condition_audits,
        "cross_condition_audit": cross_condition_audit,
        "matrix_rows": len(matrix_rows),
        "all_cells_40_scenes_200_calls": True,
        "total_calls": sum(row["calls"] for row in pooled),
    }
    write_json(OUTPUT / "audit/final_audit.json", audit)
    write_json(
        OUTPUT / "results/summary.json",
        {"pooled": pooled, "matrix": matrix_rows, "audit": audit},
    )
    write_json(
        OUTPUT / "config/source_experiments.json",
        {condition: str(path) for condition, path in EXPERIMENTS.items()},
    )

    lines = [
        "# N3 24° downward / 90 cm — 120-scene input ablation",
        "",
        "- Three scene families × 40 scenes",
        "- Four full-order prompts × five seeds",
        "- Three input conditions; 7,200 calls total",
        "- Cell format: Full-order Exact / All-pairs Accuracy",
        "",
        "## Full-order matrix",
        "",
        "| Scene family | Prompt | Numbered RGB | RGB + Gray | RGB + Numbered Gray |",
        "|---|---|---:|---:|---:|",
    ]
    for row in matrix_rows:
        cells = []
        for condition in INPUT_LABELS:
            cells.append(
                f"{row[f'{condition}_exact_pct']:.3f} / "
                f"{row[f'{condition}_all_pairs_pct']:.3f}%"
            )
        lines.append(
            f"| {row['scene_family']} | {row['method']} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "## Pooled",
            "",
            "| Input | Full-order Exact | All-pairs | Calls |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in pooled:
        lines.append(
            f"| {row['input']} | {row['full_order_exact_pct']:.3f}% | "
            f"{row['all_pairs_accuracy_pct']:.3f}% | {row['calls']} |"
        )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "- Every cell: 40 scenes × 5 seeds = 200 calls",
            "- Scene, object geometry, IDs, GT, and RGB hashes match across inputs",
            "- Camera-only change from the 16° source geometry",
            "- Format failures and non-stop completions are recorded in audit/final_audit.json",
            "",
        ]
    )
    (OUTPUT / "reports/final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"pooled": pooled, "matrix_rows": len(matrix_rows), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
