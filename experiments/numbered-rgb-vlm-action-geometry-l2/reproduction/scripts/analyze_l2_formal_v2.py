#!/usr/bin/env python3
"""Analyze the corrected-input, formal-predicate L2 rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEW = ROOT / "experiments/vlm_action_geometry_single_info_maskfix_validation"
DEFAULT_OLD = ROOT / "experiments/vlm_action_geometry_single_info_v1"
CONDITIONS = [
    ("C0", None), ("C1", None),
    ("C2", "D4"), ("C2", "D8"),
    ("C3", "D4"), ("C3", "D8"),
    ("C4", "D4"), ("C4", "D8"),
    ("C5", "D4"), ("C5", "D8"),
    ("C6", "D4"), ("C6", "D8"),
]
CONDITION_NAMES = {
    ("C0", None): "Numbered RGB only",
    ("C1", None): "R Numeric",
    ("C2", "D4"): "R Qualitative D4",
    ("C2", "D8"): "R Qualitative D8",
    ("C3", "D4"): "F Numeric D4",
    ("C3", "D8"): "F Numeric D8",
    ("C4", "D4"): "F Qualitative D4",
    ("C4", "D8"): "F Qualitative D8",
    ("C5", "D4"): "M Numeric D4",
    ("C5", "D8"): "M Numeric D8",
    ("C6", "D4"): "M Qualitative D4",
    ("C6", "D8"): "M Qualitative D8",
}
FAMILIES = ["TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE", "FC_CLEAR", "FC_BLOCKED"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_l2(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["level"] == "L2":
                rows.append(row)
    return rows


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def metric(rows: list[dict], gt_by_scene: dict[str, bool]) -> dict:
    counts = Counter()
    for row in rows:
        truth = gt_by_scene[row["scene_id"]]
        prediction = bool(row["parsed_response"]["direct_graspable"])
        counts[(truth, prediction)] += 1
    tp, fn = counts[(True, True)], counts[(True, False)]
    tn, fp = counts[(False, False)], counts[(False, True)]
    positive_recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * positive_recall / (precision + positive_recall) if precision and positive_recall else 0.0
    balanced = (positive_recall + specificity) / 2 if positive_recall is not None and specificity is not None else None
    return {
        "n": len(rows), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "predicted_true": tp + fp,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "balanced_accuracy": balanced,
        "positive_recall": positive_recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-experiment", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--old-experiment", type=Path, default=DEFAULT_OLD)
    args = parser.parse_args()
    new_base = args.new_experiment.resolve()
    old_base = args.old_experiment.resolve()
    output_dir = new_base / "results/l2_formal_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = read_l2(new_base / "logs/l2_formal_v2/runs.jsonl")
    old_rows = read_l2(old_base / "logs/runs.jsonl")
    with (old_base / "logs/runs.jsonl").open(encoding="utf-8") as handle:
        old_l1_rows = [
            row for line in handle
            if (row := json.loads(line))["level"] == "L1"
        ]
    new_by_id = {row["run_id"]: row for row in new_rows}
    old_by_id = {row["run_id"]: row for row in old_rows}
    if len(new_rows) != 1500 or len(new_by_id) != 1500:
        raise RuntimeError(f"Expected 1500 unique new L2 rows, got {len(new_rows)}/{len(new_by_id)}")
    if set(new_by_id) != set(old_by_id):
        raise RuntimeError("Old and new L2 run IDs do not match")

    gt_by_scene = {}
    for row in new_rows:
        gt = read_json(new_base / row["ground_truth"])
        gt_by_scene[row["scene_id"]] = bool(gt["L2"]["direct_graspable"])

    grouped = defaultdict(list)
    family_grouped = defaultdict(list)
    for row in new_rows:
        grouped[condition_key(row)].append(row)
        family_grouped[(condition_key(row), row["scene_family"])].append(row)
    condition_metrics = {
        f"{condition}_{resolution or 'NA'}": {
            "condition": condition,
            "resolution": resolution,
            "name": CONDITION_NAMES[(condition, resolution)],
            **metric(grouped[(condition, resolution)], gt_by_scene),
        }
        for condition, resolution in CONDITIONS
    }
    overall_new = metric(new_rows, gt_by_scene)
    overall_old = metric(old_rows, gt_by_scene)

    exact_input_ids = [
        run_id for run_id, new_row in new_by_id.items()
        if old_by_id[run_id]["numbered_rgb_sha256"] == new_row["numbered_rgb_sha256"]
        and old_by_id[run_id]["geometry_json_sha256"] == new_row["geometry_json_sha256"]
        and old_by_id[run_id]["schema_sha256"] == new_row["schema_sha256"]
        and old_by_id[run_id]["target_object_id"] == new_row["target_object_id"]
    ]
    exact_new = metric([new_by_id[key] for key in exact_input_ids], gt_by_scene)
    exact_old = metric([old_by_id[key] for key in exact_input_ids], gt_by_scene)

    paired_correctness = Counter()
    prediction_transitions = Counter()
    for run_id in new_by_id:
        old_row, new_row = old_by_id[run_id], new_by_id[run_id]
        paired_correctness[(bool(old_row["scoring"]["task_correct"]), bool(new_row["scoring"]["task_correct"]))] += 1
        prediction_transitions[(
            bool(old_row["parsed_response"]["direct_graspable"]),
            bool(new_row["parsed_response"]["direct_graspable"]),
        )] += 1

    attempts = []
    attempts_path = new_base / "logs/l2_formal_v2/attempts.jsonl"
    with attempts_path.open(encoding="utf-8") as handle:
        attempts = [json.loads(line) for line in handle if line.strip()]
    integrity = {
        "scheduled": 1500,
        "completed": len(new_rows),
        "unique_run_ids": len(new_by_id),
        "parse_errors": sum(row["parse_error"] is not None for row in new_rows),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in new_rows),
        "infrastructure_errors": sum(row["status"] != "success" for row in attempts),
        "attempt_records": len(attempts),
    }

    positive_scene_counts = {}
    for scene_id, truth in sorted(gt_by_scene.items()):
        if not truth:
            continue
        positive_scene_counts[scene_id] = {}
        for condition in CONDITIONS:
            rows = [
                row for row in grouped[condition]
                if row["scene_id"] == scene_id
            ]
            positive_scene_counts[scene_id][f"{condition[0]}_{condition[1] or 'NA'}"] = sum(
                bool(row["parsed_response"]["direct_graspable"]) for row in rows
            )

    family_metrics = {}
    for condition in CONDITIONS:
        key = f"{condition[0]}_{condition[1] or 'NA'}"
        family_metrics[key] = {
            family: metric(family_grouped[(condition, family)], gt_by_scene)
            for family in FAMILIES
        }

    positive_scenes = {scene for scene, value in gt_by_scene.items() if value}
    l1_positive_visibility = {}
    for condition in [("C0", None), ("C1", None), ("C2", "D4"), ("C2", "D8")]:
        rows = [
            row for row in old_l1_rows
            if row["scene_id"] in positive_scenes and condition_key(row) == condition
        ]
        l1_positive_visibility[f"{condition[0]}_{condition[1] or 'NA'}"] = {
            "n": len(rows),
            "predicted_full": sum(row["parsed_response"]["visibility"] == "FULL" for row in rows),
            "task_correct": sum(bool(row["scoring"]["task_correct"]) for row in rows),
        }

    results = {
        "prompt_version": read_json(new_base / "config/experiment_config.json")["l2_prompt_version"],
        "integrity": integrity,
        "gt": {
            "positive_scenes": sorted(scene for scene, value in gt_by_scene.items() if value),
            "negative_scenes": sorted(scene for scene, value in gt_by_scene.items() if not value),
            "positive_runs": sum(gt_by_scene[row["scene_id"]] for row in new_rows),
            "negative_runs": sum(not gt_by_scene[row["scene_id"]] for row in new_rows),
        },
        "overall": {"old": overall_old, "new": overall_new},
        "condition_metrics": condition_metrics,
        "family_metrics": family_metrics,
        "positive_scene_predicted_true_out_of_5": positive_scene_counts,
        "paired_correctness": {
            "old_correct_new_correct": paired_correctness[(True, True)],
            "old_correct_new_wrong": paired_correctness[(True, False)],
            "old_wrong_new_correct": paired_correctness[(False, True)],
            "old_wrong_new_wrong": paired_correctness[(False, False)],
        },
        "prediction_transitions": {
            "false_to_false": prediction_transitions[(False, False)],
            "false_to_true": prediction_transitions[(False, True)],
            "true_to_false": prediction_transitions[(True, False)],
            "true_to_true": prediction_transitions[(True, True)],
        },
        "exact_input_prompt_only_subset": {
            "n": len(exact_input_ids),
            "scene_ids": sorted({new_by_id[key]["scene_id"] for key in exact_input_ids}),
            "old": exact_old,
            "new": exact_new,
        },
        "cross_level_evidence": {
            "description": "Original L1 results on the four GT-positive L2 scenes",
            "l1_positive_visibility": l1_positive_visibility,
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metric_columns = [
        "condition", "resolution", "name", "n", "tp", "fn", "tn", "fp",
        "predicted_true", "accuracy", "balanced_accuracy", "positive_recall",
        "specificity", "precision", "f1",
    ]
    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns)
        writer.writeheader()
        for condition in CONDITIONS:
            writer.writerow(condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"])

    with (output_dir / "family_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["condition", "resolution", "name", *FAMILIES]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for condition in CONDITIONS:
            key = f"{condition[0]}_{condition[1] or 'NA'}"
            writer.writerow({
                "condition": condition[0],
                "resolution": condition[1],
                "name": CONDITION_NAMES[condition],
                **{
                    family: family_metrics[key][family]["accuracy"]
                    for family in FAMILIES
                },
            })

    report = [
        "# L2 formal-predicate v2 rerun",
        "",
        "## Integrity",
        "",
        f"- Completed: {integrity['completed']}/{integrity['scheduled']}",
        f"- Parse errors: {integrity['parse_errors']}",
        f"- Post-validation failures: {integrity['post_validation_failures']}",
        f"- Infrastructure errors: {integrity['infrastructure_errors']}",
        "",
        "## Overall comparison",
        "",
        "| Prompt | Accuracy | Balanced accuracy | True recall | Specificity | TP | FN | TN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Original | {pct(overall_old['accuracy'])} | {pct(overall_old['balanced_accuracy'])} | {pct(overall_old['positive_recall'])} | {pct(overall_old['specificity'])} | {overall_old['tp']} | {overall_old['fn']} | {overall_old['tn']} | {overall_old['fp']} |",
        f"| Formal v2 | {pct(overall_new['accuracy'])} | {pct(overall_new['balanced_accuracy'])} | {pct(overall_new['positive_recall'])} | {pct(overall_new['specificity'])} | {overall_new['tp']} | {overall_new['fn']} | {overall_new['tn']} | {overall_new['fp']} |",
        "",
        "## Formal-v2 condition metrics",
        "",
        "| Condition | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP | Predicted true |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {row['name']} | {pct(row['accuracy'])} | "
            f"{pct(row['balanced_accuracy'])} | {pct(row['positive_recall'])} | "
            f"{pct(row['specificity'])} | {row['tp']}/{row['fn']}/{row['tn']}/{row['fp']} | "
            f"{row['predicted_true']} |"
        )
    report.extend([
        "",
        "## Accuracy by scene family",
        "",
        "| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        key = f"{condition[0]}_{condition[1] or 'NA'}"
        values = family_metrics[key]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {CONDITION_NAMES[condition]} | "
            f"{pct(values['TRANSLATE']['accuracy'])} | {pct(values['ROTATE']['accuracy'])} | "
            f"{pct(values['LIFT_AND_RELOCATE']['accuracy'])} | "
            f"{pct(values['FC_CLEAR']['accuracy'])} | {pct(values['FC_BLOCKED']['accuracy'])} |"
        )
    report.extend([
        "",
        "## Same-input prompt-only subset",
        "",
        f"There are {len(exact_input_ids)} runs whose numbered RGB, geometry, schema, and target ID are byte-identical between versions.",
        "",
        "| Prompt | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Original | {pct(exact_old['accuracy'])} | {pct(exact_old['balanced_accuracy'])} | {pct(exact_old['positive_recall'])} | {pct(exact_old['specificity'])} | {exact_old['tp']}/{exact_old['fn']}/{exact_old['tn']}/{exact_old['fp']} |",
        f"| Formal v2 | {pct(exact_new['accuracy'])} | {pct(exact_new['balanced_accuracy'])} | {pct(exact_new['positive_recall'])} | {pct(exact_new['specificity'])} | {exact_new['tp']}/{exact_new['fn']}/{exact_new['tn']}/{exact_new['fp']} |",
        "",
        "## Interpretation",
        "",
        "- The revised prompt broke the near-always-false behavior, but did not solve L2.",
        "- Overall true recall rose substantially, while false positives reduced specificity and raw accuracy.",
        "- R Numeric and R Qualitative still failed to apply their authoritative clearance rule reliably.",
        "- On the same four GT-positive scenes, the original L1 task classified visibility as FULL in 20/20 runs for each of C0, C1, C2-D4, and C2-D8. Therefore the C1/C2 L2 false negatives are not explained by an inability to recognize FULL when it is queried directly; the remaining failure is in applying or combining the formal tests inside L2.",
        "- F/M conditions produced more true answers but also most false positives; they do not directly encode the benchmark's exact clearance predicate.",
        "- Because only four scenes are GT-positive, condition-level positive recall is based on 20 runs and should not be overinterpreted.",
    ])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    hashes = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "integrity": integrity,
        "overall_old": overall_old,
        "overall_new": overall_new,
        "exact_input_old": exact_old,
        "exact_input_new": exact_new,
    }, indent=2))


if __name__ == "__main__":
    main()
