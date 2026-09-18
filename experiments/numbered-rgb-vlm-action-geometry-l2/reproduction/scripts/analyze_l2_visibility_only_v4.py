#!/usr/bin/env python3
"""Analyze the visibility-only L2 v4 experiment and compare it with L1."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_l2_grasp_zone_v3 import CONDITIONS, FAMILIES, NAMES, pct, read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_visibility_only_v4"
V3_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_grasp_zone_v3"
V1_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_v1"
RUN_NAME = "l2_visibility_only_v4"


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def metric(rows: list[dict], truth: dict[str, bool], prediction) -> dict:
    counts = Counter((truth[row["scene_id"]], bool(prediction(row))) for row in rows)
    tp, fn = counts[(True, True)], counts[(True, False)]
    tn, fp = counts[(False, False)], counts[(False, True)]
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    return {
        "n": len(rows), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "predicted_true": tp + fp,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None else None
        ),
        "positive_recall": recall,
        "specificity": specificity,
        "precision": precision,
    }


def direct_prediction(row: dict) -> bool:
    return row["parsed_response"]["direct_graspable"]


def l1_full_prediction(row: dict) -> bool:
    return row["parsed_response"]["visibility"] == "FULL"


def main() -> None:
    output_dir = EXPERIMENT / "results" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    current = [
        row for row in read_jsonl(EXPERIMENT / "logs" / RUN_NAME / "runs.jsonl")
        if row["level"] == "L2"
    ]
    v3_rows = [
        row for row in read_jsonl(V3_EXPERIMENT / "logs/l2_grasp_zone_v3/runs.jsonl")
        if row["level"] == "L2"
    ]
    l1_rows = [
        row for row in read_jsonl(V1_EXPERIMENT / "logs/runs.jsonl")
        if row["level"] == "L1"
    ]
    current_by_id = {row["run_id"]: row for row in current}
    if len(current) != 1500 or len(current_by_id) != 1500:
        raise RuntimeError(f"Expected 1500 unique current rows, got {len(current)}/{len(current_by_id)}")
    if len(v3_rows) != 1500 or len(l1_rows) != 1500:
        raise RuntimeError("Expected 1500 v3 L2 rows and 1500 original L1 rows")

    truth = {}
    family = {}
    for row in current:
        scene = row["scene_id"]
        gt = read_json(EXPERIMENT / row["ground_truth"])
        truth[scene] = bool(gt["L2"]["direct_graspable"])
        family[scene] = row["scene_family"]
        if truth[scene] != (gt["L1"]["visibility"] == "FULL"):
            raise RuntimeError(f"Visibility-only GT mismatch in {scene}")

    current_grouped = defaultdict(list)
    v3_grouped = defaultdict(list)
    l1_grouped = defaultdict(list)
    family_grouped = defaultdict(list)
    for row in current:
        current_grouped[condition_key(row)].append(row)
        family_grouped[(condition_key(row), row["scene_family"])].append(row)
    for row in v3_rows:
        v3_grouped[condition_key(row)].append(row)
    for row in l1_rows:
        l1_grouped[condition_key(row)].append(row)

    condition_metrics = {}
    family_metrics = {}
    for condition in CONDITIONS:
        name = f"{condition[0]}_{condition[1] or 'NA'}"
        condition_metrics[name] = {
            "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
            "visibility_only_v4": metric(current_grouped[condition], truth, direct_prediction),
            "grasp_zone_v3_rescored": metric(v3_grouped[condition], truth, direct_prediction),
            "original_l1_full_rescored": metric(l1_grouped[condition], truth, l1_full_prediction),
        }
        family_metrics[name] = {
            scene_family: metric(family_grouped[(condition, scene_family)], truth, direct_prediction)
            for scene_family in FAMILIES
        }

    attempts = read_jsonl(EXPERIMENT / "logs" / RUN_NAME / "attempts.jsonl")
    integrity = {
        "scheduled": 1500, "completed": len(current), "unique_run_ids": len(current_by_id),
        "parse_errors": sum(row["parse_error"] is not None for row in current),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in current),
        "infrastructure_errors": sum(row["status"] != "success" for row in attempts),
        "attempt_records": len(attempts),
    }

    current_overall = metric(current, truth, direct_prediction)
    v3_overall = metric(v3_rows, truth, direct_prediction)
    l1_overall = metric(l1_rows, truth, l1_full_prediction)
    scene_metrics = []
    for scene in sorted(truth):
        rows = [row for row in current if row["scene_id"] == scene]
        predicted_true = sum(direct_prediction(row) for row in rows)
        correct = predicted_true if truth[scene] else len(rows) - predicted_true
        scene_metrics.append({
            "scene_id": scene, "scene_family": family[scene], "truth": truth[scene],
            "n": len(rows), "predicted_true": predicted_true, "correct": correct,
            "accuracy": correct / len(rows),
        })

    v3_by_id = {row["run_id"]: row for row in v3_rows}
    identical_ids = [
        run_id for run_id, row in current_by_id.items()
        if row["numbered_rgb_sha256"] == v3_by_id[run_id]["numbered_rgb_sha256"]
        and row["geometry_json_sha256"] == v3_by_id[run_id]["geometry_json_sha256"]
        and row["schema_sha256"] == v3_by_id[run_id]["schema_sha256"]
        and row["target_object_id"] == v3_by_id[run_id]["target_object_id"]
    ]
    results = {
        "prompt_version": read_json(EXPERIMENT / "config/experiment_config.json")["l2_prompt_version"],
        "integrity": integrity,
        "gt": {
            "positive_scenes": sorted(scene for scene, value in truth.items() if value),
            "negative_scenes": sorted(scene for scene, value in truth.items() if not value),
            "positive_runs": sum(truth[row["scene_id"]] for row in current),
            "negative_runs": sum(not truth[row["scene_id"]] for row in current),
        },
        "overall_same_v4_gt": {
            "visibility_only_v4": current_overall,
            "grasp_zone_v3_responses": v3_overall,
            "original_l1_FULL_responses": l1_overall,
        },
        "condition_comparison": condition_metrics,
        "family_metrics": family_metrics,
        "scene_metrics": scene_metrics,
        "byte_identical_v3_v4_input_subset": {
            "n": len(identical_ids),
            "grasp_zone_v3_responses": metric([v3_by_id[item] for item in identical_ids], truth, direct_prediction),
            "visibility_only_v4": metric([current_by_id[item] for item in identical_ids], truth, direct_prediction),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["condition", "resolution", "name", "version", "n", "tp", "fn", "tn", "fp", "accuracy", "balanced_accuracy", "positive_recall", "specificity", "precision"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for condition in CONDITIONS:
            item = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]
            for version in ("visibility_only_v4", "grasp_zone_v3_rescored", "original_l1_full_rescored"):
                writer.writerow({
                    "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
                    "version": version, **{key: value for key, value in item[version].items() if key != "predicted_true"},
                })

    with (output_dir / "family_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "resolution", "name", *FAMILIES])
        writer.writeheader()
        for condition in CONDITIONS:
            name = f"{condition[0]}_{condition[1] or 'NA'}"
            writer.writerow({
                "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
                **{item: family_metrics[name][item]["accuracy"] for item in FAMILIES},
            })

    with (output_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_metrics[0]))
        writer.writeheader()
        writer.writerows(scene_metrics)

    with (output_dir / "run_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["run_id", "scene_id", "scene_family", "condition", "resolution", "seed", "truth", "prediction", "correct"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(current, key=lambda value: value["run_id"]):
            prediction = direct_prediction(row)
            actual = truth[row["scene_id"]]
            writer.writerow({
                "run_id": row["run_id"], "scene_id": row["scene_id"], "scene_family": row["scene_family"],
                "condition": row["geometry_condition"], "resolution": row.get("direction_resolution"),
                "seed": row["seed"], "truth": actual, "prediction": prediction, "correct": actual == prediction,
            })

    report = [
        "# L2 visibility-only v4", "", "## Definition", "",
        "`direct_graspable = true` if and only if Isaac pixel-exact visibility is FULL. Object proximity and finger clearance are ignored.",
        "", "## Integrity", "",
        f"- Completed: {integrity['completed']}/{integrity['scheduled']}",
        f"- Parse errors: {integrity['parse_errors']}",
        f"- Post-validation failures: {integrity['post_validation_failures']}",
        f"- Infrastructure errors: {integrity['infrastructure_errors']}",
        "", "## Overall on the same v4 GT", "",
        "| Output formulation | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Visibility-only L2 v4 | {pct(current_overall['accuracy'])} | {pct(current_overall['balanced_accuracy'])} | {pct(current_overall['positive_recall'])} | {pct(current_overall['specificity'])} | {current_overall['tp']}/{current_overall['fn']}/{current_overall['tn']}/{current_overall['fp']} |",
        f"| Grasp-zone L2 v3 responses | {pct(v3_overall['accuracy'])} | {pct(v3_overall['balanced_accuracy'])} | {pct(v3_overall['positive_recall'])} | {pct(v3_overall['specificity'])} | {v3_overall['tp']}/{v3_overall['fn']}/{v3_overall['tn']}/{v3_overall['fp']} |",
        f"| Original L1, FULL→true | {pct(l1_overall['accuracy'])} | {pct(l1_overall['balanced_accuracy'])} | {pct(l1_overall['positive_recall'])} | {pct(l1_overall['specificity'])} | {l1_overall['tp']}/{l1_overall['fn']}/{l1_overall['tn']}/{l1_overall['fp']} |",
        "", "## Visibility-only v4 by condition", "",
        "| Condition | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]["visibility_only_v4"]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {NAMES[condition]} | {pct(row['accuracy'])} | "
            f"{pct(row['balanced_accuracy'])} | {pct(row['positive_recall'])} | {pct(row['specificity'])} | "
            f"{row['tp']}/{row['fn']}/{row['tn']}/{row['fp']} |"
        )
    report.extend([
        "", "## Accuracy by scene family", "",
        "| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        name = f"{condition[0]}_{condition[1] or 'NA'}"
        values = family_metrics[name]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {NAMES[condition]} | "
            f"{pct(values['TRANSLATE']['accuracy'])} | {pct(values['ROTATE']['accuracy'])} | "
            f"{pct(values['LIFT_AND_RELOCATE']['accuracy'])} | {pct(values['FC_CLEAR']['accuracy'])} | "
            f"{pct(values['FC_BLOCKED']['accuracy'])} |"
        )
    report.extend([
        "", "## Interpretation", "",
        "- V4 has no false negatives, but 840/960 negative runs are false positives. The model answers true whenever it can locate a visible portion of the target rather than requiring a pixel-exact FULL silhouette.",
        "- The original L1 FULL/PARTIAL/NOT_VISIBLE formulation performs substantially better on exactly the same Boolean target (balanced accuracy 74.1% versus 56.2%).",
        "- This indicates that the semantic field name and output formulation matter: asking `direct_graspable` induces a permissive graspability prior even when the prompt defines it as visibility-only.",
        "- The appropriate first-stage task is therefore the existing explicit visibility classification, or a direct `occluded` field, not a redefined `direct_graspable` Boolean.",
    ])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    sources = {
        "runs": EXPERIMENT / "logs" / RUN_NAME / "runs.jsonl",
        "attempts": EXPERIMENT / "logs" / RUN_NAME / "attempts.jsonl",
        "runtime": EXPERIMENT / "logs" / RUN_NAME / "runtime.json",
        "last_invocation_summary": EXPERIMENT / "logs" / RUN_NAME / "last_invocation_summary.json",
        "prompt_l2": EXPERIMENT / "prompts/l2_en.txt",
        "experiment_config": EXPERIMENT / "config/experiment_config.json",
        "run_table": EXPERIMENT / "config/run_table.jsonl",
        "preflight": EXPERIMENT / "audit/preflight_report.json",
        "object_coverage": EXPERIMENT / "audit/object_coverage_report.json",
    }
    (output_dir / "source_artifact_hashes.json").write_text(
        json.dumps({
            name: {"path": str(path.relative_to(EXPERIMENT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        }, indent=2) + "\n", encoding="utf-8",
    )
    hashes = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "integrity": integrity, "overall": results["overall_same_v4_gt"]}, indent=2))


if __name__ == "__main__":
    main()
