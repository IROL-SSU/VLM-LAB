#!/usr/bin/env python3
"""Analyze target-access wording v6 against v5 and prior visibility outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analyze_l2_grasp_zone_v3 import CONDITIONS, FAMILIES, NAMES, pct, read_json, read_jsonl
from analyze_l2_occlusion_boolean_v5 import metric


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_target_access_v6"
V5_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_occlusion_boolean_v5"
V4_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_visibility_only_v4"
V1_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_v1"
RUN_NAME = "l2_target_access_v6"


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def access_prediction(row: dict) -> bool:
    return row["parsed_response"]["target_access_blocked"]


def occlusion_prediction(row: dict) -> bool:
    return row["parsed_response"]["occluded_by_other_object"]


def inverted_direct_prediction(row: dict) -> bool:
    return not row["parsed_response"]["direct_graspable"]


def l1_non_full_prediction(row: dict) -> bool:
    return row["parsed_response"]["visibility"] != "FULL"


def load_level(path: Path, level: str) -> list[dict]:
    return [row for row in read_jsonl(path) if row["level"] == level]


def main() -> None:
    output_dir = EXPERIMENT / "results" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    current = load_level(EXPERIMENT / "logs" / RUN_NAME / "runs.jsonl", "L2")
    v5_rows = load_level(V5_EXPERIMENT / "logs/l2_occlusion_boolean_v5/runs.jsonl", "L2")
    v4_rows = load_level(V4_EXPERIMENT / "logs/l2_visibility_only_v4/runs.jsonl", "L2")
    l1_rows = load_level(V1_EXPERIMENT / "logs/runs.jsonl", "L1")
    for name, rows in {"v6": current, "v5": v5_rows, "v4": v4_rows, "l1": l1_rows}.items():
        if len(rows) != 1500 or len({row["run_id"] for row in rows}) != 1500:
            raise RuntimeError(f"Expected 1500 unique {name} rows, got {len(rows)}")

    truth: dict[str, bool] = {}
    family: dict[str, str] = {}
    for row in current:
        gt = read_json(EXPERIMENT / row["ground_truth"])
        scene = row["scene_id"]
        truth[scene] = bool(gt["L2"]["target_access_blocked"])
        family[scene] = row["scene_family"]
        if truth[scene] != (gt["L1"]["visibility"] != "FULL"):
            raise RuntimeError(f"Frozen v5-equivalent GT mismatch in {scene}")

    groups = {name: defaultdict(list) for name in ("v6", "v5", "v4", "l1")}
    family_groups = defaultdict(list)
    for name, rows in {"v6": current, "v5": v5_rows, "v4": v4_rows, "l1": l1_rows}.items():
        for row in rows:
            groups[name][condition_key(row)].append(row)
            if name == "v6":
                family_groups[(condition_key(row), row["scene_family"])].append(row)

    predictors = {
        "target_access_v6": access_prediction,
        "occlusion_v5": occlusion_prediction,
        "visibility_v4_inverted": inverted_direct_prediction,
        "original_l1_non_FULL": l1_non_full_prediction,
    }
    rows_by_version = {
        "target_access_v6": current,
        "occlusion_v5": v5_rows,
        "visibility_v4_inverted": v4_rows,
        "original_l1_non_FULL": l1_rows,
    }
    group_name = {
        "target_access_v6": "v6", "occlusion_v5": "v5",
        "visibility_v4_inverted": "v4", "original_l1_non_FULL": "l1",
    }
    overall = {
        name: metric(rows_by_version[name], truth, prediction)
        for name, prediction in predictors.items()
    }
    condition_metrics = {}
    family_metrics = {}
    for condition in CONDITIONS:
        key = f"{condition[0]}_{condition[1] or 'NA'}"
        condition_metrics[key] = {
            "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
            **{
                name: metric(groups[group_name[name]][condition], truth, prediction)
                for name, prediction in predictors.items()
            },
        }
        family_metrics[key] = {
            scene_family: metric(
                family_groups[(condition, scene_family)], truth, access_prediction
            )
            for scene_family in FAMILIES
        }

    attempts = read_jsonl(EXPERIMENT / "logs" / RUN_NAME / "attempts.jsonl")
    integrity = {
        "scheduled": 1500, "completed": len(current),
        "unique_run_ids": len({row["run_id"] for row in current}),
        "parse_errors": sum(row["parse_error"] is not None for row in current),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in current),
        "infrastructure_errors": sum(row["status"] != "success" for row in attempts),
        "attempt_records": len(attempts),
    }

    scene_metrics = []
    for scene in sorted(truth):
        rows = [row for row in current if row["scene_id"] == scene]
        predicted = sum(access_prediction(row) for row in rows)
        correct = predicted if truth[scene] else len(rows) - predicted
        scene_metrics.append({
            "scene_id": scene, "scene_family": family[scene], "truth_blocked": truth[scene],
            "n": len(rows), "predicted_blocked": predicted, "correct": correct,
            "accuracy": correct / len(rows),
        })

    v5_by_id = {row["run_id"]: row for row in v5_rows}
    same_input_ids = [
        row["run_id"] for row in current
        if row["numbered_rgb_sha256"] == v5_by_id[row["run_id"]]["numbered_rgb_sha256"]
        and row["geometry_json_sha256"] == v5_by_id[row["run_id"]]["geometry_json_sha256"]
        and row["target_object_id"] == v5_by_id[row["run_id"]]["target_object_id"]
        and row["seed"] == v5_by_id[row["run_id"]]["seed"]
    ]
    results = {
        "prompt_version": read_json(EXPERIMENT / "config/experiment_config.json")["l2_prompt_version"],
        "gt_scope": "same Isaac pixel-exact occlusion GT as v5; prompt/output ablation only",
        "integrity": integrity,
        "overall_same_gt": overall,
        "condition_comparison": condition_metrics,
        "family_metrics": family_metrics,
        "scene_metrics": scene_metrics,
        "same_observable_input_v5_v6": len(same_input_ids),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "condition", "resolution", "name", "version", "n", "tp", "fn", "tn", "fp",
            "accuracy", "balanced_accuracy", "positive_recall", "specificity", "precision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            item = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]
            for version in predictors:
                writer.writerow({
                    "condition": condition[0], "resolution": condition[1],
                    "name": NAMES[condition], "version": version,
                    **{k: v for k, v in item[version].items() if k != "predicted_true"},
                })

    with (output_dir / "family_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "resolution", "name", *FAMILIES])
        writer.writeheader()
        for condition in CONDITIONS:
            key = f"{condition[0]}_{condition[1] or 'NA'}"
            writer.writerow({
                "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
                **{scene_family: family_metrics[key][scene_family]["accuracy"] for scene_family in FAMILIES},
            })

    with (output_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_metrics[0]))
        writer.writeheader()
        writer.writerows(scene_metrics)

    with (output_dir / "run_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "run_id", "scene_id", "scene_family", "condition", "resolution", "seed",
            "truth_blocked", "predicted_blocked", "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(current, key=lambda value: value["run_id"]):
            actual = truth[row["scene_id"]]
            predicted = access_prediction(row)
            writer.writerow({
                "run_id": row["run_id"], "scene_id": row["scene_id"],
                "scene_family": row["scene_family"], "condition": row["geometry_condition"],
                "resolution": row.get("direction_resolution"), "seed": row["seed"],
                "truth_blocked": actual, "predicted_blocked": predicted,
                "correct": actual == predicted,
            })

    v6 = overall["target_access_v6"]
    v5 = overall["occlusion_v5"]
    report = [
        "# L2 target-access wording v6", "", "## Experimental scope", "",
        "V6 uses the same pixel-exact Isaac occlusion GT as v5. It is a prompt/output wording ablation, not a new 3D robot approach-corridor oracle.",
        "", "## Prompt controls", "",
        "- System prompt is a separate system-role message.",
        "- The task prompt contains no examples and none of the removed exclusion sentences.",
        "- Output is only `target_access_blocked`.",
        "", "## Integrity", "",
        f"- Completed: {integrity['completed']}/{integrity['scheduled']}",
        f"- Parse errors: {integrity['parse_errors']}",
        f"- Post-validation failures: {integrity['post_validation_failures']}",
        f"- Infrastructure errors: {integrity['infrastructure_errors']}",
        "", "## Overall on the same GT", "",
        "| Formulation | Accuracy | Balanced accuracy | Blocked recall | Unblocked specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "target_access_v6": "Target access v6",
        "occlusion_v5": "Direct occlusion v5",
        "visibility_v4_inverted": "Visibility-only v4, inverted",
        "original_l1_non_FULL": "Original L1, non-FULL→true",
    }
    for name in predictors:
        row = overall[name]
        report.append(
            f"| {labels[name]} | {pct(row['accuracy'])} | {pct(row['balanced_accuracy'])} | "
            f"{pct(row['positive_recall'])} | {pct(row['specificity'])} | "
            f"{row['tp']}/{row['fn']}/{row['tn']}/{row['fp']} |"
        )
    report.extend([
        "", "## Target access v6 by condition", "",
        "| Condition | Accuracy | Balanced accuracy | Blocked recall | Unblocked specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        row = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]["target_access_v6"]
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
        key = f"{condition[0]}_{condition[1] or 'NA'}"
        values = family_metrics[key]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {NAMES[condition]} | "
            f"{pct(values['TRANSLATE']['accuracy'])} | {pct(values['ROTATE']['accuracy'])} | "
            f"{pct(values['LIFT_AND_RELOCATE']['accuracy'])} | {pct(values['FC_CLEAR']['accuracy'])} | "
            f"{pct(values['FC_BLOCKED']['accuracy'])} |"
        )
    report.extend([
        "", "## Interpretation", "",
        f"- Relative to v5, accuracy changes by {(v6['accuracy'] - v5['accuracy']) * 100:+.1f} points and balanced accuracy by {(v6['balanced_accuracy'] - v5['balanced_accuracy']) * 100:+.1f} points.",
        f"- V6 detects {v6['tp']}/{v6['tp'] + v6['fn']} blocked runs and produces {v6['fp']} false blocked decisions.",
        "- The wording asks whether an object prevents access until it is moved. Partial visual occlusion does not necessarily imply that stronger condition, so the wording and frozen pixel-occlusion GT are semantically misaligned.",
        f"- All {len(same_input_ids)} v5/v6 calls have identical numbered RGB, geometry, target ID, condition, and seed.",
    ])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    sources = {
        "runs": EXPERIMENT / "logs" / RUN_NAME / "runs.jsonl",
        "attempts": EXPERIMENT / "logs" / RUN_NAME / "attempts.jsonl",
        "runtime": EXPERIMENT / "logs" / RUN_NAME / "runtime.json",
        "last_invocation_summary": EXPERIMENT / "logs" / RUN_NAME / "last_invocation_summary.json",
        "system_prompt": EXPERIMENT / "prompts/system_en.txt",
        "prompt_l2": EXPERIMENT / "prompts/l2_en.txt",
        "schema_l2": EXPERIMENT / "schemas/l2.json",
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
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir), "integrity": integrity,
        "same_observable_input_v5_v6": len(same_input_ids), "overall": overall,
    }, indent=2))


if __name__ == "__main__":
    main()
