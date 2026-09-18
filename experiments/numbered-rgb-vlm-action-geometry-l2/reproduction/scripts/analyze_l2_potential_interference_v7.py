#!/usr/bin/env python3
"""Analyze potential-grasp-interference v7 with the relaxed occlusion GT."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analyze_l2_grasp_zone_v3 import CONDITIONS, FAMILIES, NAMES, pct, read_json, read_jsonl
from analyze_l2_occlusion_boolean_v5 import metric


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_potential_interference_v7"
V6_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_target_access_v6"
V5_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_occlusion_boolean_v5"
V1_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_v1"
RUN_NAME = "l2_potential_interference_v7"


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def v7_prediction(row: dict) -> bool:
    return row["parsed_response"]["potential_grasp_interference"]


def v6_prediction(row: dict) -> bool:
    return row["parsed_response"]["target_access_blocked"]


def v5_prediction(row: dict) -> bool:
    return row["parsed_response"]["occluded_by_other_object"]


def l1_prediction(row: dict) -> bool:
    return row["parsed_response"]["visibility"] != "FULL"


def load_level(path: Path, level: str) -> list[dict]:
    return [row for row in read_jsonl(path) if row["level"] == level]


def main() -> None:
    output_dir = EXPERIMENT / "results" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    current = load_level(EXPERIMENT / "logs" / RUN_NAME / "runs.jsonl", "L2")
    v6_rows = load_level(V6_EXPERIMENT / "logs/l2_target_access_v6/runs.jsonl", "L2")
    v5_rows = load_level(V5_EXPERIMENT / "logs/l2_occlusion_boolean_v5/runs.jsonl", "L2")
    l1_rows = load_level(V1_EXPERIMENT / "logs/runs.jsonl", "L1")
    datasets = {"v7": current, "v6": v6_rows, "v5": v5_rows, "l1": l1_rows}
    for name, rows in datasets.items():
        if len(rows) != 1500 or len({row["run_id"] for row in rows}) != 1500:
            raise RuntimeError(f"Expected 1500 unique {name} rows, got {len(rows)}")

    truth: dict[str, bool] = {}
    ratios: dict[str, float] = {}
    family: dict[str, str] = {}
    for row in current:
        scene = row["scene_id"]
        gt = read_json(EXPERIMENT / row["ground_truth"])
        truth[scene] = bool(gt["L2"]["potential_grasp_interference"])
        ratios[scene] = float(gt["L2"]["occlusion_ratio"])
        family[scene] = row["scene_family"]
        expected = ratios[scene] >= float(gt["L2"]["meaningful_occlusion_threshold"])
        if truth[scene] != expected:
            raise RuntimeError(f"Relaxed GT mismatch in {scene}")

    versions = {
        "potential_interference_v7": (current, v7_prediction, "v7"),
        "direct_occlusion_v5_rescored": (v5_rows, v5_prediction, "v5"),
        "target_access_v6_rescored": (v6_rows, v6_prediction, "v6"),
        "original_l1_non_FULL_rescored": (l1_rows, l1_prediction, "l1"),
    }
    grouped = {name: defaultdict(list) for name in datasets}
    family_grouped = defaultdict(list)
    for dataset_name, rows in datasets.items():
        for row in rows:
            grouped[dataset_name][condition_key(row)].append(row)
            if dataset_name == "v7":
                family_grouped[(condition_key(row), row["scene_family"])].append(row)

    overall = {
        name: metric(rows, truth, predictor)
        for name, (rows, predictor, _) in versions.items()
    }
    condition_metrics = {}
    family_metrics = {}
    for condition in CONDITIONS:
        key = f"{condition[0]}_{condition[1] or 'NA'}"
        condition_metrics[key] = {
            "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
            **{
                name: metric(grouped[group_name][condition], truth, predictor)
                for name, (_, predictor, group_name) in versions.items()
            },
        }
        family_metrics[key] = {
            scene_family: metric(
                family_grouped[(condition, scene_family)], truth, v7_prediction
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
        predicted = sum(v7_prediction(row) for row in rows)
        correct = predicted if truth[scene] else len(rows) - predicted
        scene_metrics.append({
            "scene_id": scene, "scene_family": family[scene],
            "occlusion_ratio": ratios[scene], "truth_interference": truth[scene],
            "n": len(rows), "predicted_interference": predicted,
            "correct": correct, "accuracy": correct / len(rows),
        })

    family_overall = {}
    for scene_family in FAMILIES:
        rows = [row for row in current if row["scene_family"] == scene_family]
        family_overall[scene_family] = metric(rows, truth, v7_prediction)

    prior_by_id = {
        name: {row["run_id"]: row for row in rows}
        for name, rows in {"v5": v5_rows, "v6": v6_rows}.items()
    }
    same_input = {}
    for name, by_id in prior_by_id.items():
        same_input[name] = sum(
            row["numbered_rgb_sha256"] == by_id[row["run_id"]]["numbered_rgb_sha256"]
            and row["geometry_json_sha256"] == by_id[row["run_id"]]["geometry_json_sha256"]
            and row["target_object_id"] == by_id[row["run_id"]]["target_object_id"]
            and row["seed"] == by_id[row["run_id"]]["seed"]
            for row in current
        )

    results = {
        "prompt_version": read_json(EXPERIMENT / "config/experiment_config.json")["l2_prompt_version"],
        "gt": {
            "rule": "occlusion_ratio >= 0.05",
            "positive_scenes": sorted(scene for scene, value in truth.items() if value),
            "negative_scenes": sorted(scene for scene, value in truth.items() if not value),
            "positive_runs": sum(truth[row["scene_id"]] for row in current),
            "negative_runs": sum(not truth[row["scene_id"]] for row in current),
        },
        "integrity": integrity,
        "overall_same_relaxed_gt": overall,
        "condition_comparison": condition_metrics,
        "family_metrics": family_metrics,
        "family_overall": family_overall,
        "scene_metrics": scene_metrics,
        "same_observable_input": same_input,
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
            for version in versions:
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
            "occlusion_ratio", "truth_interference", "predicted_interference", "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(current, key=lambda value: value["run_id"]):
            actual = truth[row["scene_id"]]
            predicted = v7_prediction(row)
            writer.writerow({
                "run_id": row["run_id"], "scene_id": row["scene_id"],
                "scene_family": row["scene_family"], "condition": row["geometry_condition"],
                "resolution": row.get("direction_resolution"), "seed": row["seed"],
                "occlusion_ratio": ratios[row["scene_id"]], "truth_interference": actual,
                "predicted_interference": predicted, "correct": actual == predicted,
            })

    labels = {
        "potential_interference_v7": "Potential grasp interference v7",
        "direct_occlusion_v5_rescored": "Direct occlusion v5, rescored",
        "target_access_v6_rescored": "Target access v6, rescored",
        "original_l1_non_FULL_rescored": "Original L1 non-FULL, rescored",
    }
    report = [
        "# L2 potential-grasp-interference v7", "", "## Experimental definition", "",
        "The task prompt contains no numeric threshold. Ground truth treats `occlusion_ratio >= 0.05` as meaningful interference and ignores smaller render-level overlaps.",
        "", "## Integrity", "",
        f"- Completed: {integrity['completed']}/{integrity['scheduled']}",
        f"- Parse errors: {integrity['parse_errors']}",
        f"- Post-validation failures: {integrity['post_validation_failures']}",
        f"- Infrastructure errors: {integrity['infrastructure_errors']}",
        "", "## Overall on the same relaxed GT", "",
        "| Formulation | Accuracy | Balanced accuracy | Interference recall | No-interference specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in versions:
        row = overall[name]
        report.append(
            f"| {labels[name]} | {pct(row['accuracy'])} | {pct(row['balanced_accuracy'])} | "
            f"{pct(row['positive_recall'])} | {pct(row['specificity'])} | "
            f"{row['tp']}/{row['fn']}/{row['tn']}/{row['fp']} |"
        )
    report.extend([
        "", "## V7 by condition", "",
        "| Condition | Accuracy | Balanced accuracy | Interference recall | No-interference specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        row = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]["potential_interference_v7"]
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
    v7 = overall["potential_interference_v7"]
    v5 = overall["direct_occlusion_v5_rescored"]
    v6 = overall["target_access_v6_rescored"]
    report.extend([
        "", "## Interpretation", "",
        f"- V7 improves balanced accuracy by {(v7['balanced_accuracy'] - v6['balanced_accuracy']) * 100:+.1f} points over v6, but remains {(v7['balanced_accuracy'] - v5['balanced_accuracy']) * 100:+.1f} points below v5 on the relaxed GT.",
        f"- V7 detects {v7['tp']}/{v7['tp'] + v7['fn']} interference runs and makes {v7['fp']} false-interference decisions.",
        "- Relaxing the GT correctly removes the 0.417% edge overlap, but the grasp-interference wording still asks for stronger physical evidence than direct visual occlusion and therefore lowers recall.",
        f"- V7 has identical observable inputs to v5 in {same_input['v5']}/1500 calls and to v6 in {same_input['v6']}/1500 calls.",
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
        "same_observable_input": same_input, "overall": overall,
    }, indent=2))


if __name__ == "__main__":
    main()
