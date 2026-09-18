#!/usr/bin/env python3
"""Analyze L2 finger-placement-zone results against the formal-v2 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    ("C0", None), ("C1", None),
    ("C2", "D4"), ("C2", "D8"),
    ("C3", "D4"), ("C3", "D8"),
    ("C4", "D4"), ("C4", "D8"),
    ("C5", "D4"), ("C5", "D8"),
    ("C6", "D4"), ("C6", "D8"),
]
NAMES = {
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


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def metrics(rows: list[dict], truth: dict[str, bool]) -> dict:
    counts = Counter()
    for row in rows:
        actual = truth[row["scene_id"]]
        predicted = bool(row["parsed_response"]["direct_graspable"])
        counts[(actual, predicted)] += 1
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


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=Path,
        default=ROOT / "experiments/vlm_action_geometry_single_info_l2_grasp_zone_v3",
    )
    parser.add_argument(
        "--baseline-experiment", type=Path,
        default=ROOT / "experiments/vlm_action_geometry_single_info_maskfix_validation",
    )
    parser.add_argument("--run-name", default="l2_grasp_zone_v3")
    parser.add_argument("--baseline-runs", default="logs/l2_formal_v2/runs.jsonl")
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    baseline = args.baseline_experiment.resolve()
    output_dir = experiment / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    current = [row for row in read_jsonl(experiment / "logs" / args.run_name / "runs.jsonl") if row["level"] == "L2"]
    previous = [row for row in read_jsonl(baseline / args.baseline_runs) if row["level"] == "L2"]
    current_by_id = {row["run_id"]: row for row in current}
    previous_by_id = {row["run_id"]: row for row in previous}
    if len(current) != 1500 or len(current_by_id) != 1500:
        raise RuntimeError(f"Expected 1500 unique current rows, got {len(current)}/{len(current_by_id)}")
    if set(current_by_id) != set(previous_by_id):
        raise RuntimeError("Current and baseline run IDs differ")

    truth = {}
    previous_truth = {}
    scene_family = {}
    for row in current:
        scene = row["scene_id"]
        truth[scene] = bool(read_json(experiment / row["ground_truth"])["L2"]["direct_graspable"])
        previous_truth[scene] = bool(read_json(baseline / previous_by_id[row["run_id"]]["ground_truth"])["L2"]["direct_graspable"])
        scene_family[scene] = row["scene_family"]

    grouped = defaultdict(list)
    previous_grouped = defaultdict(list)
    family_grouped = defaultdict(list)
    for row in current:
        grouped[key(row)].append(row)
        family_grouped[(key(row), row["scene_family"])].append(row)
    for row in previous:
        previous_grouped[key(row)].append(row)

    condition_metrics = {}
    baseline_condition_metrics = {}
    family_metrics = {}
    for condition in CONDITIONS:
        name = f"{condition[0]}_{condition[1] or 'NA'}"
        condition_metrics[name] = {
            "condition": condition[0], "resolution": condition[1],
            "name": NAMES[condition], **metrics(grouped[condition], truth),
        }
        baseline_condition_metrics[name] = {
            "condition": condition[0], "resolution": condition[1],
            "name": NAMES[condition], **metrics(previous_grouped[condition], truth),
        }
        family_metrics[name] = {
            family: metrics(family_grouped[(condition, family)], truth)
            for family in FAMILIES
        }

    identical_ids = [
        run_id for run_id, row in current_by_id.items()
        if row["numbered_rgb_sha256"] == previous_by_id[run_id]["numbered_rgb_sha256"]
        and row["geometry_json_sha256"] == previous_by_id[run_id]["geometry_json_sha256"]
        and row["schema_sha256"] == previous_by_id[run_id]["schema_sha256"]
        and row["target_object_id"] == previous_by_id[run_id]["target_object_id"]
    ]
    transitions = Counter()
    paired_correctness = Counter()
    for run_id, row in current_by_id.items():
        old_prediction = bool(previous_by_id[run_id]["parsed_response"]["direct_graspable"])
        new_prediction = bool(row["parsed_response"]["direct_graspable"])
        actual = truth[row["scene_id"]]
        transitions[(old_prediction, new_prediction)] += 1
        paired_correctness[(old_prediction == actual, new_prediction == actual)] += 1

    attempts = read_jsonl(experiment / "logs" / args.run_name / "attempts.jsonl")
    integrity = {
        "scheduled": 1500,
        "completed": len(current),
        "unique_run_ids": len(current_by_id),
        "parse_errors": sum(row["parse_error"] is not None for row in current),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in current),
        "infrastructure_errors": sum(row["status"] != "success" for row in attempts),
        "attempt_records": len(attempts),
    }
    changed_labels = [
        {
            "scene_id": scene,
            "family": scene_family[scene],
            "formal_v2": previous_truth[scene],
            "grasp_zone_v3": truth[scene],
        }
        for scene in sorted(truth) if truth[scene] != previous_truth[scene]
    ]
    results = {
        "prompt_version": read_json(experiment / "config/experiment_config.json")["l2_prompt_version"],
        "integrity": integrity,
        "gt": {
            "positive_scenes": sorted(scene for scene, value in truth.items() if value),
            "negative_scenes": sorted(scene for scene, value in truth.items() if not value),
            "changed_from_formal_v2": changed_labels,
        },
        "overall_rescored_to_v3_gt": {
            "formal_v2_responses": metrics(previous, truth),
            "grasp_zone_v3_responses": metrics(current, truth),
        },
        "condition_metrics": condition_metrics,
        "formal_v2_condition_metrics_rescored_to_v3_gt": baseline_condition_metrics,
        "family_metrics": family_metrics,
        "same_input_prompt_only_subset": {
            "n": len(identical_ids),
            "scene_ids": sorted({current_by_id[item]["scene_id"] for item in identical_ids}),
            "formal_v2_responses": metrics([previous_by_id[item] for item in identical_ids], truth),
            "grasp_zone_v3_responses": metrics([current_by_id[item] for item in identical_ids], truth),
        },
        "prediction_transitions": {
            "false_to_false": transitions[(False, False)],
            "false_to_true": transitions[(False, True)],
            "true_to_false": transitions[(True, False)],
            "true_to_true": transitions[(True, True)],
        },
        "paired_correctness_rescored_to_v3_gt": {
            "old_correct_new_correct": paired_correctness[(True, True)],
            "old_correct_new_wrong": paired_correctness[(True, False)],
            "old_wrong_new_correct": paired_correctness[(False, True)],
            "old_wrong_new_wrong": paired_correctness[(False, False)],
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    columns = [
        "condition", "resolution", "name", "n", "tp", "fn", "tn", "fp",
        "predicted_true", "accuracy", "balanced_accuracy", "positive_recall",
        "specificity", "precision",
    ]
    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for condition in CONDITIONS:
            writer.writerow(condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"])

    with (output_dir / "family_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "resolution", "name", *FAMILIES])
        writer.writeheader()
        for condition in CONDITIONS:
            name = f"{condition[0]}_{condition[1] or 'NA'}"
            writer.writerow({
                "condition": condition[0], "resolution": condition[1], "name": NAMES[condition],
                **{family: family_metrics[name][family]["accuracy"] for family in FAMILIES},
            })

    with (output_dir / "run_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["run_id", "scene_id", "scene_family", "condition", "resolution", "seed", "truth", "prediction", "correct"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(current, key=lambda item: item["run_id"]):
            prediction = bool(row["parsed_response"]["direct_graspable"])
            actual = truth[row["scene_id"]]
            writer.writerow({
                "run_id": row["run_id"], "scene_id": row["scene_id"],
                "scene_family": row["scene_family"], "condition": row["geometry_condition"],
                "resolution": row.get("direction_resolution"), "seed": row["seed"],
                "truth": actual, "prediction": prediction, "correct": actual == prediction,
            })

    scene_rows = []
    for scene in sorted(truth):
        rows = [row for row in current if row["scene_id"] == scene]
        predicted_true = sum(bool(row["parsed_response"]["direct_graspable"]) for row in rows)
        scene_rows.append({
            "scene_id": scene,
            "scene_family": scene_family[scene],
            "truth": truth[scene],
            "n": len(rows),
            "predicted_true": predicted_true,
            "correct": predicted_true if truth[scene] else len(rows) - predicted_true,
            "accuracy": (predicted_true if truth[scene] else len(rows) - predicted_true) / len(rows),
        })
    with (output_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)

    old_overall = results["overall_rescored_to_v3_gt"]["formal_v2_responses"]
    new_overall = results["overall_rescored_to_v3_gt"]["grasp_zone_v3_responses"]
    report = [
        "# L2 finger-placement-zone v3 rerun", "", "## Integrity", "",
        f"- Completed: {integrity['completed']}/{integrity['scheduled']}",
        f"- Parse errors: {integrity['parse_errors']}",
        f"- Post-validation failures: {integrity['post_validation_failures']}",
        f"- Infrastructure errors: {integrity['infrastructure_errors']}", "",
        "## GT change", "",
        f"- Positive scenes: {sum(truth.values())}/25",
        f"- Labels changed from formal v2: {len(changed_labels)}",
    ]
    report.extend(f"  - {row['scene_id']}: {row['formal_v2']} -> {row['grasp_zone_v3']}" for row in changed_labels)
    report.extend([
        "", "## Overall comparison (both rescored against v3 GT)", "",
        "| Responses | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Formal v2 | {pct(old_overall['accuracy'])} | {pct(old_overall['balanced_accuracy'])} | {pct(old_overall['positive_recall'])} | {pct(old_overall['specificity'])} | {old_overall['tp']}/{old_overall['fn']}/{old_overall['tn']}/{old_overall['fp']} |",
        f"| Grasp-zone v3 | {pct(new_overall['accuracy'])} | {pct(new_overall['balanced_accuracy'])} | {pct(new_overall['positive_recall'])} | {pct(new_overall['specificity'])} | {new_overall['tp']}/{new_overall['fn']}/{new_overall['tn']}/{new_overall['fp']} |",
        "", "## Grasp-zone v3 condition metrics", "",
        "| Condition | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP | Predicted true |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        row = condition_metrics[f"{condition[0]}_{condition[1] or 'NA'}"]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {row['name']} | {pct(row['accuracy'])} | "
            f"{pct(row['balanced_accuracy'])} | {pct(row['positive_recall'])} | "
            f"{pct(row['specificity'])} | {row['tp']}/{row['fn']}/{row['tn']}/{row['fp']} | "
            f"{row['predicted_true']} |"
        )
    report.extend([
        "", "## Per-condition change from formal v2", "",
        "Both versions below are scored against the same v3 GT.", "",
        "| Condition | Balanced accuracy v2→v3 | True recall v2→v3 | Specificity v2→v3 |",
        "|---|---:|---:|---:|",
    ])
    for condition in CONDITIONS:
        name = f"{condition[0]}_{condition[1] or 'NA'}"
        old = baseline_condition_metrics[name]
        new = condition_metrics[name]
        report.append(
            f"| {condition[0]} {condition[1] or ''} {NAMES[condition]} | "
            f"{pct(old['balanced_accuracy'])} → {pct(new['balanced_accuracy'])} | "
            f"{pct(old['positive_recall'])} → {pct(new['positive_recall'])} | "
            f"{pct(old['specificity'])} → {pct(new['specificity'])} |"
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
    same = results["same_input_prompt_only_subset"]
    report.extend([
        "", "## Byte-identical input subset", "",
        f"- Runs with identical image, geometry, schema, and target ID: {same['n']}",
        "- These isolate the prompt/decision-rule effect; C1/C2 are excluded where R geometry changed.",
        "", "## Diagnostics", "",
        "- All three false negatives are C0 responses for scene_fc_blocked_v02 (seeds 28102, 28103, and 28104).",
        "- C1 R Numeric returned true in all 20 runs on the four GT-negative FC_BLOCKED scenes, even though each payload contains a zero distance to one finger-placement zone.",
        "- C2 also returned true in 17/20 D4 runs and 18/20 D8 runs on those four scenes despite an authoritative OCCUPIED side status.",
        "- scene_fc_blocked_v02 is the sole manifest/Isaac design mismatch: its actual minimum mesh gap is 2.352752 cm, slightly larger than the required 2.2 cm side clearance, so the Isaac geometry oracle labels it true.",
        "", "## Interpretation", "",
        "- Replacing the universal 5 cm rule removed the near-always-false behavior: true recall rose from 25.8% to 99.2% when both versions are scored on v3 GT.",
        "- The change overcorrected toward true. Specificity fell from 95.3% to 41.5%, so raw accuracy fell even though balanced accuracy rose from 60.5% to 70.3%.",
        "- C0 Numbered RGB only is the strongest v3 condition by balanced accuracy (84.5%). C1 R Numeric is second (77.4%).",
        "- The main remaining error is failure to preserve the independent FULL-visibility test and, for R inputs, failure to obey explicit occupied-zone evidence. The revised clearance concept is physically better grounded, but this prompt does not yet make the two-test conjunction reliable.",
    ])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    source_artifacts = {
        "runs": experiment / "logs" / args.run_name / "runs.jsonl",
        "attempts": experiment / "logs" / args.run_name / "attempts.jsonl",
        "runtime": experiment / "logs" / args.run_name / "runtime.json",
        "last_invocation_summary": experiment / "logs" / args.run_name / "last_invocation_summary.json",
        "prompt_l2": experiment / "prompts/l2_en.txt",
        "experiment_config": experiment / "config/experiment_config.json",
        "run_table": experiment / "config/run_table.jsonl",
        "preflight": experiment / "audit/preflight_report.json",
        "object_coverage": experiment / "audit/object_coverage_report.json",
    }
    (output_dir / "source_artifact_hashes.json").write_text(
        json.dumps(
            {
                name: {"path": str(path.relative_to(experiment)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in source_artifacts.items()
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    hashes = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "integrity": integrity, "overall": results["overall_rescored_to_v3_gt"]}, indent=2))


if __name__ == "__main__":
    main()
