#!/usr/bin/env python3
"""Run L3 with a single strongest/first-removal causal blocker prompt."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l3_maskfix_v17"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
VERSION = "l3_primary_blocker_v19"
CONDITIONS = [("C0", None), ("C1", None)] + [
    (f"C{i}", resolution) for i in range(2, 7) for resolution in ("D4", "D8")
]
FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]

TASK_PROMPT = """Identify why the target is not directly graspable and select exactly one
visible causal blocker for the next action.

OCCLUSION means the target is not FULL.
CLEARANCE_OVERLAP means another object is within 5 cm of the target's outer
boundary. BOTH means both conditions hold. NONE means the target is directly
graspable.

Choose the single object that most strongly prevents direct graspability and
should be removed first. If several objects contribute, select the one whose
removal would most improve direct graspability. Do not list alternatives. Use
null when the reason is NONE."""


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write(path, value):
    common.write_json(path, value)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def condition(row):
    return row["geometry_condition"] + (
        "_" + row["direction_resolution"] if row["direction_resolution"] else ""
    )


def verify_frozen():
    config = read(DEST / "config/experiment_config.json")
    audit = read(DEST / "audit/preflight_report.json")
    assert config["experiment_version"] == VERSION and audit["status"] == "pass"
    assert common.sha256_path(DEST / "config/run_table.jsonl") == audit["run_table_sha256"]
    for relative, digest in read(DEST / "audit/frozen_file_hashes.json").items():
        assert common.sha256_path(DEST / relative) == digest, f"Frozen file changed: {relative}"
    schedule = rows(DEST / "config/run_table.jsonl")
    assert len(schedule) == len({row["run_id"] for row in schedule}) == 1500
    return config, schedule


def prepare():
    if (DEST / "audit/preflight_report.json").exists():
        return verify_frozen()
    if (DEST / "logs/runs.jsonl").exists():
        raise RuntimeError("Existing inference without frozen preflight; refusing overwrite")

    source_config = read(SOURCE / "config/experiment_config.json")
    config = copy.deepcopy(source_config)
    config.update({
        "experiment_version": VERSION,
        "scheduled_calls": 1500,
        "levels": ["L3"],
        "reference_experiment": str(SOURCE),
        "task_definition": (
            "Exact blocking reason and exactly one visible causal blocker selected as the "
            "strongest obstruction and first object to remove."
        ),
        "prompt_change": (
            "Arbitrary valid blocker choice replaced by strongest-blocker/first-removal "
            "selection; scalar blocker_id schema retained."
        ),
        "comparison_note": (
            "The current 25-scene set has no scene with multiple visible valid blocker "
            "IDs, so blocker ranking quality cannot be directly evaluated."
        ),
    })

    for relative in (
        "prompts/system_en.txt", "prompts/l1_en.txt", "prompts/l2_en.txt", "prompts/l4_en.txt",
        "schemas/l1.json", "schemas/l2.json", "schemas/l3.json",
        "schemas/l4_d4.json", "schemas/l4_d8.json",
    ):
        copy_file(SOURCE / relative, DEST / relative)
    (DEST / "prompts").mkdir(parents=True, exist_ok=True)
    (DEST / "prompts/l3_en.txt").write_text(TASK_PROMPT + "\n", encoding="utf-8")

    source_scenes = read(SOURCE / "config/frozen_scenes.json")
    scenes = []
    cardinalities = Counter()
    for original in source_scenes:
        scene = copy.deepcopy(original)
        for relative in (
            scene["numbered_rgb"], scene["id_mapping"], scene["scene_geometry"],
            scene["ground_truth"],
        ):
            copy_file(SOURCE / relative, DEST / relative)
        for geometry in scene["geometry_files"].values():
            copy_file(SOURCE / geometry["path"], DEST / geometry["path"])
        gt = read(DEST / scene["ground_truth"])["L3"]
        cardinalities[len(gt["valid_blocker_ids"])] += 1
        scenes.append(scene)
    assert cardinalities == Counter({1: 21, 0: 4})
    write(DEST / "config/frozen_scenes.json", scenes)
    write(DEST / "audit/gt_blocker_cardinality.json", {
        "distribution": {str(key): value for key, value in sorted(cardinalities.items())},
        "multi_visible_valid_blocker_scene_count": 0,
        "ranking_evaluation_supported": False,
    })

    schedule = []
    for original in rows(SOURCE / "config/run_table.jsonl"):
        row = copy.deepcopy(original)
        row["single_valid_reference_run_id"] = original["run_id"]
        row["run_id"] = "v19__" + original["run_id"].removeprefix("v17__")
        row["experiment_version"] = VERSION
        schedule.append(row)
    assert len(schedule) == len({row["run_id"] for row in schedule}) == 1500
    assert set(Counter(condition(row) for row in schedule).values()) == {125}
    run_table = DEST / "config/run_table.jsonl"
    run_table.parent.mkdir(parents=True, exist_ok=True)
    with run_table.open("w", encoding="utf-8") as handle:
        for row in schedule:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    write(DEST / "config/experiment_config.json", config)
    hashes = {
        str(path.relative_to(DEST)): common.sha256_path(path)
        for folder in ("config", "prompts", "schemas", "geometry", "images")
        for path in (DEST / folder).rglob("*")
        if path.is_file()
    }
    write(DEST / "audit/frozen_file_hashes.json", hashes)
    write(DEST / "audit/preflight_report.json", {
        "status": "pass",
        "checked_at": now(),
        "scheduled_calls": 1500,
        "scene_count": 25,
        "run_table_sha256": common.sha256_path(run_table),
        "frozen_file_count": len(hashes),
        "source_inputs_schema_model_generation_unchanged_from_v17": True,
        "only_l3_task_prompt_changed": True,
        "visible_valid_blocker_cardinality_distribution": {
            str(key): value for key, value in sorted(cardinalities.items())
        },
        "multi_visible_valid_blocker_scene_count": 0,
        "failures": [],
    })
    return verify_frozen()


def expected_scores(parsed, gt):
    if not isinstance(parsed, dict):
        return {"reason_correct": False, "blocker_correct": False, "joint_correct": False}
    reason = parsed.get("blocking_reason") == gt["blocking_reason"]
    blocker = parsed.get("blocker_id")
    blocker_ok = blocker is None if gt["blocking_reason"] == "NONE" else blocker in gt["valid_blocker_ids"]
    return {
        "reason_correct": reason,
        "blocker_correct": blocker_ok,
        "joint_correct": reason and blocker_ok,
    }


def metrics(records):
    return {
        "n": len(records),
        "reason_correct": sum(row["reason_correct"] for row in records),
        "blocker_correct": sum(row["blocker_correct"] for row in records),
        "joint_correct": sum(row["joint_correct"] for row in records),
    }


def analyze():
    config, schedule = verify_frozen()
    raw = rows(DEST / "logs/runs.jsonl")
    assert len(raw) == len({row["run_id"] for row in raw}) == 1500
    planned = {row["run_id"]: row for row in schedule}
    assert {row["run_id"] for row in raw} == set(planned)
    scene_rows = read(DEST / "config/frozen_scenes.json")
    gts = {
        scene["scene_id"]: read(DEST / scene["ground_truth"])["L3"]
        for scene in scene_rows
    }
    system = (DEST / "prompts/system_en.txt").read_text(encoding="utf-8").strip()
    task = (DEST / "prompts/l3_en.txt").read_text(encoding="utf-8").strip()
    schema = read(DEST / "schemas/l3.json")
    by_condition = defaultdict(list)
    by_scene_condition = defaultdict(list)
    enriched = []
    for row in raw:
        assert all(row[key] == value for key, value in planned[row["run_id"]].items())
        assert row["generation"] == config["generation"]
        assert row["system_prompt"] == system and row["user_prompt"].endswith(task)
        assert row["json_schema"] == schema and row["level"] == "L3"
        scores = expected_scores(row.get("parsed_response"), gts[row["scene_id"]])
        assert row["scoring"]["blocking_reason_correct"] == scores["reason_correct"]
        assert row["scoring"]["blocker_valid_set_member"] == scores["blocker_correct"]
        assert row["scoring"]["task_correct"] == scores["joint_correct"]
        scores["joint_correct"] = scores["joint_correct"] and row["post_validation_pass"]
        record = {**row, **scores, "ground_truth_effective": gts[row["scene_id"]]}
        enriched.append(record)
        by_condition[condition(row)].append(record)
        by_scene_condition[(row["scene_id"], condition(row))].append(record)

    condition_results = []
    for key_base, resolution in CONDITIONS:
        key = key_base + ("_" + resolution if resolution else "")
        group = by_condition[key]
        assert len(group) == 125
        condition_results.append({
            "condition_key": key,
            **metrics(group),
            "valid": sum(row["post_validation_pass"] for row in group),
            "by_family": {
                family: metrics([row for row in group if row["scene_family"] == family])
                for family in FAMILIES
            },
        })

    scene_results = []
    for sid in sorted(gts):
        for key_base, resolution in CONDITIONS:
            key = key_base + ("_" + resolution if resolution else "")
            group = sorted(by_scene_condition[(sid, key)], key=lambda row: row["seed"])
            assert len(group) == 5
            scene_results.append({
                "scene_id": sid,
                "condition_key": key,
                "target_object_id": group[0]["target_object_id"],
                "scene_family": group[0]["scene_family"],
                "image_path": str(DEST / group[0]["numbered_rgb"]),
                "gt": gts[sid],
                **metrics(group),
                "outputs": [
                    {
                        "seed": row["seed"],
                        "response": row["parsed_response"],
                        "valid": row["post_validation_pass"],
                        "reason_correct": row["reason_correct"],
                        "blocker_correct": row["blocker_correct"],
                        "joint_correct": row["joint_correct"],
                    }
                    for row in group
                ],
            })

    overall = metrics(enriched)
    reason_confusion = Counter(
        (row["ground_truth_effective"]["blocking_reason"], row["parsed_response"]["blocking_reason"])
        for row in enriched
    )
    audit = {
        "status": "pass",
        "completed": len(raw),
        "unique_run_ids": len(planned),
        "parse_errors": sum(row["parse_error"] is not None for row in raw),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in raw),
        "infrastructure_errors": sum(
            row["status"] == "infrastructure_error"
            for row in rows(DEST / "logs/attempts.jsonl")
        ),
        "condition_counts": dict(Counter(condition(row) for row in raw)),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in raw)),
        "logs_sha256": common.sha256_path(DEST / "logs/runs.jsonl"),
        "finished_at": now(),
    }
    source_summary = read(SOURCE / "results/summary.json")
    source_overall = {
        "reason_correct": sum(row["reason_correct"] for row in read(SOURCE / "results/scene_results.json")),
        "blocker_correct": sum(row["blocker_correct"] for row in read(SOURCE / "results/scene_results.json")),
        "joint_correct": sum(row["joint_correct"] for row in read(SOURCE / "results/scene_results.json")),
    }
    summary = {
        "experiment_version": VERSION,
        "model_id": config["model_id"],
        "task_definition": config["task_definition"],
        "dataset_limitation": config["comparison_note"],
        "audit": audit,
        "overall": overall,
        "v17_reference_overall": source_overall,
        "delta_vs_v17": {key: overall[key] - source_overall[key] for key in source_overall},
        "reason_confusion": [
            {"gt": gt, "prediction": pred, "n": count}
            for (gt, pred), count in sorted(reason_confusion.items())
        ],
        "by_family": {
            family: metrics([row for row in enriched if row["scene_family"] == family])
            for family in FAMILIES
        },
        "condition_results": condition_results,
        "source_summary_experiment_version": source_summary["experiment_version"],
    }
    write(DEST / "results/summary.json", summary)
    write(DEST / "results/scene_results.json", scene_results)
    write(DEST / "audit/completion_report.json", audit)

    fields = [
        "condition_key", "n", "reason_correct", "blocker_correct", "joint_correct", "valid",
    ]
    (DEST / "results").mkdir(parents=True, exist_ok=True)
    with (DEST / "results/condition_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(condition_results)

    lines = [
        "# L3 strongest-primary-blocker rerun v19",
        "",
        "25 scenes × 12 conditions × 5 seeds = 1,500 independent calls. The scalar blocker_id must be the visible causal object whose removal would most improve direct graspability.",
        "",
        "Dataset limitation: every blocked scene has exactly one visible valid blocker ID, so ranking among multiple valid blockers is not directly evaluated.",
        "",
        f"Completion: {len(raw)}/1500; parse errors: {audit['parse_errors']}; semantic validation failures: {audit['post_validation_failures']}; infrastructure errors: {audit['infrastructure_errors']}",
        "",
        f"Overall reason: {overall['reason_correct']}/1500 ({overall['reason_correct']/1500:.1%}); blocker: {overall['blocker_correct']}/1500 ({overall['blocker_correct']/1500:.1%}); joint: {overall['joint_correct']}/1500 ({overall['joint_correct']/1500:.1%}).",
        "",
        "| Condition | Reason /125 | Blocker /125 | Joint /125 | Joint accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in condition_results:
        lines.append(
            f"| {result['condition_key']} | {result['reason_correct']} | "
            f"{result['blocker_correct']} | {result['joint_correct']} | "
            f"{result['joint_correct']/125:.1%} |"
        )
    (DEST / "results/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()
    if args.analyze_only:
        analyze()
        return
    prepare()
    print(json.dumps(read(DEST / "audit/preflight_report.json"), indent=2), flush=True)
    if args.prepare_only:
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/run_vlm_action_geometry_single_info_v1.py"),
        "--experiment", str(DEST),
        "--level", "L3",
    ]
    if args.max_new_calls is not None:
        command += ["--max-new-calls", str(args.max_new_calls)]
    subprocess.run(command, check=True)
    if args.max_new_calls is None:
        analyze()


if __name__ == "__main__":
    main()
