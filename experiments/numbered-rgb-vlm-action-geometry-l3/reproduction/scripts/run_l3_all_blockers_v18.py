#!/usr/bin/env python3
"""Run L3 with an exhaustive visible blocker-ID list and exact-set scoring."""

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
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l3_all_blockers_v18"
VERSION = "l3_all_blockers_v18"
CONDITIONS = [("C0", None), ("C1", None)] + [
    (f"C{i}", resolution) for i in range(2, 7) for resolution in ("D4", "D8")
]
FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]

TASK_PROMPT = """Identify why the target is not directly graspable and list every visible
causal blocker.

OCCLUSION means the target is not FULL.
CLEARANCE_OVERLAP means another object is within 5 cm of the target's outer
boundary. BOTH means both conditions hold. NONE means the target is directly
graspable.

Return blocker_ids as the ascending list of every visible numbered object that
causally contributes to the blocking reason. For BOTH, include the union of all
visible occlusion and clearance blockers. Do not stop after the first blocker.
Do not include the target itself or unrelated nearby objects. Use [] when the
reason is NONE."""

L3_SCHEMA = {
    "additionalProperties": False,
    "properties": {
        "blocker_ids": {
            "items": {"type": "integer"},
            "type": "array",
        },
        "blocking_reason": {
            "enum": ["OCCLUSION", "CLEARANCE_OVERLAP", "BOTH", "NONE"],
            "type": "string",
        },
    },
    "required": ["blocking_reason", "blocker_ids"],
    "type": "object",
}


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
            "L3 exhaustive visible-blocker listing: exact blocking reason and exact set "
            "of all visible causal blocker IDs."
        ),
        "prompt_change": (
            "blocker_id scalar replaced by blocker_ids array; explicitly enumerate every "
            "visible causal blocker and do not stop after the first"
        ),
        "gt_policy": (
            "all_blocker_ids is the visible-ID intersection of the union of physical "
            "occlusion and clearance blocker annotations"
        ),
        "comparison_note": (
            "The current 25-scene set has 4 empty and 21 singleton visible blocker sets; "
            "it contains no scene with two or more visible GT blockers."
        ),
    })

    for relative in (
        "prompts/system_en.txt", "prompts/l1_en.txt", "prompts/l2_en.txt", "prompts/l4_en.txt",
        "schemas/l1.json", "schemas/l2.json", "schemas/l4_d4.json", "schemas/l4_d8.json",
    ):
        copy_file(SOURCE / relative, DEST / relative)
    (DEST / "prompts").mkdir(parents=True, exist_ok=True)
    (DEST / "schemas").mkdir(parents=True, exist_ok=True)
    (DEST / "prompts/l3_en.txt").write_text(TASK_PROMPT + "\n", encoding="utf-8")
    write(DEST / "schemas/l3.json", L3_SCHEMA)

    source_scenes = read(SOURCE / "config/frozen_scenes.json")
    scenes = []
    cardinalities = Counter()
    cardinality_rows = []
    by_scene = {}
    for original in source_scenes:
        scene = copy.deepcopy(original)
        sid = scene["scene_id"]
        for relative in (scene["numbered_rgb"], scene["id_mapping"], scene["scene_geometry"]):
            copy_file(SOURCE / relative, DEST / relative)
        for geometry in scene["geometry_files"].values():
            copy_file(SOURCE / geometry["path"], DEST / geometry["path"])

        mapping = read(SOURCE / scene["id_mapping"])
        visible = {
            int(instance["display_id"])
            for instance in mapping["instances"]
            if instance["visible"]
        }
        gt = read(SOURCE / scene["ground_truth"])
        l3 = gt["L3"]
        all_blockers = sorted(
            (
                set(l3["occlusion_blocker_ids"])
                | set(l3["clearance_blocker_ids"])
            )
            & visible
        )
        assert all_blockers == l3["valid_blocker_ids"]
        assert (l3["blocking_reason"] == "NONE") == (not all_blockers)
        l3["all_blocker_ids"] = all_blockers
        gt.setdefault("methods", {})["l3_all_visible_blockers_policy"] = (
            "sorted union of occlusion_blocker_ids and clearance_blocker_ids, "
            "intersected with visible display IDs"
        )
        write(DEST / scene["ground_truth"], gt)
        scene["ground_truth_sha256"] = common.sha256_path(DEST / scene["ground_truth"])
        cardinalities[len(all_blockers)] += 1
        cardinality_rows.append({
            "scene_id": sid,
            "blocking_reason": l3["blocking_reason"],
            "all_blocker_ids": all_blockers,
            "visible_blocker_count": len(all_blockers),
        })
        by_scene[sid] = scene
        scenes.append(scene)

    assert cardinalities == Counter({1: 21, 0: 4})
    write(DEST / "config/frozen_scenes.json", scenes)
    write(DEST / "audit/gt_blocker_cardinality.json", {
        "distribution": {str(key): value for key, value in sorted(cardinalities.items())},
        "multi_visible_blocker_scene_count": sum(
            count for size, count in cardinalities.items() if size >= 2
        ),
        "scenes": cardinality_rows,
    })

    schema_hash = common.sha256_path(DEST / "schemas/l3.json")
    schedule = []
    for original in rows(SOURCE / "config/run_table.jsonl"):
        row = copy.deepcopy(original)
        scene = by_scene[row["scene_id"]]
        row["single_blocker_reference_run_id"] = original["run_id"]
        row["run_id"] = "v18__" + original["run_id"].removeprefix("v17__")
        row["experiment_version"] = VERSION
        row["ground_truth_sha256"] = scene["ground_truth_sha256"]
        row["schema"] = "schemas/l3.json"
        row["schema_sha256"] = schema_hash
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
        "source_inputs_unchanged_from_v17": True,
        "only_task_prompt_schema_and_gt_evaluation_changed": True,
        "visible_blocker_cardinality_distribution": {
            str(key): value for key, value in sorted(cardinalities.items())
        },
        "multi_visible_blocker_scene_count": 0,
        "failures": [],
    })
    return verify_frozen()


def expected_scores(parsed, gt):
    if not isinstance(parsed, dict):
        return {
            "reason_correct": False,
            "set_exact": False,
            "presence_correct": False,
            "joint_correct": False,
            "tp": 0,
            "predicted_count": 0,
            "expected_count": len(gt["all_blocker_ids"]),
        }
    blockers = parsed.get("blocker_ids")
    actual = {
        item for item in blockers if type(item) is int
    } if isinstance(blockers, list) else set()
    expected = set(gt["all_blocker_ids"])
    reason = parsed.get("blocking_reason") == gt["blocking_reason"]
    exact = isinstance(blockers, list) and actual == expected
    return {
        "reason_correct": reason,
        "set_exact": exact,
        "presence_correct": bool(actual) == bool(expected),
        "joint_correct": reason and exact,
        "tp": len(actual & expected),
        "predicted_count": len(actual),
        "expected_count": len(expected),
    }


def metrics(records):
    tp = sum(row["tp"] for row in records)
    predicted = sum(row["predicted_count"] for row in records)
    expected = sum(row["expected_count"] for row in records)
    precision = tp / predicted if predicted else float(expected == 0)
    recall = tp / expected if expected else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(records),
        "reason_correct": sum(row["reason_correct"] for row in records),
        "blocker_set_exact": sum(row["set_exact"] for row in records),
        "blocker_presence_correct": sum(row["presence_correct"] for row in records),
        "joint_correct": sum(row["joint_correct"] for row in records),
        "micro_true_positive": tp,
        "micro_predicted": predicted,
        "micro_expected": expected,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def error_type(record):
    expected = set(record["ground_truth"]["all_blocker_ids"])
    response = record.get("parsed_response") or {}
    blockers = response.get("blocker_ids")
    actual = {
        item for item in blockers if type(item) is int
    } if isinstance(blockers, list) else set()
    if actual == expected:
        return "set_exact"
    if not expected and actual:
        return "false_positive_ids"
    if expected and not actual:
        return "missed_all_ids"
    if expected and actual and not (expected & actual):
        return "wrong_replacement_ids"
    return "partial_or_extra_ids"


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
        assert row["scoring"]["blocker_set_exact_match"] == scores["set_exact"]
        assert row["scoring"]["task_correct"] == scores["joint_correct"]
        scores["joint_correct"] = scores["joint_correct"] and row["post_validation_pass"]
        record = {**row, **scores, "ground_truth": gts[row["scene_id"]]}
        record["error_type"] = error_type(record)
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
            "error_types": dict(Counter(row["error_type"] for row in group)),
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
                        "blocker_set_exact": row["set_exact"],
                        "joint_correct": row["joint_correct"],
                        "error_type": row["error_type"],
                    }
                    for row in group
                ],
            })

    overall = metrics(enriched)
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
    summary = {
        "experiment_version": VERSION,
        "model_id": config["model_id"],
        "task_definition": config["task_definition"],
        "dataset_limitation": config["comparison_note"],
        "audit": audit,
        "overall": overall,
        "error_types": dict(Counter(row["error_type"] for row in enriched)),
        "by_family": {
            family: metrics([row for row in enriched if row["scene_family"] == family])
            for family in FAMILIES
        },
        "condition_results": condition_results,
    }
    write(DEST / "results/summary.json", summary)
    write(DEST / "results/scene_results.json", scene_results)
    write(DEST / "audit/completion_report.json", audit)

    fields = [
        "condition_key", "n", "reason_correct", "blocker_set_exact",
        "blocker_presence_correct", "joint_correct", "micro_precision",
        "micro_recall", "micro_f1", "valid",
    ]
    (DEST / "results").mkdir(parents=True, exist_ok=True)
    with (DEST / "results/condition_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(condition_results)

    lines = [
        "# L3 all-visible-blockers rerun v18",
        "",
        "25 scenes × 12 conditions × 5 seeds = 1,500 independent calls. The model must return blocker_ids as the exhaustive ascending set of visible causal blockers; scoring uses exact set match.",
        "",
        "Dataset limitation: 4 scenes have zero visible blockers and 21 have exactly one; no scene has two or more visible GT blockers. This run therefore measures exhaustive-output prompting and exact-set formatting on empty/singleton sets, not true multi-blocker recall.",
        "",
        f"Completion: {len(raw)}/1500; parse errors: {audit['parse_errors']}; semantic validation failures: {audit['post_validation_failures']}; infrastructure errors: {audit['infrastructure_errors']}",
        "",
        f"Overall reason: {overall['reason_correct']}/1500 ({overall['reason_correct']/1500:.1%}); blocker set exact: {overall['blocker_set_exact']}/1500 ({overall['blocker_set_exact']/1500:.1%}); joint: {overall['joint_correct']}/1500 ({overall['joint_correct']/1500:.1%}); micro F1: {overall['micro_f1']:.1%}.",
        "",
        "| Condition | Reason /125 | Set exact /125 | Joint /125 | Set accuracy | Micro P/R/F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in condition_results:
        lines.append(
            f"| {result['condition_key']} | {result['reason_correct']} | "
            f"{result['blocker_set_exact']} | {result['joint_correct']} | "
            f"{result['blocker_set_exact']/125:.1%} | "
            f"{result['micro_precision']:.1%} / {result['micro_recall']:.1%} / {result['micro_f1']:.1%} |"
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
