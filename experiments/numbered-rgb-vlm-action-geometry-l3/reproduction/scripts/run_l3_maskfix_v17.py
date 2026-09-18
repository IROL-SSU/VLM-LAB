#!/usr/bin/env python3
"""Rerun the original L3 task on audited, corrected numbered images."""

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

from PIL import Image

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments/vlm_action_geometry_single_info_v1"
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_maskfix_validation"
RECENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l3_maskfix_v17"
VERSION = "l3_maskfix_v17"
CONDITIONS = [("C0", None), ("C1", None)] + [
    (f"C{i}", resolution) for i in range(2, 7) for resolution in ("D4", "D8")
]
FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    common.write_json(path, value)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def condition(row):
    return row["geometry_condition"] + ("_" + row["direction_resolution"] if row["direction_resolution"] else "")


def pair_key(row):
    return row["scene_id"], condition(row), row["seed"]


def verify_frozen():
    config = read(DEST / "config/experiment_config.json")
    audit = read(DEST / "audit/preflight_report.json")
    assert audit["status"] == "pass" and config["experiment_version"] == VERSION
    assert common.sha256_path(DEST / "config/run_table.jsonl") == audit["run_table_sha256"]
    for path, digest in read(DEST / "audit/frozen_file_hashes.json").items():
        assert common.sha256_path(DEST / path) == digest, f"Frozen file changed: {path}"
    schedule = rows(DEST / "config/run_table.jsonl")
    assert len(schedule) == len({r["run_id"] for r in schedule}) == 1500
    return config, schedule


def prepare():
    if (DEST / "audit/preflight_report.json").exists():
        return verify_frozen()
    if (DEST / "logs/runs.jsonl").exists():
        raise RuntimeError("Existing inference without a frozen preflight; refusing overwrite")
    reference = read(OLD / "config/experiment_config.json")
    config = copy.deepcopy(reference)
    config.update({
        "experiment_version": VERSION, "scheduled_calls": 1500, "levels": ["L3"],
        "reference_experiment": str(OLD), "corrected_input_source": str(SOURCE),
        "image_crosscheck_source": str(RECENT),
        "task_definition": "Original L3: exact blocking reason AND valid visible blocker ID; FULL visibility and 5cm clearance rules. Not the revised direct-retrieval task.",
        "geometry_policy": "Original corrected R/F/M measurements, sorted visible-object records only. Hidden-object records excluded; remaining measurements unchanged.",
        "gt_policy": "Corrected source L3 reason unchanged. Valid blocker set intersected with visible IDs to match the original output validator.",
        "gripper_information_supplied": False,
    })
    for relative in set(reference["prompts"].values()) | set(reference["schemas"].values()):
        copy_file(OLD / relative, DEST / relative)
    for relative in ("prompts/system_en.txt", "prompts/l3_en.txt", "schemas/l3.json"):
        assert (OLD / relative).read_bytes() == (SOURCE / relative).read_bytes()
    coverage = read(SOURCE / "audit/object_coverage_report.json")
    assert coverage["status"] == "pass" and not coverage["failures"]
    copy_file(SOURCE / "audit/object_coverage_report.json", DEST / "audit/source_object_coverage_report.json")
    scenes = read(SOURCE / "config/frozen_scenes.json")
    assert len(scenes) == 25
    by_scene, scene_audit, payload_audit = {}, [], []
    for original in scenes:
        scene = copy.deepcopy(original)
        sid = scene["scene_id"]
        source_mapping = SOURCE / scene["id_mapping"]
        source_image = SOURCE / scene["numbered_rgb"]
        assert common.sha256_path(source_mapping) == scene["id_mapping_sha256"]
        assert common.sha256_path(source_image) == scene["numbered_rgb_sha256"]
        assert common.sha256_path(RECENT / scene["numbered_rgb"]) == scene["numbered_rgb_sha256"]
        mapping = read(source_mapping)
        old_mapping = read(OLD / scene["id_mapping"])
        recent_mapping = read(RECENT / scene["id_mapping"])
        identity = lambda m: sorted((r["display_id"], r["simulator_instance_id"], r["visible"]) for r in m["instances"])
        assert identity(mapping) == identity(recent_mapping)
        assert mapping["target_object_id"] == scene["target_object_id"]
        visible = {r["display_id"] for r in mapping["instances"] if r["visible"]}
        assert scene["target_object_id"] in visible
        old_by_id = {r["display_id"]: r for r in old_mapping["instances"]}
        restored = []
        for obj in mapping["instances"]:
            mask_path = SOURCE / f"audit/object_visible_masks/{sid}/{obj['display_id']:03d}_{obj['simulator_instance_id']}.png"
            with Image.open(mask_path) as im:
                mask = im.convert("L")
                area = sum(mask.histogram()[1:])
                assert area == obj["visible_area_px"]
                assert (area > 0) == obj["visible"]
                if obj["visible"]:
                    assert obj["badge_center_xy"] is not None
                    assert mask.getpixel(tuple(obj["badge_center_xy"])) > 0
                else:
                    assert obj["badge_center_xy"] is None
            assert old_by_id[obj["display_id"]]["simulator_instance_id"] == obj["simulator_instance_id"]
            if obj["visible"] and not old_by_id[obj["display_id"]]["visible"]:
                restored.append(obj["display_id"])
        copy_file(source_image, DEST / scene["numbered_rgb"])
        copy_file(source_mapping, DEST / scene["id_mapping"])
        copy_file(SOURCE / scene["scene_geometry"], DEST / scene["scene_geometry"])
        gt = read(SOURCE / scene["ground_truth"])
        assert common.sha256_path(SOURCE / scene["ground_truth"]) == scene["ground_truth_sha256"]
        physical_valid = gt["L3"]["valid_blocker_ids"][:]
        gt["L3"]["valid_blocker_ids"] = sorted(set(physical_valid) & visible)
        assert (gt["L3"]["blocking_reason"] == "NONE") == (len(gt["L3"]["valid_blocker_ids"]) == 0)
        gt["methods"]["l3_visible_blocker_policy"] = "valid_blocker_ids intersect visible display IDs; reason and physical blocker annotations unchanged"
        write(DEST / scene["ground_truth"], gt)
        scene["ground_truth_sha256"] = common.sha256_path(DEST / scene["ground_truth"])
        old_gt = read(OLD / scene["ground_truth"])
        for geometry in scene["geometry_files"].values():
            source_path = SOURCE / geometry["path"]
            assert common.sha256_path(source_path) == geometry["sha256"]
            payload = read(source_path)
            records_key = "relations" if "relations" in payload else "objects"
            original_records = payload[records_key]
            excluded = sorted(r["object_id"] for r in original_records if r["object_id"] not in visible)
            payload[records_key] = sorted((r for r in original_records if r["object_id"] in visible), key=lambda r: r["object_id"])
            expected = visible - {scene["target_object_id"]} if records_key == "relations" else visible
            assert {r["object_id"] for r in payload[records_key]} == expected
            assert len(payload[records_key]) == len(expected)
            serialized = json.dumps(payload).lower()
            assert not any(term in serialized for term in ("gripper", "finger_placement", "direct_graspable", "blocker_id", "best_action"))
            write(DEST / geometry["path"], payload)
            payload_audit.append({"scene_id": sid, "path": geometry["path"], "source_sha256": geometry["sha256"], "excluded_hidden_ids": excluded})
            geometry["sha256"] = common.sha256_path(DEST / geometry["path"])
        scene["visible_ids"] = sorted(visible)
        by_scene[sid] = scene
        scene_audit.append({
            "scene_id": sid, "image_sha256": scene["numbered_rgb_sha256"],
            "image_changed_vs_v1": common.sha256_path(OLD / scene["numbered_rgb"]) != scene["numbered_rgb_sha256"],
            "matches_recent_l2_image": True, "restored_visible_ids": restored,
            "old_l3_gt": old_gt["L3"], "source_l3_gt": read(SOURCE / scene["ground_truth"])["L3"],
            "effective_l3_gt": gt["L3"],
            "excluded_hidden_blocker_ids": sorted(set(physical_valid) - visible),
        })
    original_schedule = [r for r in rows(OLD / "config/run_table.jsonl") if r["level"] == "L3"]
    schedule = []
    for original in original_schedule:
        row = copy.deepcopy(original)
        scene = by_scene[row["scene_id"]]
        row["reference_run_id"] = row["run_id"]
        row["run_id"] = "v17__" + row["run_id"].removeprefix("v1__")
        row["experiment_version"] = VERSION
        for field in ("numbered_rgb", "numbered_rgb_sha256", "id_mapping", "id_mapping_sha256", "ground_truth", "ground_truth_sha256", "target_object_id"):
            row[field] = scene[field]
        if row["geometry_json"]:
            row["geometry_json_sha256"] = scene["geometry_files"][row["geometry_key"]]["sha256"]
        schedule.append(row)
    assert len(schedule) == len({r["run_id"] for r in schedule}) == 1500
    assert set(Counter(condition(r) for r in schedule).values()) == {125}
    assert sum(len(r["restored_visible_ids"]) for r in scene_audit) == 17
    assert sum(r["image_changed_vs_v1"] for r in scene_audit) == 15
    for sid in by_scene:
        for c, resolution in CONDITIONS:
            subset = [r for r in schedule if r["scene_id"] == sid and (r["geometry_condition"], r["direction_resolution"]) == (c, resolution)]
            assert len(subset) == 5 and {r["seed"] for r in subset} == set(config["main_seeds"])
    write(DEST / "config/experiment_config.json", config)
    write(DEST / "config/frozen_scenes.json", list(by_scene.values()))
    schedule_path = DEST / "config/run_table.jsonl"
    with schedule_path.open("w", encoding="utf-8") as handle:
        for row in schedule:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write(DEST / "audit/scene_input_changes.json", scene_audit)
    write(DEST / "audit/geometry_payloads.json", payload_audit)
    hashes = {str(p.relative_to(DEST)): common.sha256_path(p) for folder in ("config", "prompts", "schemas", "geometry", "images") for p in (DEST / folder).rglob("*") if p.is_file()}
    write(DEST / "audit/frozen_file_hashes.json", hashes)
    write(DEST / "audit/preflight_report.json", {
        "status": "pass", "checked_at": now(), "scheduled_calls": 1500,
        "run_table_sha256": common.sha256_path(schedule_path), "scene_count": 25,
        "all_25_images_match_corrected_and_recent_l2": True,
        "restored_objects": 17, "changed_image_scenes": 15,
        "visible_objects": 114, "physical_objects": 115,
        "frozen_file_count": len(hashes), "system_and_l3_prompt_unchanged": True,
        "generation_and_model_unchanged": True,
        "geometry_payloads_excluding_hidden_records": sum(bool(r["excluded_hidden_ids"]) for r in payload_audit),
        "hidden_gt_choices_excluded": [{"scene_id": r["scene_id"], "ids": r["excluded_hidden_blocker_ids"]} for r in scene_audit if r["excluded_hidden_blocker_ids"]],
        "failures": [],
    })
    return verify_frozen()


def expected_scores(row, gt):
    parsed = row.get("parsed_response")
    if not isinstance(parsed, dict):
        return {"reason_correct": False, "blocker_correct": False, "joint_correct": False}
    reason = parsed.get("blocking_reason") == gt["blocking_reason"]
    blocker = parsed.get("blocker_id") is None if gt["blocking_reason"] == "NONE" else parsed.get("blocker_id") in gt["valid_blocker_ids"]
    return {"reason_correct": reason, "blocker_correct": blocker, "joint_correct": reason and blocker}


def metrics(records):
    return {"n": len(records), **{field: sum(r[field] for r in records) for field in ("reason_correct", "blocker_correct", "joint_correct")}}


def analyze():
    config, schedule = verify_frozen()
    raw = rows(DEST / "logs/runs.jsonl")
    assert len(raw) == len({r["run_id"] for r in raw}) == 1500
    planned = {r["run_id"]: r for r in schedule}
    assert {r["run_id"] for r in raw} == set(planned)
    old_rows = [r for r in rows(OLD / "logs/runs.jsonl") if r["level"] == "L3"]
    old = {pair_key(r): r for r in old_rows}
    source_gts = {s["scene_id"]: read(DEST / s["ground_truth"])["L3"] for s in read(DEST / "config/frozen_scenes.json")}
    by_condition, by_scene = defaultdict(list), defaultdict(list)
    system = (DEST / "prompts/system_en.txt").read_text().strip()
    task = (DEST / "prompts/l3_en.txt").read_text().strip()
    schema = read(DEST / "schemas/l3.json")
    for row in raw:
        assert all(row[k] == v for k, v in planned[row["run_id"]].items())
        assert row["generation"] == config["generation"] and row["model_id"] == config["model_id"]
        assert row["system_prompt"] == system and row["user_prompt"].endswith(task)
        assert row["json_schema"] == schema and row["level"] == "L3"
        payload = read(DEST / row["geometry_json"]) if row["geometry_json"] else None
        assert row["geometry_json_raw"] == payload
        geometry_text = "null" if payload is None else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        assert row["user_prompt"] == f"Target object ID: {row['target_object_id']}\n\nGeometry information:\n{geometry_text}\n\n{task}"
        new_scores = expected_scores(row, source_gts[row["scene_id"]])
        assert row["scoring"]["blocking_reason_correct"] == new_scores["reason_correct"]
        assert row["scoring"]["blocker_valid_set_member"] == new_scores["blocker_correct"]
        assert row["scoring"]["task_correct"] == new_scores["joint_correct"]
        new_scores["joint_correct"] &= row["post_validation_pass"]
        record = {**row, **new_scores}
        by_condition[condition(row)].append(record)
        by_scene[(row["scene_id"], condition(row))].append(record)
    conditions, scene_results, changed = [], [], []
    for c, res in CONDITIONS:
        key = c + ("_" + res if res else "")
        group = by_condition[key]
        assert len(group) == 125
        previous = [old[pair_key(r)] for r in group]
        prev_on_new = [expected_scores(r, source_gts[r["scene_id"]]) for r in previous]
        result = {"condition_key": key, **metrics(group),
            "old_v1_joint_correct_original_gt": sum(r["scoring"]["task_correct"] for r in previous),
            "old_v1_joint_correct_rescored_new_gt": sum(r["joint_correct"] for r in prev_on_new),
            "by_family": {f: metrics([r for r in group if r["scene_family"] == f]) for f in FAMILIES},
            "valid": sum(r["post_validation_pass"] for r in group),
            "changed_outputs_vs_v1": sum(r["parsed_response"] != old[pair_key(r)]["parsed_response"] for r in group),
        }
        conditions.append(result)
        for sid in sorted(source_gts):
            group_scene = sorted(by_scene[(sid, key)], key=lambda r: r["seed"])
            assert len(group_scene) == 5 and {r["seed"] for r in group_scene} == set(config["main_seeds"])
            scene_results.append({"scene_id": sid, "condition_key": key, "target_object_id": group_scene[0]["target_object_id"], "scene_family": group_scene[0]["scene_family"], "image_path": str(DEST / group_scene[0]["numbered_rgb"]), "gt": source_gts[sid], **metrics(group_scene), "outputs": [{"seed": r["seed"], "response": r["parsed_response"], "valid": r["post_validation_pass"], **{f: r[f] for f in ("reason_correct", "blocker_correct", "joint_correct")}} for r in group_scene]})
        for r in group:
            prior = old[pair_key(r)]
            if r["parsed_response"] != prior["parsed_response"]:
                changed.append({"scene_id": r["scene_id"], "condition_key": key, "seed": r["seed"], "old": prior["parsed_response"], "new": r["parsed_response"], "old_gt": read(OLD / prior["ground_truth"])["L3"], "new_gt": source_gts[r["scene_id"]]})
    audit = {
        "status": "pass", "completed": len(raw), "unique_run_ids": len(planned),
        "all_input_hashes_verified": True, "all_25_images_match_corrected_and_recent_l2": True,
        "all_scene_condition_seed_sets_complete": True,
        "parse_errors": sum(r["parse_error"] is not None for r in raw),
        "post_validation_failures": sum(not r["post_validation_pass"] for r in raw),
        "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in rows(DEST / "logs/attempts.jsonl")),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in raw)),
        "condition_counts": dict(Counter(condition(r) for r in raw)),
        "logs_sha256": common.sha256_path(DEST / "logs/runs.jsonl"), "finished_at": now(),
    }
    summary = {"experiment_version": VERSION, "model_id": config["model_id"], "task_definition": config["task_definition"], "audit": audit, "condition_results": conditions}
    write(DEST / "results/summary.json", summary)
    write(DEST / "results/scene_results.json", scene_results)
    write(DEST / "results/changed_outputs_vs_v1.json", changed)
    write(DEST / "audit/completion_report.json", audit)
    fields = ["condition_key", "n", "reason_correct", "blocker_correct", "joint_correct", "old_v1_joint_correct_original_gt", "old_v1_joint_correct_rescored_new_gt", "valid", "changed_outputs_vs_v1"]
    with (DEST / "results/condition_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(conditions)
    scene_fields = ["scene_id", "condition_key", "target_object_id", "scene_family", "reason_correct", "blocker_correct", "joint_correct"]
    with (DEST / "results/scene_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scene_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scene_results)
    lines = ["# L3 corrected-input rerun v17", "", "25 scenes × 12 conditions × 5 seeds = 1,500 independent calls. Original v1 L3 prompt, model, generation settings and output schema are unchanged. Corrected numbered images and matching geometry/GT are used; every image matches the corrected input used in recent L2 runs.", "", "This remains the original FULL-visibility/5cm-clearance L3 task, NOT the revised L2 direct-retrieval task. Joint correctness requires the exact reason and one valid visible blocker (or null for NONE). It is not physical retrieval success.", "", "15 images differ from v1, restoring 17 previously missing visible badges. Geometry includes only visible-object records. Fully hidden ID 16 in lift_v04 is excluded from 11 geometry payloads and the selectable blocker set; reason and physical measurements are unchanged. Old-v1 columns are historical references, not a clean image-only ablation.", "", f"Completion: {len(raw)}/1500; parse errors: {audit['parse_errors']}; semantic validation failures: {audit['post_validation_failures']}; infrastructure errors: {audit['infrastructure_errors']}", "", "| Condition | Reason /125 | Blocker /125 | Joint /125 | Joint accuracy | Old v1 joint, original GT | Old v1 joint, rescored new GT |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in conditions:
        lines.append(f"| {r['condition_key']} | {r['reason_correct']} | {r['blocker_correct']} | {r['joint_correct']} | {r['joint_correct']/125:.1%} | {r['old_v1_joint_correct_original_gt']} | {r['old_v1_joint_correct_rescored_new_gt']} |")
    keys = [c + ("_" + r if r else "") for c, r in CONDITIONS]
    lookup = {(r["scene_id"], r["condition_key"]): r for r in scene_results}
    lines += ["", "## Per-scene reason / blocker / joint correct counts (each out of five)", "", "| Scene | Target | " + " | ".join(keys) + " |", "|---|---:|" + "---|" * len(keys)]
    for sid in sorted(source_gts):
        lines.append(f"| {sid} | {lookup[(sid, 'C0')]['target_object_id']} | " + " | ".join(" / ".join(str(lookup[(sid, k)][f]) for f in ("reason_correct", "blocker_correct", "joint_correct")) for k in keys) + " |")
    (DEST / "results/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze()
        return
    prepare()
    print(json.dumps(read(DEST / "audit/preflight_report.json"), indent=2), flush=True)
    if args.prepare_only:
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/run_vlm_action_geometry_single_info_v1.py"), "--experiment", str(DEST), "--level", "L3"], check=True)
    analyze()


if __name__ == "__main__":
    main()
