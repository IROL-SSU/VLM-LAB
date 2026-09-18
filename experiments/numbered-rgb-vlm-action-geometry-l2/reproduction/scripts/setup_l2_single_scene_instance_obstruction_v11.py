#!/usr/bin/env python3
"""Create a five-seed, RGB-only instance obstruction experiment for one scene."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10"
DESTINATION = ROOT / "experiments/vlm_action_geometry_single_info_l2_single_scene_instance_obstruction_v11"
SCENE_ID = "scene_lift_and_relocate_v05"
TARGET_ID = 43
SEEDS = (28101, 28102, 28103, 28104, 28105)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def main() -> None:
    image_rel = f"images/numbered_rgb/{SCENE_ID}.png"
    mapping_rel = f"images/id_mappings/{SCENE_ID}.json"
    gt_rel = f"geometry/ground_truth/{SCENE_ID}.json"
    schema_rel = f"schemas/l2_instances/{SCENE_ID}.json"
    for relative in (image_rel, mapping_rel):
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, destination)

    mapping = json.loads((DESTINATION / mapping_rel).read_text(encoding="utf-8"))
    visible_ltr = sorted(
        (item for item in mapping["instances"] if item["visible"]),
        key=lambda item: (item["visible_centroid_xy"][0], int(item["display_id"])),
    )
    assessment_ids = [
        int(item["display_id"])
        for item in visible_ltr
        if int(item["display_id"]) != TARGET_ID
    ]
    source_gt = json.loads(
        (SOURCE / f"geometry/ground_truth/{SCENE_ID}.json").read_text(encoding="utf-8")
    )
    visual_blocker = source_gt["L2"]["object_to_remove_id"]
    gt = {
        "scene_id": SCENE_ID,
        "target_object_id": TARGET_ID,
        "L2": {
            "target_id": TARGET_ID,
            "grasp_obstruction_assessments_left_to_right": [
                {
                    "object_id": object_id,
                    "obstructs_grasp": object_id == visual_blocker,
                }
                for object_id in assessment_ids
            ],
            "reference_only": True,
            "reference_definition": "Isaac Sim visual occlusion >= 5%; not robot grasp rollout",
        },
    }
    write_json(DESTINATION / gt_rel, gt)

    assessment_schemas = [
        {
            "type": "object",
            "properties": {
                "object_id": {"const": object_id},
                "obstructs_grasp": {"type": "boolean"},
            },
            "required": ["object_id", "obstructs_grasp"],
            "additionalProperties": False,
        }
        for object_id in assessment_ids
    ]
    schema = {
        "type": "object",
        "properties": {
            "target_id": {"const": TARGET_ID},
            "grasp_obstruction_assessments_left_to_right": {
                "type": "array",
                "prefixItems": assessment_schemas,
                "items": False,
                "minItems": len(assessment_schemas),
                "maxItems": len(assessment_schemas),
            },
        },
        "required": ["target_id", "grasp_obstruction_assessments_left_to_right"],
        "additionalProperties": False,
    }
    write_json(DESTINATION / schema_rel, schema)

    system = """You are a robot manipulation planner.
Return only valid JSON."""
    task = """The robot must grasp the target object.

For each numbered object other than the target, determine whether that object
obstructs grasping the target.

Return one assessment for every non-target numbered object, from left to right."""
    write_text(DESTINATION / "prompts/system_en.txt", system)
    write_text(DESTINATION / "prompts/l2_en.txt", task)
    for level in ("l1", "l3", "l4"):
        write_text(DESTINATION / f"prompts/{level}_en.txt", "Unused in this experiment.")

    config = {
        "experiment_version": "l2_single_scene_instance_obstruction_v11",
        "l2_prompt_version": "single_scene_instance_obstruction_v11",
        "scene_count": 1,
        "scheduled_calls": len(SEEDS),
        "main_seeds": list(SEEDS),
        "run_order_seed": 20260917,
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_path": str(ROOT / "models/Qwen3-VL-30B-A3B-Instruct"),
        "generation": {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 1024,
        },
        "runtime": {
            "backend": "vllm_offline",
            "batch_size": 1,
            "max_num_seqs": 1,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
            "moe_backend": "triton",
            "structured_output": "json_schema",
        },
        "prompts": {
            "system": "prompts/system_en.txt",
            "l1": "prompts/l1_en.txt",
            "l2": "prompts/l2_en.txt",
            "l3": "prompts/l3_en.txt",
            "l4": "prompts/l4_en.txt",
        },
    }
    write_json(DESTINATION / "config/experiment_config.json", config)

    rows = []
    for index, seed in enumerate(SEEDS, 1):
        rows.append({
            "run_id": f"v11__{SCENE_ID}__L2__C0_NA__r{index:02d}_s{seed}",
            "experiment_version": config["experiment_version"],
            "scene_id": SCENE_ID,
            "scene_family": "LIFT_AND_RELOCATE",
            "level": "L2",
            "geometry_condition": "C0",
            "geometry_key": None,
            "direction_resolution": None,
            "repeat_index": index,
            "seed": seed,
            "target_object_id": TARGET_ID,
            "numbered_rgb": image_rel,
            "numbered_rgb_sha256": common.sha256_path(DESTINATION / image_rel),
            "source_usd": None,
            "source_usd_sha256": None,
            "id_mapping": mapping_rel,
            "id_mapping_sha256": common.sha256_path(DESTINATION / mapping_rel),
            "ground_truth": gt_rel,
            "ground_truth_sha256": common.sha256_path(DESTINATION / gt_rel),
            "geometry_json": None,
            "geometry_json_sha256": common.sha256_text("null"),
            "schema": schema_rel,
            "schema_sha256": common.sha256_path(DESTINATION / schema_rel),
        })
    random.Random(config["run_order_seed"]).shuffle(rows)
    run_table = DESTINATION / "config/run_table.jsonl"
    run_table.parent.mkdir(parents=True, exist_ok=True)
    run_table.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    preflight = {
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_table_rows": len(rows),
        "unique_run_ids": len({row["run_id"] for row in rows}),
        "run_table_sha256": common.sha256_path(run_table),
        "failures": [],
    }
    write_json(DESTINATION / "audit/preflight_report.json", preflight)
    print(json.dumps({
        "experiment": str(DESTINATION),
        "assessment_ids_left_to_right": assessment_ids,
        "visual_reference_blocker": visual_blocker,
        "scheduled_calls": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
