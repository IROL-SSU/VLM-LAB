#!/usr/bin/env python3
"""Create a plain-RGB named-object grasp-obstruction experiment for one scene."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCENES = ROOT / "experiments/action_scene_variants_20260915"
SOURCE_META = ROOT / "experiments/vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10"
DESTINATION = ROOT / "experiments/vlm_action_geometry_single_info_l2_plain_rgb_named_object_obstruction_v12"
SCENE_ID = "scene_lift_and_relocate_v05"
SEEDS = (28101, 28102, 28103, 28104, 28105)
TARGET_DESCRIPTION = (
    "the light blue cup partially visible immediately to the right of the "
    "Cheez-It box"
)
CANDIDATES = (
    ("black_mug", "the black mug on the left", False),
    (
        "dark_blue_bottle",
        "the dark blue bottle immediately to the left of and partly behind the Cheez-It box",
        False,
    ),
    ("cheezit_box", "the Cheez-It box in the center", True),
    ("purple_cup", "the purple cup on the right", False),
    ("brown_mug", "the brown mug behind the purple cup on the right", False),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def main() -> None:
    image_rel = f"images/rgb/{SCENE_ID}.png"
    mapping_rel = f"metadata/{SCENE_ID}_id_mapping.json"
    image_destination = DESTINATION / image_rel
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_SCENES / f"images/{SCENE_ID}.png", image_destination)
    mapping_destination = DESTINATION / mapping_rel
    mapping_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        SOURCE_META / f"images/id_mappings/{SCENE_ID}.json", mapping_destination
    )

    system = """You are a robot manipulation planner.
Return only valid JSON."""
    task = f"""The robot must grasp {TARGET_DESCRIPTION}.

Does {{candidate_description}} obstruct grasping the light blue cup?
Return the answer as obstructs_grasp=true or obstructs_grasp=false."""
    write_text(DESTINATION / "prompts/system_en.txt", system)
    write_text(DESTINATION / "prompts/l2_en.txt", task)
    for level in ("l1", "l3", "l4"):
        write_text(DESTINATION / f"prompts/{level}_en.txt", "Unused in this experiment.")

    schema = {
        "type": "object",
        "properties": {"obstructs_grasp": {"type": "boolean"}},
        "required": ["obstructs_grasp"],
        "additionalProperties": False,
    }
    schema_rel = "schemas/l2.json"
    write_json(DESTINATION / schema_rel, schema)

    config = {
        "experiment_version": "l2_plain_rgb_named_object_obstruction_v12",
        "l2_prompt_version": "plain_rgb_named_object_obstruction_v12",
        "scene_count": 1,
        "candidate_count": len(CANDIDATES),
        "scheduled_calls": len(CANDIDATES) * len(SEEDS),
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
            "max_tokens": 128,
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
    for candidate_key, candidate_description, visual_reference in CANDIDATES:
        gt_rel = f"ground_truth/{candidate_key}.json"
        write_json(DESTINATION / gt_rel, {
            "scene_id": SCENE_ID,
            "candidate_key": candidate_key,
            "candidate_description": candidate_description,
            "L2": {
                "obstructs_grasp": visual_reference,
                "reference_only": True,
                "reference_definition": (
                    "Isaac Sim visual-occlusion reference; not robot-arm grasp rollout"
                ),
            },
        })
        for repeat_index, seed in enumerate(SEEDS, 1):
            rows.append({
                "run_id": (
                    f"v12__{SCENE_ID}__{candidate_key}__L2__C0_NA__"
                    f"r{repeat_index:02d}_s{seed}"
                ),
                "experiment_version": config["experiment_version"],
                "scene_id": SCENE_ID,
                "scene_family": "LIFT_AND_RELOCATE",
                "candidate_key": candidate_key,
                "candidate_description": candidate_description,
                "natural_language_only": True,
                "level": "L2",
                "geometry_condition": "C0",
                "geometry_key": None,
                "direction_resolution": None,
                "repeat_index": repeat_index,
                "seed": seed,
                "target_object_id": 43,
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
    write_json(DESTINATION / "audit/preflight_report.json", {
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_table_rows": len(rows),
        "unique_run_ids": len({row["run_id"] for row in rows}),
        "run_table_sha256": common.sha256_path(run_table),
        "plain_rgb_sha256": common.sha256_path(DESTINATION / image_rel),
        "failures": [],
    })
    print(json.dumps({
        "experiment": str(DESTINATION),
        "target_description": TARGET_DESCRIPTION,
        "candidates": [item[0] for item in CANDIDATES],
        "scheduled_calls": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
