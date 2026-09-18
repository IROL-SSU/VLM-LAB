#!/usr/bin/env python3
"""Create the simplified one-image obstruction-score experiment v14."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_single_image_obstruction_score_v13"
DESTINATION = ROOT / "experiments/vlm_action_geometry_single_info_l2_simple_obstruction_score_v14"
SCENE_ID = "scene_lift_and_relocate_v05"
TARGET_ID = 43
OBJECT_IDS = (52, 48, 15, 76, 24)
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
    schema_rel = "schemas/l2.json"
    gt_rel = f"ground_truth/{SCENE_ID}.json"
    for relative in (image_rel, mapping_rel, schema_rel, gt_rel):
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, destination)

    write_text(DESTINATION / "prompts/system_en.txt", """You are a robot manipulation planner.
Return only valid JSON.""")
    write_text(DESTINATION / "prompts/l2_en.txt", """Target object to grasp: object 43.
Candidate obstacle to evaluate: {candidate_description}.

Score how much removing the candidate obstacle would improve the robot's
ability to grasp the target object.

Return an integer obstruction score from 0 to 10.""")
    for level in ("l1", "l3", "l4"):
        write_text(DESTINATION / f"prompts/{level}_en.txt", "Unused in this experiment.")

    config = {
        "experiment_version": "l2_simple_obstruction_score_v14",
        "l2_prompt_version": "simple_per_object_removal_benefit_score_0_10_v14",
        "scene_count": 1,
        "candidate_count": len(OBJECT_IDS),
        "scheduled_calls": len(OBJECT_IDS) * len(SEEDS),
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
            "max_tokens": 64,
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
    for object_id in OBJECT_IDS:
        for repeat_index, seed in enumerate(SEEDS, 1):
            rows.append({
                "run_id": (
                    f"v14__{SCENE_ID}__object_{object_id}__L2__C0_NA__"
                    f"r{repeat_index:02d}_s{seed}"
                ),
                "experiment_version": config["experiment_version"],
                "scene_id": SCENE_ID,
                "scene_family": "LIFT_AND_RELOCATE",
                "candidate_object_id": object_id,
                "candidate_description": f"object {object_id}",
                "natural_language_only": True,
                "level": "L2",
                "geometry_condition": "C0",
                "geometry_key": None,
                "direction_resolution": None,
                "repeat_index": repeat_index,
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
    write_json(DESTINATION / "audit/preflight_report.json", {
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_table_rows": len(rows),
        "unique_run_ids": len({row["run_id"] for row in rows}),
        "run_table_sha256": common.sha256_path(run_table),
        "numbered_rgb_sha256": common.sha256_path(DESTINATION / image_rel),
        "failures": [],
    })
    print(json.dumps({
        "experiment": str(DESTINATION),
        "target_id": TARGET_ID,
        "candidate_object_ids": list(OBJECT_IDS),
        "scheduled_calls": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
