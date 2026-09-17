#!/usr/bin/env python3
"""Freeze the two missing task/dataset cells for the initial N3 75-scene benchmark.

The original experiment evaluated left-to-right only on the 27 non-occluded
N3-D0 scenes and front-to-back only on the 48 occluded N3-D1 scenes.  This
setup keeps the exact images and evaluation conditions while crossing tasks:

* N3-D0 (27 scenes): front-to-back
* N3-D1 (48 scenes): left-to-right
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from setup_n3_d0_ltr_d1_depth_75scenes import (
    D0_PROMPT,
    D1_PROMPT,
    SCHEMAS,
    sha256,
    write_json,
)


ROOT = Path("/home/ssu/ShelfScene")
SOURCE = ROOT / "benchmark_n3/depth_reasoning_75scenes_randomized"
ORIGINAL = ROOT / "experiments/n3_d0_ltr_d1_depth_75scenes_5seeds_20260828"
EXPERIMENT = ROOT / "experiments/n3_initial75_cross_tasks_75scenes_5seeds_20260906"


def main() -> None:
    if EXPERIMENT.exists() and any(EXPERIMENT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {EXPERIMENT}")
    for name in ("config", "prompts", "scenes/numbered_rgb", "logs", "results", "reports", "audit"):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)

    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    samples = metadata["samples"]
    if len(samples) != 75:
        raise ValueError(f"expected 75 source scenes, found {len(samples)}")

    frozen: list[dict[str, Any]] = []
    d1_center_orders_agree = True
    for sample in samples:
        source_split = sample["benchmark_split"]
        badge_by_instance = {
            badge["instance_id"]: int(badge["display_id"])
            for badge in sample["badges"]
        }
        candidate_ids = sorted(badge_by_instance.values())

        if source_split == "N3-D0":
            task_id = "D1_FRONT_TO_BACK"
            output_key = "front_to_back"
            benchmark_split = "N3-D0-FB-CROSS"
            expected_order = [int(value) for value in sample["ground_truth"]["front_to_back"]]
            annotation = {
                "camera_depth_cm": [
                    float(value) for value in sample["ground_truth"]["camera_depth_cm"]
                ],
                "adjacent_depth_margins_cm": [
                    float(value)
                    for value in sample["ground_truth"]["adjacent_depth_margins_cm"]
                ],
            }
        elif source_split == "N3-D1":
            task_id = "D0_LEFT_TO_RIGHT"
            output_key = "left_to_right"
            benchmark_split = "N3-D1-LR-CROSS"
            centroids = {
                badge_by_instance[instance_id]: float(box["center_xy"][0])
                for instance_id, box in sample["unoccluded_projected_bboxes"].items()
            }
            bbox_midpoints = {
                badge_by_instance[instance_id]: (
                    float(box["xyxy"][0]) + float(box["xyxy"][2])
                ) / 2.0
                for instance_id, box in sample["unoccluded_projected_bboxes"].items()
            }
            expected_order = sorted(candidate_ids, key=centroids.__getitem__)
            midpoint_order = sorted(candidate_ids, key=bbox_midpoints.__getitem__)
            d1_center_orders_agree &= expected_order == midpoint_order
            annotation = {
                "horizontal_mask_centroid_x_px": centroids,
                "horizontal_bbox_midpoint_x_px": bbox_midpoints,
                "bbox_midpoint_order": midpoint_order,
            }
        else:
            raise ValueError(f"unknown benchmark split: {source_split}")

        if sorted(expected_order) != candidate_ids:
            raise ValueError(f"GT/candidate mismatch for {sample['scene_id']}")

        source_image = SOURCE / sample["numbered_rgb"]
        target_rel = Path("scenes/numbered_rgb") / source_image.name
        target_image = EXPERIMENT / target_rel
        shutil.copy2(source_image, target_image)
        image_hash = sha256(target_image)
        if image_hash != sample["numbered_rgb_sha256"]:
            raise RuntimeError(f"image hash mismatch: {sample['scene_id']}")

        frozen.append({
            "scene_id": sample["scene_id"],
            "source_benchmark_split": source_split,
            "benchmark_split": benchmark_split,
            "task_id": task_id,
            "output_key": output_key,
            "image": str(target_rel),
            "image_sha256": image_hash,
            "visible_ids": candidate_ids,
            "expected_order": expected_order,
            "annotation": annotation,
            "factors": {
                "object_count": int(sample["object_count"]),
                "size_cue": sample["size_cue"],
                "occlusion_level": sample["occlusion_level"],
                "occluder_side": sample["occluder_side"],
                "variant_index": int(sample["variant_index"]),
            },
        })

    seeds = [28101, 28102, 28103, 28104, 28105]
    config = {
        "experiment_id": EXPERIMENT.name,
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_path": str(ROOT / "models/Qwen3-VL-30B-A3B-Instruct"),
        "source_dataset": str(SOURCE),
        "original_experiment": str(ORIGINAL),
        "scene_count": len(frozen),
        "split_scene_counts": dict(Counter(scene["benchmark_split"] for scene in frozen)),
        "source_split_scene_counts": dict(
            Counter(scene["source_benchmark_split"] for scene in frozen)
        ),
        "task_assignment": {
            "N3-D0": "D1_FRONT_TO_BACK",
            "N3-D1": "D0_LEFT_TO_RIGHT",
        },
        "main_seeds": seeds,
        "scheduled_calls": len(frozen) * len(seeds),
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
        },
        "prompt_language": "English",
        "system_prompt": None,
        "input_candidate_order": "ascending numeric ID; independent of spatial GT",
        "output": "forced JSON",
        "primary_metrics": {
            "N3-D0-FB-CROSS": "front_to_back exact",
            "N3-D1-LR-CROSS": "left_to_right exact",
        },
    }
    write_json(EXPERIMENT / "config/experiment_config.json", config)
    write_json(EXPERIMENT / "config/frozen_scenes.json", frozen)
    write_json(
        EXPERIMENT / "config/json_schema_d0.json",
        SCHEMAS["D0_LEFT_TO_RIGHT"],
        sort_keys=False,
    )
    write_json(
        EXPERIMENT / "config/json_schema_d1.json",
        SCHEMAS["D1_FRONT_TO_BACK"],
        sort_keys=False,
    )
    (EXPERIMENT / "prompts/d0_left_to_right_en.txt").write_text(
        D0_PROMPT + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "prompts/d1_front_to_back_en.txt").write_text(
        D1_PROMPT + "\n", encoding="utf-8"
    )

    original_prompt_hashes = {
        "d0_left_to_right": sha256(ORIGINAL / "prompts/d0_left_to_right_en.txt"),
        "d1_front_to_back": sha256(ORIGINAL / "prompts/d1_front_to_back_en.txt"),
    }
    new_prompt_hashes = {
        "d0_left_to_right": sha256(EXPERIMENT / "prompts/d0_left_to_right_en.txt"),
        "d1_front_to_back": sha256(EXPERIMENT / "prompts/d1_front_to_back_en.txt"),
    }
    audit = {
        "scene_count": len(frozen),
        "scheduled_calls": config["scheduled_calls"],
        "split_scene_counts": config["split_scene_counts"],
        "all_candidate_ids_ascending": all(
            scene["visible_ids"] == sorted(scene["visible_ids"]) for scene in frozen
        ),
        "all_expected_orders_are_permutations": all(
            sorted(scene["expected_order"]) == scene["visible_ids"] for scene in frozen
        ),
        "all_images_match": all(
            sha256(EXPERIMENT / scene["image"]) == scene["image_sha256"] for scene in frozen
        ),
        "all_source_images_match_original_experiment": all(
            sha256(EXPERIMENT / scene["image"])
            == sha256(ORIGINAL / scene["image"])
            for scene in frozen
        ),
        "all_prompt_hashes_match_original_experiment": (
            new_prompt_hashes == original_prompt_hashes
        ),
        "d1_centroid_and_bbox_midpoint_orders_agree": d1_center_orders_agree,
        "schemas_have_no_array_or_id_bounds": not any(
            token in json.dumps(SCHEMAS)
            for token in ("minimum", "maximum", "minItems", "maxItems")
        ),
        "batch_size": 1,
        "system_prompt_absent": config["system_prompt"] is None,
        "original_prompt_hashes": original_prompt_hashes,
        "new_prompt_hashes": new_prompt_hashes,
    }
    if not all(value for key, value in audit.items() if key.startswith("all_")):
        raise RuntimeError("setup validation failed")
    if not d1_center_orders_agree:
        raise RuntimeError("D1 horizontal center definitions produce different orders")
    write_json(EXPERIMENT / "audit/setup_audit.json", audit)
    print(json.dumps({"experiment": str(EXPERIMENT), **audit}, indent=2))


if __name__ == "__main__":
    main()
