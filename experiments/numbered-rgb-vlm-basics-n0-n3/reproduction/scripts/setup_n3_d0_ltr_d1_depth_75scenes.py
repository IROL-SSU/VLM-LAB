#!/usr/bin/env python3
"""Freeze the N3 D0 left-to-right and D1 front-to-back experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
SOURCE = ROOT / "benchmark_n3/depth_reasoning_75scenes_randomized"
EXPERIMENT = ROOT / "experiments/n3_d0_ltr_d1_depth_75scenes_5seeds_20260828"


D0_PROMPT = r'''Examine the numbered RGB image.

The following IDs are confirmed to be the complete and correct set of all visible object-marker IDs:
{visible_ids_json}

Order all confirmed IDs from left to right according to the positions of their associated physical objects in the image.

Determine the horizontal position of each object using the horizontal center
of the complete physical body of the object.

The red numbered marker only identifies the object.
Do not use the position of the numbered marker itself as the horizontal position of the object.

If an object is partially occluded by another object, use the estimated horizontal center
of the complete physical body rather than the center of only its visible region
or the position of its numbered marker.

Do not use vertical position, apparent size, camera distance, or depth
to determine horizontal order.

Process every confirmed ID exactly once.

In the visible_ids array, return the complete provided ID list in its original order.

In the left_to_right array, return the same complete set of IDs ordered
from the leftmost physical object to the rightmost physical object in the image.

Before returning the response, verify all of the following:

1. visible_ids exactly matches the complete provided ID list.
2. Every confirmed ID appears exactly once in left_to_right.
3. No ID is duplicated in left_to_right.
4. No ID outside the confirmed ID set has been added.
5. left_to_right is ordered using the horizontal centers of the complete physical objects.

Return only one valid JSON object using the following structure:

{
"visible_ids": [],
"left_to_right": []
}

Do not output explanations, Markdown, confidence scores, coordinates,
object descriptions, or additional fields.'''


D1_PROMPT = r'''Examine the numbered RGB image.

The following IDs are confirmed to be the complete and correct set of all visible object-marker IDs:
{visible_ids_json}

Order all confirmed IDs from the physical object closest to the camera
to the physical object farthest from the camera.

In this prompt:

- "Front" means closer to the camera.
- "Back" means farther from the camera.
- front_to_back means ordering the objects from closest to farthest from the camera.

Determine the front-to-back position using the depth of the physical object associated
with each numbered marker, not the numbered marker itself.

The red numbered marker only identifies the object.
Do not use the position or size of the numbered marker as evidence of object depth.

Do not determine front-to-back order using apparent object size alone.
Different objects may have different physical sizes.

If objects overlap or partially occlude one another, use all visible spatial evidence,
including the occlusion relationships between the physical objects.

Do not determine front-to-back order using horizontal position alone.

Process every confirmed ID exactly once.

In the visible_ids array, return the complete provided ID list in its original order.

In the front_to_back array, return the same complete set of IDs ordered
from the physical object closest to the camera to the physical object
farthest from the camera.

Before returning the response, verify all of the following:

1. visible_ids exactly matches the complete provided ID list.
2. Every confirmed ID appears exactly once in front_to_back.
3. No ID is duplicated in front_to_back.
4. No ID outside the confirmed ID set has been added.
5. The first ID in front_to_back is the object closest to the camera.
6. The final ID in front_to_back is the object farthest from the camera.
7. front_to_back is correctly ordered using the depths of the physical objects.

Return only one valid JSON object using the following structure:

{
"visible_ids": [],
"front_to_back": []
}

Do not output explanations, Markdown, confidence scores, coordinates,
object descriptions, or additional fields.'''


SCHEMAS: dict[str, dict[str, Any]] = {
    "D0_LEFT_TO_RIGHT": {
        "type": "object",
        "properties": {
            "visible_ids": {"type": "array", "items": {"type": "integer"}},
            "left_to_right": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["visible_ids", "left_to_right"],
        "additionalProperties": False,
    },
    "D1_FRONT_TO_BACK": {
        "type": "object",
        "properties": {
            "visible_ids": {"type": "array", "items": {"type": "integer"}},
            "front_to_back": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["visible_ids", "front_to_back"],
        "additionalProperties": False,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


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
    center_definition_orders_agree = True
    for sample in samples:
        split = sample["benchmark_split"]
        badge_by_instance = {
            badge["instance_id"]: int(badge["display_id"])
            for badge in sample["badges"]
        }
        candidate_ids = sorted(badge_by_instance.values())
        if split == "N3-D0":
            task_id = "D0_LEFT_TO_RIGHT"
            output_key = "left_to_right"
            centroids = {
                badge_by_instance[instance_id]: float(box["center_xy"][0])
                for instance_id, box in sample["unoccluded_projected_bboxes"].items()
            }
            bbox_midpoints = {
                badge_by_instance[instance_id]: (float(box["xyxy"][0]) + float(box["xyxy"][2])) / 2.0
                for instance_id, box in sample["unoccluded_projected_bboxes"].items()
            }
            expected_order = sorted(candidate_ids, key=centroids.__getitem__)
            midpoint_order = sorted(candidate_ids, key=bbox_midpoints.__getitem__)
            center_definition_orders_agree &= expected_order == midpoint_order
            annotation = {
                "horizontal_mask_centroid_x_px": centroids,
                "horizontal_bbox_midpoint_x_px": bbox_midpoints,
                "bbox_midpoint_order": midpoint_order,
            }
        elif split == "N3-D1":
            task_id = "D1_FRONT_TO_BACK"
            output_key = "front_to_back"
            expected_order = [int(value) for value in sample["ground_truth"]["front_to_back"]]
            annotation = {
                "camera_depth_cm": [float(value) for value in sample["ground_truth"]["camera_depth_cm"]],
                "adjacent_depth_margins_cm": [
                    float(value) for value in sample["ground_truth"]["adjacent_depth_margins_cm"]
                ],
            }
        else:
            raise ValueError(f"unknown benchmark split: {split}")

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
            "benchmark_split": split,
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
        "scene_count": len(frozen),
        "split_scene_counts": dict(Counter(scene["benchmark_split"] for scene in frozen)),
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
            "N3-D0": "left_to_right exact",
            "N3-D1": "front_to_back exact",
        },
    }
    write_json(EXPERIMENT / "config/experiment_config.json", config)
    write_json(EXPERIMENT / "config/frozen_scenes.json", frozen)
    write_json(EXPERIMENT / "config/json_schema_d0.json", SCHEMAS["D0_LEFT_TO_RIGHT"], sort_keys=False)
    write_json(EXPERIMENT / "config/json_schema_d1.json", SCHEMAS["D1_FRONT_TO_BACK"], sort_keys=False)
    (EXPERIMENT / "prompts/d0_left_to_right_en.txt").write_text(D0_PROMPT + "\n", encoding="utf-8")
    (EXPERIMENT / "prompts/d1_front_to_back_en.txt").write_text(D1_PROMPT + "\n", encoding="utf-8")

    audit = {
        "scene_count": len(frozen),
        "split_scene_counts": dict(Counter(scene["benchmark_split"] for scene in frozen)),
        "scheduled_calls": config["scheduled_calls"],
        "all_candidate_ids_ascending": all(scene["visible_ids"] == sorted(scene["visible_ids"]) for scene in frozen),
        "all_expected_orders_are_permutations": all(
            sorted(scene["expected_order"]) == scene["visible_ids"] for scene in frozen
        ),
        "all_images_match": all(sha256(EXPERIMENT / scene["image"]) == scene["image_sha256"] for scene in frozen),
        "d0_centroid_and_bbox_midpoint_orders_agree": center_definition_orders_agree,
        "schemas_have_no_array_or_id_bounds": not any(
            token in json.dumps(SCHEMAS) for token in ("minimum", "maximum", "minItems", "maxItems")
        ),
        "batch_size": 1,
        "system_prompt_absent": config["system_prompt"] is None,
    }
    if not all(value for key, value in audit.items() if key.startswith("all_")):
        raise RuntimeError("setup validation failed")
    if not center_definition_orders_agree:
        raise RuntimeError("D0 horizontal center definitions produce different orders")
    write_json(EXPERIMENT / "audit/setup_audit.json", audit)
    print(json.dumps({"experiment": str(EXPERIMENT), **audit}, indent=2))


if __name__ == "__main__":
    main()
