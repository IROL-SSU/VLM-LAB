#!/usr/bin/env python3
"""Build the 60-scene B0-N0 sparse two-digit ID TEXT-vs-JSON experiment."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/ssu/ShelfScene")
EXPERIMENT = ROOT / "experiments/b0_n0_sparse_ids_text_vs_json_60scenes_20260826"
MODEL_PATH = ROOT / "models/Qwen3-VL-30B-A3B-Instruct"
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
GLOBAL_ID_SEED = 20260826
SOURCES = (
    ("L3", ROOT / "benchmark_b0/l3_random_xy_20scenes"),
    (
        "L4",
        ROOT / "benchmark_b0/l4_random_20scenes_pixel_calibrated_30_45_45_60",
    ),
    (
        "L5",
        ROOT / "benchmark_b0/l5_random_20scenes_two_occlusions_pixel_calibrated_30_45_45_60",
    ),
)

SYSTEM_PROMPT = """You are a careful visual marker-reading evaluator.
Use only the provided RGB image and follow the requested output format exactly.
Do not reveal hidden reasoning."""

COMMON_PROMPT = (
    "List all integer IDs visible inside the red circular markers in the image."
)

TEXT_SUFFIX = """Return only the integer IDs as a comma-separated line.
Do not use brackets, labels, words, or explanations."""

JSON_SUFFIX = """Return only a JSON object with exactly one key named \"visible_ids\".
The value must be an array of integers.
Do not use Markdown and do not add any other keys or text."""

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible_ids": {
            "type": "array",
            "items": {"type": "integer"},
        }
    },
    "required": ["visible_ids"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(dataset_name: str, group_key: str, count: int) -> int:
    material = f"{GLOBAL_ID_SEED}|sparse-v1|{dataset_name}|{group_key}|{count}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def spearman_identity(values: list[int]) -> float:
    positions = np.arange(len(values), dtype=np.float64)
    ranks = np.argsort(np.argsort(np.asarray(values, dtype=np.int64))).astype(np.float64)
    return float(np.corrcoef(positions, ranks)[0, 1])


def make_sparse_ids(dataset_name: str, group_key: str, count: int) -> tuple[list[int], int, float]:
    seed = stable_seed(dataset_name, group_key, count)
    rng = random.Random(seed)
    pool = list(range(10, 100))
    for _ in range(10000):
        rng.shuffle(pool)
        candidate = pool[:count]
        # Avoid near-contiguous sets that can be reconstructed as a range.
        if any(abs(a - b) <= 1 for i, a in enumerate(candidate) for b in candidate[i + 1 :]):
            continue
        correlation = spearman_identity(candidate)
        if abs(correlation) <= 0.45:
            return candidate.copy(), seed, correlation
    raise RuntimeError(f"could not generate sparse IDs for {group_key}")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def draw_sparse_badges(
    source_image: Path,
    instances: list[dict[str, Any]],
    ids_left_to_right: list[int],
    destination: Path,
) -> dict[str, int]:
    image = Image.open(source_image).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    max_text_width = 0
    max_text_height = 0
    for instance, display_id in zip(instances, ids_left_to_right):
        x, y = [int(value) for value in instance["badge_center_xy"]]
        radius = int(instance["badge_radius_px"])
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(220, 38, 38, 238),
            outline=(255, 255, 255, 255),
            width=2,
        )
        # Two bold digits fit inside the existing 32-pixel diameter badge.
        font = load_font(max(15, int(round(radius * 1.05))))
        text = str(display_id)
        box = draw.textbbox((0, 0), text, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
        max_text_width = max(max_text_width, width)
        max_text_height = max(max_text_height, height)
        if width > radius * 2 - 4 or height > radius * 2 - 4:
            raise ValueError(f"ID {display_id} does not fit radius {radius}: {width}x{height}")
        draw.text(
            (x - width / 2 - box[0], y - height / 2 - box[1]),
            text,
            font=font,
            fill=(255, 255, 255, 255),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(destination)
    return {"max_text_width_px": max_text_width, "max_text_height_px": max_text_height}


def factors_for(level: str, sample: dict[str, Any], count: int) -> dict[str, Any]:
    factors: dict[str, Any] = {"object_count": count}
    if level in {"L4", "L5"}:
        factors.update(
            {
                "occlusion_band": sample.get("occlusion_band"),
                "base_index": sample.get("base_index"),
                "matched_group_id": sample.get("matched_group_id"),
            }
        )
    if level == "L5":
        factors["occlusion_pair_count"] = sample.get("occlusion_pair_count")
    return factors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if EXPERIMENT.exists() and any(EXPERIMENT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {EXPERIMENT}")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)
    for name in ("config", "prompts", "logs", "results", "reports", "figures", "audit"):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)

    scenes: list[dict[str, Any]] = []
    assignments: dict[tuple[str, str], tuple[list[int], int, float]] = {}
    source_hashes: dict[str, str] = {}
    max_text_width = 0
    max_text_height = 0

    for level, dataset in SOURCES:
        source_metadata_path = dataset / "metadata.json"
        badge_metadata_path = dataset / "numbered_rgb_metadata.json"
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        badge_metadata = json.loads(badge_metadata_path.read_text(encoding="utf-8"))
        badge_by_scene = {row["scene_id"]: row for row in badge_metadata["scenes"]}
        source_hashes[f"{level}_source_metadata"] = sha256(source_metadata_path)
        source_hashes[f"{level}_badge_metadata"] = sha256(badge_metadata_path)

        for sample in source_metadata["samples"]:
            scene_id = sample["scene_id"]
            badge_row = badge_by_scene[scene_id]
            instances = sorted(badge_row["instances"], key=lambda row: row["number"])
            count = len(instances)
            group_key = sample.get("matched_group_id") or scene_id
            assignment_key = (dataset.name, group_key)
            if assignment_key not in assignments:
                assignments[assignment_key] = make_sparse_ids(dataset.name, group_key, count)
            ids_left_to_right, assignment_seed, correlation = assignments[assignment_key]
            if len(ids_left_to_right) != count:
                raise ValueError(f"matched group count changed: {group_key}")

            source_image = dataset / sample["image"]
            target_image = (
                EXPERIMENT
                / "scenes/images"
                / level
                / f"{scene_id}_numbered_rgb_sparse_ids.png"
            )
            text_stats = draw_sparse_badges(
                source_image, instances, ids_left_to_right, target_image
            )
            max_text_width = max(max_text_width, text_stats["max_text_width_px"])
            max_text_height = max(max_text_height, text_stats["max_text_height_px"])
            image_hash = sha256(target_image)
            randomized_instances = [
                {
                    "display_id": display_id,
                    "left_to_right_position": instance["number"],
                    "instance_id": instance["instance_id"],
                    "asset_id": instance["asset_id"],
                    "category": instance["category"],
                    "badge_center_xy": instance["badge_center_xy"],
                    "badge_radius_px": instance["badge_radius_px"],
                }
                for instance, display_id in zip(instances, ids_left_to_right)
            ]
            scenes.append(
                {
                    "scene_id": scene_id,
                    "level": level,
                    "image": str(target_image.relative_to(EXPERIMENT)),
                    "image_sha256": image_hash,
                    "ground_truth": {
                        "visible_ids_sorted": sorted(ids_left_to_right),
                        "ids_left_to_right": ids_left_to_right,
                    },
                    "factors": factors_for(level, sample, count),
                    "id_assignment": {
                        "group_key": group_key,
                        "seed": assignment_seed,
                        "position_rank_correlation": round(correlation, 6),
                    },
                    "instances_left_to_right": randomized_instances,
                    "source_image": str(source_image.relative_to(ROOT)),
                }
            )

    level_counts = Counter(row["level"] for row in scenes)
    if len(scenes) != 60 or level_counts != Counter({"L3": 20, "L4": 20, "L5": 20}):
        raise ValueError(f"unexpected scene counts: {level_counts}")

    seeds = [28101, 28102, 28103, 28104, 28105]
    config = {
        "experiment_id": EXPERIMENT.name,
        "model_id": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "scene_count": len(scenes),
        "level_counts": dict(level_counts),
        "modes": ["TEXT", "JSON_FORCED"],
        "main_seeds": seeds,
        "preflight_seeds": [],
        "preflight_scene_ids": [],
        "scheduled_main_calls": len(scenes) * len(seeds) * 2,
        "scheduled_preflight_calls": 0,
        "generation": {
            "temperature": 0.7,
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
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.90,
            "moe_backend": "triton",
        },
        "prompt_language": "English",
        "id_policy": (
            "deterministic scene-specific unique two-digit IDs in 10..99; "
            "no pair differs by <=1; scenes sharing an explicit matched_group_id share IDs"
        ),
        "ground_truth_order": "none; score visible red-marker IDs as a set",
        "primary_metric": "id_set_exact",
        "json_constraint": "strict vLLM structured output; no array length or ID bounds",
    }
    write_json(EXPERIMENT / "config/experiment_config.json", config)
    write_json(EXPERIMENT / "config/frozen_scenes.json", scenes)
    write_json(EXPERIMENT / "config/json_schema.json", JSON_SCHEMA)
    (EXPERIMENT / "prompts/system_prompt_en.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (EXPERIMENT / "prompts/common_prompt_en.txt").write_text(COMMON_PROMPT + "\n", encoding="utf-8")
    (EXPERIMENT / "prompts/text_suffix_en.txt").write_text(TEXT_SUFFIX + "\n", encoding="utf-8")
    (EXPERIMENT / "prompts/json_suffix_en.txt").write_text(JSON_SUFFIX + "\n", encoding="utf-8")
    write_json(
        EXPERIMENT / "audit/setup_audit.json",
        {
            "scene_count": len(scenes),
            "unique_scene_ids": len({row["scene_id"] for row in scenes}),
            "unique_image_hashes": len({row["image_sha256"] for row in scenes}),
            "level_counts": dict(level_counts),
            "assignment_group_count": len(assignments),
            "all_ids_two_digit": all(
                10 <= value <= 99
                for row in scenes
                for value in row["ground_truth"]["visible_ids_sorted"]
            ),
            "all_scene_ids_unique": all(
                len(values) == len(set(values))
                for values in (row["ground_truth"]["visible_ids_sorted"] for row in scenes)
            ),
            "all_scene_ids_sparse": all(
                all(abs(a - b) > 1 for i, a in enumerate(values) for b in values[i + 1 :])
                for values in (row["ground_truth"]["visible_ids_sorted"] for row in scenes)
            ),
            "max_rendered_text_width_px": max_text_width,
            "max_rendered_text_height_px": max_text_height,
            "badge_diameter_px": 32,
            "source_metadata_sha256": source_hashes,
            "scheduled_calls": config["scheduled_main_calls"],
            "batch_size": 1,
        },
    )
    print(json.dumps({"experiment": str(EXPERIMENT), "scenes": 60, "calls": 600}, indent=2))


if __name__ == "__main__":
    main()
