#!/usr/bin/env python3
"""Freeze the missing 20 RGB-only structured-occlusion scenes for five prompts."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("/home/ssu/ShelfScene")
SOURCE = ROOT / "experiments/n3_rgbd_3scene_40each_5methods_5seeds_20260831"
OUTPUT = ROOT / "experiments/n3_occlusion_rgb_missing20_5methods_5seeds_20260907"
SEEDS = [28101, 28102, 28103, 28104, 28105]
METHODS = [
    "minimal_direct",
    "detailed_direct",
    "contact_point",
    "explicit_pairs",
    "single_target",
]
BASELINES = {
    method: ROOT / "experiments" / f"n3_occlusion_{method}_20scenes_5seeds_20260831"
    for method in METHODS
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {OUTPUT}")
    for relative in (
        "config",
        "prompts",
        "scenes/numbered_rgb",
        "logs",
        "results",
        "reports",
        "audit",
    ):
        (OUTPUT / relative).mkdir(parents=True, exist_ok=True)

    all_source_rows = [
        row
        for row in read_json(SOURCE / "config/frozen_scenes.json")
        if row["scene_type"] == "structured_occlusion"
    ]
    if len(all_source_rows) != 40:
        raise ValueError(f"expected 40 structured-occlusion rows, found {len(all_source_rows)}")

    original_rows = read_json(
        BASELINES["minimal_direct"] / "config/frozen_scenes.json"
    )
    original_ids = {row["scene_id"] for row in original_rows}
    if len(original_ids) != 20:
        raise ValueError(f"expected 20 original RGB scene IDs, found {len(original_ids)}")
    selected = sorted(
        (row for row in all_source_rows if row["scene_id"] not in original_ids),
        key=lambda row: row["scene_id"],
    )
    if len(selected) != 20:
        raise ValueError(f"expected 20 missing scene rows, found {len(selected)}")

    frozen = []
    for source in selected:
        source_image = SOURCE / source["rgb_image"]
        target_rel = Path("scenes/numbered_rgb") / source_image.name
        target_image = OUTPUT / target_rel
        shutil.copy2(source_image, target_image)
        image_hash = sha256(target_image)
        if image_hash != source["rgb_sha256"]:
            raise RuntimeError(f"image hash mismatch for {source['scene_id']}")
        frozen.append(
            {
                "scene_uid": source["scene_uid"],
                "scene_type": source["scene_type"],
                "scene_id": source["scene_id"],
                "condition_id": source["condition_id"],
                "benchmark_split": source["benchmark_split"],
                "rgb_image": str(target_rel),
                "rgb_sha256": image_hash,
                "visible_ids": source["visible_ids"],
                "expected_order": source["expected_order"],
                "target_id": source["target_id"],
                "target_rank_front_zero_based": source[
                    "target_rank_front_zero_based"
                ],
                "factors": source["factors"],
            }
        )

    source_config = read_json(SOURCE / "config/experiment_config.json")
    config = {
        "experiment_id": OUTPUT.name,
        "model_id": source_config["model_id"],
        "model_path": source_config["model_path"],
        "source_rgbd_experiment": str(SOURCE),
        "original_rgb_baselines": {
            method: str(path) for method, path in BASELINES.items()
        },
        "input": "one numbered RGB image",
        "input_condition": "numbered_rgb",
        "scene_count": len(frozen),
        "scene_count_per_type": len(frozen),
        "scene_types": ["structured_occlusion"],
        "scene_labels": {"structured_occlusion": "Structured occlusion"},
        "methods": METHODS,
        "main_seeds": SEEDS,
        "scheduled_calls": len(frozen) * len(METHODS) * len(SEEDS),
        "generation": source_config["generation"],
        "runtime": {
            **source_config["runtime"],
            "images_per_prompt": 1,
            "limit_mm_per_prompt": {"image": 1},
        },
        "prompt_language": "English",
        "system_prompt": None,
        "output": "forced JSON",
        "skip_external_baseline_comparison": True,
        "supplement_for": "complete 40-scene RGB/Gray/Numbered-Gray comparison",
    }
    write_json(OUTPUT / "config/experiment_config.json", config)
    write_json(OUTPUT / "config/frozen_scenes.json", frozen)

    prompt_hash_matches = {}
    schema_hash_matches = {}
    for method, baseline in BASELINES.items():
        source_prompt = baseline / "prompts/d1_front_to_back_en.txt"
        source_schema = baseline / "config/json_schema_d1.json"
        target_prompt = OUTPUT / "prompts" / f"{method}_en.txt"
        target_schema = OUTPUT / "config" / f"json_schema_{method}.json"
        shutil.copy2(source_prompt, target_prompt)
        shutil.copy2(source_schema, target_schema)
        prompt_hash_matches[method] = sha256(source_prompt) == sha256(target_prompt)
        schema_hash_matches[method] = sha256(source_schema) == sha256(target_schema)

    expected_missing_structure_counts = {
        "o3_single_pair_extra_front": 4,
        "o3_three_object_chain": 6,
        "o4_four_object_chain": 2,
        "o4_three_chain_extra_front": 4,
        "o4_two_disjoint_pairs": 4,
    }
    missing_counts = Counter(
        row["factors"]["occlusion_structure"] for row in frozen
    )
    audit = {
        "source_structured_scene_count": len(all_source_rows),
        "original_rgb_scene_count": len(original_ids),
        "missing_rgb_scene_count": len(frozen),
        "scheduled_calls": config["scheduled_calls"],
        "method_count": len(METHODS),
        "seed_count": len(SEEDS),
        "original_and_missing_disjoint": not original_ids
        & {row["scene_id"] for row in frozen},
        "original_plus_missing_is_full_40": len(
            original_ids | {row["scene_id"] for row in frozen}
        )
        == 40,
        "missing_structure_counts": dict(sorted(missing_counts.items())),
        "missing_structure_counts_expected": dict(missing_counts)
        == expected_missing_structure_counts,
        "all_images_match_frozen_hash": all(
            sha256(OUTPUT / row["rgb_image"]) == row["rgb_sha256"]
            for row in frozen
        ),
        "all_targets_are_valid": all(
            row["target_id"] in row["visible_ids"] for row in frozen
        ),
        "all_gt_orders_are_permutations": all(
            sorted(row["expected_order"]) == sorted(row["visible_ids"])
            for row in frozen
        ),
        "prompt_hash_matches_original_rgb_baseline": prompt_hash_matches,
        "schema_hash_matches_original_rgb_baseline": schema_hash_matches,
        "batch_size": config["runtime"]["batch_size"],
        "images_per_prompt": config["runtime"]["images_per_prompt"],
        "system_prompt_absent": config["system_prompt"] is None,
    }
    write_json(OUTPUT / "audit/setup_audit.json", audit)
    print(json.dumps({"experiment": str(OUTPUT), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
