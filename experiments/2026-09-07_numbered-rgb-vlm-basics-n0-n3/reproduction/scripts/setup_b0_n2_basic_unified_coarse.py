#!/usr/bin/env python3
"""Freeze the unified-prompt N2-Basic coarse-color experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
SOURCE = ROOT / "experiments/b0_n2_explicit_constraint_scaffold_60scenes_20260828"
EXPERIMENT = ROOT / "experiments/b0_n2_basic_unified_coarse_60scenes_20260903"

BASIC_COLORS = (
    "black",
    "blue",
    "brown",
    "green",
    "orange",
    "pink",
    "purple",
    "white",
    "yellow",
)

# Coarse, rendered-appearance labels. Fine shade words are deliberately merged.
BASIC_ASSET_COLOR: dict[str, str] = {
    "acafela": "blue",
    "coldgrape": "purple",
    "top": "green",
    "cocopalm": "pink",
    "minutemad": "yellow",
    "letsbe": "blue",
    "cantata": "brown",
    "biracsikhye": "yellow",
    "Cup_2": "white",
    "Cup_4": "blue",
    "cup_7": "purple",
    "cup_8": "purple",
    "Mug_2": "black",
    "Mug_4": "yellow",
    "mug_7": "orange",
}

SYSTEM_PROMPT = """You are a careful visual target-grounding evaluator.
Use only the provided numbered RGB image, confirmed candidate IDs, and user request.
Follow the requested output format exactly.
Do not reveal hidden reasoning."""

USER_PROMPT_TEMPLATE = """Examine the numbered RGB image.

The following IDs are confirmed to be the complete and correct set of all visible object-marker IDs:
{visible_ids_json}

User request:
\"{target_description}\"

First, decompose the request into these optional constraints:

1. The required object type
2. The required basic color

If no object type is specified, set object_type to null.
If no color is specified, set color to null.
At least one constraint is specified.
Use an open-vocabulary description for object_type.

Use exactly one of these basic color names for every visible object:
black, blue, brown, green, orange, pink, purple, white, yellow

Normalize fine shades to their basic color family:

- dark blue and navy -> blue
- dark purple and violet -> purple
- beige, tan, light brown, and dark brown -> brown
- dark green and teal -> green
- gold and golden yellow -> yellow

For multicolored packaging, choose the single basic color a person would most naturally use
to distinguish the object. Base it on the largest visible region of the main body or main label.
Ignore the cap or lid, handle, logo, printed text, small graphics, narrow accent colors,
the red ID marker, highlights, and shadows.

Process every confirmed ID exactly once.

For each ID:

1. Identify the physical object associated with the marker.
2. Assign exactly one basic color from the list above.
3. If an object type is required, set object_type_matches to whether it matches;
   otherwise set object_type_matches to null.
4. If a color is required, set color_matches to whether the basic color matches;
   otherwise set color_matches to null.
5. Set matches_request using every specified constraint.

Apply these rules exactly:

- Object type only: matches_request = object_type_matches
- Color only: matches_request = color_matches
- Object type and color: matches_request = object_type_matches AND color_matches

The matching_ids array must contain all and only the IDs whose matches_request value is true.

Before returning the response, verify:

1. Every confirmed ID appears exactly once in instances.
2. A comparison field is null if and only if its corresponding constraint is null.
3. Every matches_request follows the appropriate logical rule.
4. matching_ids contains all and only IDs with matches_request=true.
5. No ID outside the confirmed set has been added.

Return only one valid JSON object using this structure:

{{
  \"request_constraints\": {{
    \"object_type\": string or null,
    \"color\": string or null
  }},
  \"instances\": [
    {{
      \"id\": integer,
      \"object_type\": string,
      \"color\": string,
      \"object_type_matches\": boolean or null,
      \"color_matches\": boolean or null,
      \"matches_request\": boolean
    }}
  ],
  \"matching_ids\": [integer, ...]
}}

Do not output explanations, Markdown, confidence scores, or additional fields."""

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_constraints": {
            "type": "object",
            "properties": {
                "object_type": {"type": ["string", "null"]},
                "color": {"anyOf": [{"enum": list(BASIC_COLORS)}, {"type": "null"}]},
            },
            "required": ["object_type", "color"],
            "additionalProperties": False,
        },
        "instances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "object_type": {"type": "string"},
                    "color": {"enum": list(BASIC_COLORS)},
                    "object_type_matches": {"type": ["boolean", "null"]},
                    "color_matches": {"type": ["boolean", "null"]},
                    "matches_request": {"type": "boolean"},
                },
                "required": [
                    "id",
                    "object_type",
                    "color",
                    "object_type_matches",
                    "color_matches",
                    "matches_request",
                ],
                "additionalProperties": False,
            },
        },
        "matching_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["request_constraints", "instances", "matching_ids"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def choose_balanced(
    candidates: list[str], usage: Counter[str], scene_id: str, query_id: str
) -> str:
    if not candidates:
        raise ValueError(f"no candidates for {scene_id} {query_id}")
    return min(
        sorted(set(candidates)),
        key=lambda value: (usage[value], stable_rank(scene_id, query_id, value)),
    )


def main() -> None:
    if EXPERIMENT.exists() and any(EXPERIMENT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {EXPERIMENT}")
    for name in ("config", "prompts", "logs", "results", "reports", "audit"):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)

    source_scenes = json.loads(
        (SOURCE / "config/frozen_scenes.json").read_text(encoding="utf-8")
    )
    if len(source_scenes) != 60:
        raise ValueError(f"expected 60 scenes, found {len(source_scenes)}")

    q2_color_usage: Counter[str] = Counter()
    q3_color_usage: Counter[str] = Counter()
    frozen: list[dict[str, Any]] = []

    for source_scene in source_scenes:
        scene_id = str(source_scene["scene_id"])
        source_q2 = next(
            query
            for query in source_scene["queries"]
            if query["query_id"] == "Q2_CATEGORY_COLOR_PRESENT"
        )
        source_target_asset = str(source_q2["target_asset"])
        if source_target_asset not in BASIC_ASSET_COLOR:
            raise ValueError(f"unmapped target asset {source_target_asset}: {scene_id}")

        instances: list[dict[str, Any]] = []
        for item in source_scene["instances"]:
            asset = str(item["asset_id"])
            if asset not in BASIC_ASSET_COLOR:
                raise ValueError(f"unmapped asset {asset}: {scene_id}")
            instances.append({**item, "n2_basic_color": BASIC_ASSET_COLOR[asset]})

        has_mug7 = any(item["asset_id"] == "mug_7" for item in instances)
        target_family = "bottle" if has_mug7 else "can"
        target_word = "bottle" if target_family == "bottle" else "can"
        visible_ids = sorted(int(item["id"]) for item in instances)
        family_items = [item for item in instances if item["object_family"] == target_family]
        if not family_items:
            raise ValueError(f"missing target family {target_family}: {scene_id}")

        q1_ids = sorted(int(item["id"]) for item in family_items)
        source_candidates = [
            item for item in family_items if item["asset_id"] == source_target_asset
        ]
        if not source_candidates:
            # Defensive fallback: choose a present family color deterministically.
            q2_color = choose_balanced(
                [str(item["n2_basic_color"]) for item in family_items],
                q2_color_usage,
                scene_id,
                "Q2",
            )
        else:
            q2_color = str(source_candidates[0]["n2_basic_color"])
        q2_color_usage[q2_color] += 1
        q2_ids = sorted(
            int(item["id"])
            for item in family_items
            if item["n2_basic_color"] == q2_color
        )
        q1_5_ids = sorted(
            int(item["id"])
            for item in instances
            if item["n2_basic_color"] == q2_color
        )

        target_family_colors = {
            str(item["n2_basic_color"]) for item in family_items
        }
        q3_by_color: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in instances:
            if item["object_family"] == target_family:
                continue
            if item["asset_id"] == "mug_7":
                continue
            color = str(item["n2_basic_color"])
            if color not in target_family_colors:
                q3_by_color[color].append(item)
        used_mug7_fallback = False
        if not q3_by_color:
            for item in instances:
                if item["object_family"] == target_family:
                    continue
                color = str(item["n2_basic_color"])
                if color not in target_family_colors:
                    q3_by_color[color].append(item)
            used_mug7_fallback = True
        q3_color = choose_balanced(
            list(q3_by_color), q3_color_usage, scene_id, "Q3"
        )
        q3_color_usage[q3_color] += 1
        q3_source_ids = sorted(int(item["id"]) for item in q3_by_color[q3_color])

        queries = [
            {
                "query_id": "Q1_CATEGORY_PRESENT",
                "query_type": "category_only_present",
                "target_description": f"Bring me a {target_word}.",
                "target_family": target_family,
                "target_color": None,
                "expected_ids": q1_ids,
                "hard_negative_source_ids": [],
            },
            {
                "query_id": "Q1_5_COLOR_ONLY_PRESENT",
                "query_type": "color_only_present",
                "target_description": f"Bring me something {q2_color}.",
                "target_family": None,
                "target_color": q2_color,
                "expected_ids": q1_5_ids,
                "hard_negative_source_ids": [],
            },
            {
                "query_id": "Q2_CATEGORY_COLOR_PRESENT",
                "query_type": "category_color_present",
                "target_description": f"Bring me the {q2_color} {target_word}.",
                "target_family": target_family,
                "target_color": q2_color,
                "target_asset": source_target_asset,
                "expected_ids": q2_ids,
                "hard_negative_source_ids": [],
            },
            {
                "query_id": "Q3_HARD_NEGATIVE_ABSENT",
                "query_type": "hard_negative_absent",
                "target_description": f"Bring me the {q3_color} {target_word}.",
                "target_family": target_family,
                "target_color": q3_color,
                "expected_ids": [],
                "hard_negative_source_ids": q3_source_ids,
                "uses_mug7_color_source_fallback": used_mug7_fallback,
            },
        ]

        source_image = SOURCE / source_scene["image"]
        target_image = EXPERIMENT / source_scene["image"]
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
        if sha256(target_image) != source_scene["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {scene_id}")

        frozen.append(
            {
                **source_scene,
                "has_mug7": has_mug7,
                "target_family": target_family,
                "visible_ids": visible_ids,
                "instances": instances,
                "queries": queries,
            }
        )

    seeds = [28101, 28102, 28103, 28104, 28105]
    config = {
        "experiment_id": EXPERIMENT.name,
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_path": str(ROOT / "models/Qwen3-VL-30B-A3B-Instruct"),
        "source_experiment": str(SOURCE),
        "scene_count": len(frozen),
        "query_count_per_scene": 4,
        "main_seeds": seeds,
        "scheduled_calls": len(frozen) * 4 * len(seeds),
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
        "input": "numbered RGB + confirmed visible-ID candidates + natural request",
        "output": "forced JSON unified optional constraints + coarse-color inventory + matching_ids",
        "primary_metric": "matching_id_set_exact_against_n2_basic_coarse_color",
        "design": {
            "Q1": "category-only present",
            "Q1.5": "coarse-color-only present",
            "Q2": "category plus present coarse color",
            "Q3": "category plus visible but compositionally absent coarse color",
            "same_prompt_template_for_all_queries": True,
            "repeats_per_scene_query": 5,
        },
        "basic_colors": list(BASIC_COLORS),
        "basic_asset_color": BASIC_ASSET_COLOR,
        "color_policy": {
            "dark_blue_and_navy": "blue",
            "dark_purple_and_violet": "purple",
            "beige_tan_light_or_dark_brown": "brown",
            "dark_green_and_teal": "green",
            "gold_and_golden_yellow": "yellow",
        },
    }
    write_json(EXPERIMENT / "config/experiment_config.json", config)
    write_json(EXPERIMENT / "config/frozen_scenes.json", frozen)
    write_json(EXPERIMENT / "config/json_schema.json", JSON_SCHEMA, sort_keys=False)
    write_json(
        EXPERIMENT / "config/query_manifest.json",
        [
            {
                "scene_id": scene["scene_id"],
                "level": scene["level"],
                "visible_ids": scene["visible_ids"],
                "queries": scene["queries"],
            }
            for scene in frozen
        ],
    )
    (EXPERIMENT / "prompts/system_prompt_en.txt").write_text(
        SYSTEM_PROMPT + "\n", encoding="utf-8"
    )
    (EXPERIMENT / "prompts/user_prompt_template_en.txt").write_text(
        USER_PROMPT_TEMPLATE + "\n", encoding="utf-8"
    )

    query_counts = Counter(
        query["query_id"] for scene in frozen for query in scene["queries"]
    )
    q3_valid = all(
        not scene["queries"][3]["expected_ids"]
        and scene["queries"][3]["hard_negative_source_ids"]
        and scene["queries"][3]["target_color"]
        not in {
            item["n2_basic_color"]
            for item in scene["instances"]
            if item["object_family"] == scene["target_family"]
        }
        for scene in frozen
    )
    audit = {
        "scene_count": len(frozen),
        "level_counts": dict(Counter(scene["level"] for scene in frozen)),
        "query_counts": dict(query_counts),
        "scheduled_calls": config["scheduled_calls"],
        "q2_color_scene_counts": dict(q2_color_usage),
        "q3_color_scene_counts": dict(q3_color_usage),
        "q1_5_target_cardinality_scene_counts": dict(
            Counter(len(scene["queries"][1]["expected_ids"]) for scene in frozen)
        ),
        "q2_target_cardinality_scene_counts": dict(
            Counter(len(scene["queries"][2]["expected_ids"]) for scene in frozen)
        ),
        "all_four_queries_present": all(len(scene["queries"]) == 4 for scene in frozen),
        "all_q1_5_and_q2_targets_present": all(
            scene["queries"][1]["expected_ids"] and scene["queries"][2]["expected_ids"]
            for scene in frozen
        ),
        "all_q3_hard_negatives_valid": q3_valid,
        "all_expected_targets_subset_candidates": all(
            set(query["expected_ids"]).issubset(scene["visible_ids"])
            for scene in frozen
            for query in scene["queries"]
        ),
        "all_images_match": all(
            sha256(EXPERIMENT / scene["image"]) == scene["image_sha256"]
            for scene in frozen
        ),
        "schema_has_no_id_or_array_bounds": not any(
            token in json.dumps(JSON_SCHEMA)
            for token in ("minimum", "maximum", "minItems", "maxItems")
        ),
        "same_prompt_template_for_all_queries": True,
        "batch_size": config["runtime"]["batch_size"],
        "q3_mug7_fallback_scenes": [
            scene["scene_id"]
            for scene in frozen
            if scene["queries"][3].get("uses_mug7_color_source_fallback")
        ],
    }
    if not all(value for key, value in audit.items() if key.startswith("all_")):
        raise RuntimeError(f"setup validation failed: {audit}")
    write_json(EXPERIMENT / "audit/setup_audit.json", audit)
    print(json.dumps({"experiment": str(EXPERIMENT), **audit}, indent=2))


if __name__ == "__main__":
    main()
