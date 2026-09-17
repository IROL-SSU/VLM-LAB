#!/usr/bin/env python3
"""Freeze the explicit request-constraint scaffold N2 experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
SOURCE = ROOT / "experiments/b0_n2_numbered_rgb_semantic_scaffold_60scenes_20260827"
EXPERIMENT = ROOT / "experiments/b0_n2_explicit_constraint_scaffold_60scenes_20260828"

SYSTEM_PROMPT = """You are a careful visual target-grounding evaluator.
Use only the provided numbered RGB image, confirmed candidate IDs, and user request.
Follow the requested output format exactly.
Do not reveal hidden reasoning."""

USER_PROMPT_TEMPLATE = """Examine the numbered RGB image.

The following IDs are confirmed to be the complete and correct set of all visible object-marker IDs:
{visible_ids_json}

User request:
\"{target_description}\"

First, decompose the user request into the following two constraints:

1. The required object type
2. The required color

If the user does not specify a color, set the required color to null.
Use open-vocabulary descriptions. Do not use a predefined object or color vocabulary.

Process every confirmed ID exactly once.

For each ID:

1. Identify the physical object associated with the numbered marker.
2. Identify the primary visible color of the object as a whole.
3. Determine whether the identified object type matches the required object type.
4. If the user specified a color, determine whether the identified color matches the required color.
5. Compute whether the object satisfies the entire user request.

For the primary color, use the color a person would most naturally use to distinguish the object.
Base the decision on the largest visible region of the main body or main label.
Ignore the cap or lid, handle, logo, printed text, small graphics, narrow accent colors,
the red ID marker, highlights, and shadows.

Apply the following logical rule exactly.

If a color is required:

matches_request = object_type_matches AND color_matches

If no color is required:

matches_request = object_type_matches

When both an object type and a color are required, follow this truth table:

object_type_matches | color_matches | matches_request
true                | true          | true
true                | false         | false
false               | true          | false
false               | false         | false

Matching only the required object type is not sufficient.
Matching only the required color is not sufficient.

The matching_ids array must contain exactly the IDs whose matches_request value is true.

Before returning the response, verify all of the following:

1. Every confirmed ID appears exactly once in instances.
2. Every matches_request value follows the logical rule above.
3. matching_ids contains all and only the IDs with matches_request=true.
4. No ID outside the confirmed ID set has been added.

Return only one valid JSON object using the following structure:

{{
  \"request_constraints\": {{
    \"object_type\": string,
    \"color\": string or null
  }},
  \"instances\": [
    {{
      \"id\": integer,
      \"object_type\": string,
      \"color\": string,
      \"object_type_matches\": boolean,
      \"color_matches\": boolean or null,
      \"matches_request\": boolean
    }}
  ],
  \"matching_ids\": [integer, ...]
}}

If the user does not specify a color, return null for color_matches.

Do not output explanations, Markdown, or additional fields."""

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_constraints": {
            "type": "object",
            "properties": {
                "object_type": {"type": "string"},
                "color": {"type": ["string", "null"]},
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
                    "color": {"type": "string"},
                    "object_type_matches": {"type": "boolean"},
                    "color_matches": {"type": ["boolean", "null"]},
                    "matches_request": {"type": "boolean"},
                },
                "required": [
                    "id", "object_type", "color", "object_type_matches",
                    "color_matches", "matches_request",
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


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if EXPERIMENT.exists() and any(EXPERIMENT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {EXPERIMENT}")
    for name in ("config", "prompts", "logs", "results", "reports", "audit"):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)

    scenes = json.loads((SOURCE / "config/frozen_scenes.json").read_text(encoding="utf-8"))
    if len(scenes) != 60:
        raise ValueError(f"expected 60 scenes, found {len(scenes)}")
    for scene in scenes:
        source_image = SOURCE / scene["image"]
        target_image = EXPERIMENT / scene["image"]
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
        if sha256(target_image) != scene["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {scene['scene_id']}")

    seeds = [28101, 28102, 28103, 28104, 28105]
    config = {
        "experiment_id": EXPERIMENT.name,
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_path": str(ROOT / "models/Qwen3-VL-30B-A3B-Instruct"),
        "source_experiment": str(SOURCE),
        "scene_count": len(scenes),
        "query_count_per_scene": 3,
        "main_seeds": seeds,
        "scheduled_calls": len(scenes) * 3 * len(seeds),
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
        "input": "numbered RGB + GT visible-ID candidates + natural user request",
        "output": "forced JSON request decomposition + explicit per-condition booleans + matching_ids",
        "primary_metric": "matching_id_set_exact",
        "paired_baseline": str(SOURCE),
    }
    write_json(EXPERIMENT / "config/experiment_config.json", config)
    write_json(EXPERIMENT / "config/frozen_scenes.json", scenes)
    write_json(EXPERIMENT / "config/json_schema.json", JSON_SCHEMA, sort_keys=False)
    (EXPERIMENT / "prompts/system_prompt_en.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (EXPERIMENT / "prompts/user_prompt_template_en.txt").write_text(USER_PROMPT_TEMPLATE + "\n", encoding="utf-8")

    query_counts = Counter(q["query_id"] for scene in scenes for q in scene["queries"])
    audit = {
        "scene_count": len(scenes),
        "level_counts": dict(Counter(scene["level"] for scene in scenes)),
        "query_counts": dict(query_counts),
        "scheduled_calls": config["scheduled_calls"],
        "all_natural_requests": all(
            q["target_description"].startswith("Bring me ")
            for scene in scenes for q in scene["queries"]
        ),
        "all_candidate_ids_unique": all(
            len(scene["visible_ids"]) == len(set(scene["visible_ids"])) for scene in scenes
        ),
        "all_expected_targets_subset_candidates": all(
            set(q["expected_ids"]).issubset(scene["visible_ids"])
            for scene in scenes for q in scene["queries"]
        ),
        "all_images_match": all(
            sha256(EXPERIMENT / scene["image"]) == scene["image_sha256"] for scene in scenes
        ),
        "schema_has_no_id_or_array_bounds": not any(
            token in json.dumps(JSON_SCHEMA) for token in ("minimum", "maximum", "minItems", "maxItems")
        ),
        "batch_size": 1,
        "paired_input_manifest_identical": sha256(SOURCE / "config/frozen_scenes.json")
        == sha256(EXPERIMENT / "config/frozen_scenes.json"),
    }
    if not all(value for key, value in audit.items() if key.startswith("all_")):
        raise RuntimeError("setup validation failed")
    write_json(EXPERIMENT / "audit/setup_audit.json", audit)
    print(json.dumps({"experiment": str(EXPERIMENT), **audit}, indent=2))


if __name__ == "__main__":
    main()
