#!/usr/bin/env python3
"""Freeze the N1 explicit-visible-ID 3x2x2 ablation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
SOURCE = ROOT / "experiments/b0_n1_primary_color_prompt_ablation_60scenes_20260827"
EXPERIMENT = ROOT / "experiments/b0_n1_explicit_visible_ids_ablation_60scenes_20260827"

COLOR_INSTRUCTIONS = """For each ID, report the primary color of the object as a whole:
the color a person would most naturally use to distinguish that object.

Base the answer on the largest visible region of the main body or main label.
Ignore the cap or lid, handle, logo, printed text, small graphics,
narrow accent colors, the red ID marker, highlights, and shadows.

Do not choose the color of a small accessory or accent region unless
that color also dominates the main body or main label.
For a partially occluded object, use only its identifiable visible
main-body or main-label regions."""

BASE_PROMPT = f"""For every visible red circular marker ID in the numbered RGB image, identify the physical object associated with that marker.

{COLOR_INSTRUCTIONS}

Use a short singular English noun phrase for the object type.
Use a concise English description for the color.
Do not use a predefined object or color vocabulary.
Include each visible marker ID exactly once."""

ENUM_PROMPT = f"""Examine the numbered RGB image.

First, read every visible numeric ID inside a red circular marker and record all of those IDs in an explicit visible-ID list.

Then, for every ID in that visible-ID list, identify the physical object associated with that marker.

{COLOR_INSTRUCTIONS}

Use a short singular English noun phrase for the object type.
Use a concise English description for the color.
Identify the object only from its visible appearance.
Do not use a predefined object or color vocabulary.

Include every visible marker ID exactly once in the visible-ID list.
Return exactly one instance for every ID in that list.
The set of IDs in the instance mappings must exactly match the set in the visible-ID list.
Do not include an ID that is not visibly present in the image."""

TEXT_SUFFIX_BASE = """Return one instance per line using exactly this format:

<ID> | <object_type> | <color>

Do not include a header, Markdown, or any explanation."""

TEXT_SUFFIX_ENUM = """Return the visible-ID list first using exactly this format:

visible_ids: [<integer>, ...]

Then return exactly one instance per line using exactly this format:

<ID> | <object_type> | <color>

Do not include Markdown, an explanation, or any other text."""

JSON_SUFFIX_BASE = """Return only a JSON object with exactly one key named \"instances\".
Each element of \"instances\" must contain exactly:
- \"id\": the integer shown inside the red marker
- \"object_type\": the physical object type
- \"color\": the primary visible color of the object as a whole

Do not use Markdown and do not add any other keys or text."""

JSON_SUFFIX_ENUM = """Return only one valid JSON object.
The first key must be \"visible_ids\", containing the explicit visible-ID list.
The second key must be \"instances\", containing exactly one mapping for every ID in \"visible_ids\".
Each element of \"instances\" must contain exactly:
- \"id\": the integer shown inside the red marker
- \"object_type\": the physical object type
- \"color\": the primary visible color of the object as a whole

Do not use Markdown and do not add any other keys or text."""

BASE_SCHEMA = {
    "type": "object",
    "properties": {
        "instances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "object_type": {"type": "string"},
                    "color": {"type": "string"},
                },
                "required": ["id", "object_type", "color"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["instances"],
    "additionalProperties": False,
}

ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "visible_ids": {"type": "array", "items": {"type": "integer"}},
        "instances": BASE_SCHEMA["properties"]["instances"],
    },
    "required": ["visible_ids", "instances"],
    "additionalProperties": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_ordered_json(path: Path, value: Any) -> None:
    """Write schemas without sorting because property order is an experimental manipulation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if EXPERIMENT.exists() and any(EXPERIMENT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {EXPERIMENT}")
    for name in ("config", "prompts", "logs", "results", "reports", "figures", "audit"):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)

    scenes = read_json(SOURCE / "config/frozen_scenes.json")
    for scene in scenes:
        source_image = SOURCE / scene["image"]
        target_image = EXPERIMENT / scene["image"]
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
        if sha256(target_image) != scene["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {scene['scene_id']}")

    source_config = read_json(SOURCE / "config/experiment_config.json")
    conditions: list[dict[str, Any]] = []
    for temperature in (0.3, 0.5, 0.7):
        for enumeration in (False, True):
            for output_format in ("TEXT", "JSON_FORCED"):
                conditions.append(
                    {
                        "condition_id": (
                            f"T{int(temperature * 10):02d}_"
                            f"{'ENUM' if enumeration else 'DIRECT'}_"
                            f"{'JSON' if output_format == 'JSON_FORCED' else 'TEXT'}"
                        ),
                        "temperature": temperature,
                        "enumeration": enumeration,
                        "output_format": output_format,
                    }
                )

    config = {
        **source_config,
        "experiment_id": EXPERIMENT.name,
        "conditions": conditions,
        "generation": {**source_config["generation"], "temperature": "condition-specific"},
        "scheduled_main_calls": len(scenes) * len(source_config["main_seeds"]) * len(conditions),
        "primary_metric": "full_output_joint_exact",
        "prompt_ablation": "explicit visible-ID enumeration before ID-to-attribute mapping",
        "design": {
            "temperature": [0.3, 0.5, 0.7],
            "explicit_visible_id_enumeration": [False, True],
            "output_format": ["TEXT", "JSON_FORCED"],
            "repeats_per_scene_condition": 5,
            "paired_by": ["scene_id", "seed"],
        },
        "held_constant": [
            "60 frozen numbered-RGB scenes",
            "five seeds 28101-28105",
            "Qwen3-VL-30B-A3B-Instruct",
            "batch size 1",
            "top_p 0.9",
            "top_k 0",
            "max_tokens 1024",
            "same whole-object primary-color rule",
            "same hidden open-vocabulary scoring references",
        ],
        "json_constraint": "factor-specific strict vLLM structured output; no array length, ID, object, or color bounds",
    }
    write_json(EXPERIMENT / "config/experiment_config.json", config)
    write_json(EXPERIMENT / "config/frozen_scenes.json", scenes)
    write_ordered_json(EXPERIMENT / "config/json_schema_direct.json", BASE_SCHEMA)
    write_ordered_json(EXPERIMENT / "config/json_schema_enumeration.json", ENUM_SCHEMA)
    for relative in ("config/hidden_asset_references.json", "prompts/system_prompt_en.txt"):
        destination = EXPERIMENT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, destination)

    prompts = {
        "direct_common_prompt_en.txt": BASE_PROMPT,
        "enumeration_common_prompt_en.txt": ENUM_PROMPT,
        "text_suffix_direct_en.txt": TEXT_SUFFIX_BASE,
        "text_suffix_enumeration_en.txt": TEXT_SUFFIX_ENUM,
        "json_suffix_direct_en.txt": JSON_SUFFIX_BASE,
        "json_suffix_enumeration_en.txt": JSON_SUFFIX_ENUM,
    }
    for filename, prompt in prompts.items():
        (EXPERIMENT / "prompts" / filename).write_text(prompt + "\n", encoding="utf-8")

    write_json(
        EXPERIMENT / "audit/setup_audit.json",
        {
            "scene_count": len(scenes),
            "condition_count": len(conditions),
            "scheduled_calls": config["scheduled_main_calls"],
            "batch_size": config["runtime"]["batch_size"],
            "image_hashes_match_source": all(
                sha256(EXPERIMENT / row["image"]) == row["image_sha256"] for row in scenes
            ),
            "source_experiment": str(SOURCE),
            "conditions": conditions,
            "prompt_hashes": {name: sha256(EXPERIMENT / "prompts" / name) for name in prompts},
            "schema_hashes": {
                "direct": sha256(EXPERIMENT / "config/json_schema_direct.json"),
                "enumeration": sha256(EXPERIMENT / "config/json_schema_enumeration.json"),
            },
        },
    )
    print(json.dumps({"experiment": str(EXPERIMENT), "scenes": len(scenes), "conditions": len(conditions), "calls": config["scheduled_main_calls"]}, indent=2))


if __name__ == "__main__":
    main()
