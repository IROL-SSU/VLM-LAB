#!/usr/bin/env python3
"""Freeze the v25 inputs while asking L4 for the action label only."""

from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l4_blocker_conditioned_v25"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l4_action_only_v26"
VERSION = "l4_action_only_from_l3_v19_v26"

PROMPT = """The perception stage has already selected one blocker for this exact
scene, geometry condition, and seed:

- blocking_reason: {selected_blocking_reason}
- selected blocker_id: {selected_blocker_id}

Choose exactly ONE next action: RETRIEVE, TRANSLATE, ROTATE, or
LIFT_AND_RELOCATE.

If blocking_reason is NONE and blocker_id is null, choose RETRIEVE for the
target. Otherwise, decide which ONE action on the selected blocker would make
the target directly graspable without contacting another object.

Output only a JSON object with the single key action. Do not output an object
ID, direction, distance, rotation angle, destination, explanation, or sequence."""

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["RETRIEVE", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"],
        }
    },
    "required": ["action"],
    "additionalProperties": False,
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify() -> dict:
    config = read_json(DEST / "config/experiment_config.json")
    audit = read_json(DEST / "audit/preflight_report.json")
    schedule = read_jsonl(DEST / "config/run_table.jsonl")
    assert config["experiment_version"] == VERSION
    assert audit["status"] == "pass"
    assert len(schedule) == len({row["run_id"] for row in schedule}) == 1750
    assert common.sha256_path(DEST / "config/run_table.jsonl") == audit["run_table_sha256"]
    assert all(row["action_only"] and row["schema"] == "schemas/l4_action_only.json" for row in schedule)
    assert Counter((row["geometry_condition"], row["direction_resolution"]) for row in schedule) == Counter({
        (f"C{c}", d): 125 for c in range(7) for d in ("D4", "D8")
    })
    for relative, digest in read_json(DEST / "audit/frozen_file_hashes.json").items():
        assert common.sha256_path(DEST / relative) == digest, relative
    return {"status": "prepared", "version": VERSION, "scheduled_calls": len(schedule)}


def main() -> None:
    if (DEST / "audit/preflight_report.json").exists():
        print(json.dumps(verify(), indent=2))
        return
    if DEST.exists():
        raise RuntimeError(f"Destination exists without preflight: {DEST}")

    # Copy immutable inputs; deliberately do not copy inference or replay outputs.
    for directory in ("images", "geometry", "prompts", "schemas"):
        shutil.copytree(SOURCE / directory, DEST / directory)
    shutil.copytree(SOURCE / "audit/masks", DEST / "audit/masks")
    shutil.copytree(SOURCE / "audit/scene_geometry", DEST / "audit/scene_geometry")

    source_config = read_json(SOURCE / "config/experiment_config.json")
    config = copy.deepcopy(source_config)
    config.update({
        "experiment_version": VERSION,
        "reference_experiment": str(SOURCE),
        "task_definition": "Frozen L3 blocker input; output only one L4 action label.",
        "evaluation": "Action-label agreement with scene manifest advisory; no physical replay without parameters.",
        "stage_policy": {
            "stage_1": "Frozen L3 v19 blocking_reason and blocker_id from v25",
            "stage_2": "Choose one action label only; no object ID or action parameters",
            "none_case": "RETRIEVE target",
            "blocked_case": "Choose TRANSLATE, ROTATE, or LIFT_AND_RELOCATE for frozen blocker",
        },
        "schemas": {**config["schemas"], "L4_D4": "schemas/l4_action_only.json", "L4_D8": "schemas/l4_action_only.json"},
    })
    config.pop("comparison_experiment", None)
    common.write_json(DEST / "config/experiment_config.json", config)
    shutil.copy2(SOURCE / "config/frozen_scenes.json", DEST / "config/frozen_scenes.json")
    (DEST / "prompts/l4_en.txt").write_text(PROMPT + "\n", encoding="utf-8")
    common.write_json(DEST / "schemas/l4_action_only.json", SCHEMA)

    source_rows = read_jsonl(SOURCE / "config/run_table.jsonl")
    schedule = []
    for original in source_rows:
        row = copy.deepcopy(original)
        row["run_id"] = "v26__" + original["run_id"].removeprefix("v25__")
        row["experiment_version"] = VERSION
        row["schema"] = "schemas/l4_action_only.json"
        row["schema_sha256"] = common.sha256_path(DEST / row["schema"])
        row["action_only"] = True
        row["source_v25_run_id"] = original["run_id"]
        schedule.append(row)
    write_jsonl(DEST / "config/run_table.jsonl", schedule)

    frozen_hashes = {
        str(path.relative_to(DEST)): common.sha256_path(path)
        for directory in ("config", "prompts", "schemas", "geometry", "images", "audit/masks", "audit/scene_geometry")
        for path in (DEST / directory).rglob("*") if path.is_file()
    }
    common.write_json(DEST / "audit/frozen_file_hashes.json", frozen_hashes)
    common.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": VERSION,
        "scheduled_calls": len(schedule),
        "run_table_sha256": common.sha256_path(DEST / "config/run_table.jsonl"),
        "source_v25_run_table_sha256": common.sha256_path(SOURCE / "config/run_table.jsonl"),
        "source_l3_inference_sha256": common.sha256_path(
            ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19/logs/runs.jsonl"
        ),
        "frozen_file_count": len(frozen_hashes),
        "failures": [],
    })
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
