#!/usr/bin/env python3
"""Prepare a full native-L4 action experiment from the corrected v19 inputs."""

from __future__ import annotations

import copy
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
MASK_SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_maskfix_validation"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l4_full_from_v19_v24"
VERSION = "l4_action_full_from_v19_v24"
RESOLUTIONS = ("D4", "D8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def condition_key(row: dict) -> str:
    return f"{row['geometry_condition']}_{row['direction_resolution']}"


def verify() -> tuple[dict, list[dict]]:
    config = read_json(DEST / "config/experiment_config.json")
    audit = read_json(DEST / "audit/preflight_report.json")
    schedule = read_jsonl(DEST / "config/run_table.jsonl")
    assert config["experiment_version"] == VERSION
    assert audit["status"] == "pass"
    assert common.sha256_path(DEST / "config/run_table.jsonl") == audit["run_table_sha256"]
    assert len(schedule) == len({row["run_id"] for row in schedule}) == 1750
    assert all(row["level"] == "L4" for row in schedule)
    assert Counter(condition_key(row) for row in schedule) == Counter({
        f"C{condition}_{resolution}": 125
        for condition in range(7)
        for resolution in RESOLUTIONS
    })
    for relative, digest in read_json(DEST / "audit/frozen_file_hashes.json").items():
        assert common.sha256_path(DEST / relative) == digest, relative
    return config, schedule


def main() -> None:
    if (DEST / "audit/preflight_report.json").exists():
        config, schedule = verify()
        print(json.dumps({
            "status": "already_prepared",
            "experiment": str(DEST),
            "version": config["experiment_version"],
            "scheduled_calls": len(schedule),
        }, indent=2))
        return
    if DEST.exists():
        raise RuntimeError(f"Destination exists without a frozen preflight: {DEST}")

    source_config = read_json(SOURCE / "config/experiment_config.json")
    source_schedule = read_jsonl(SOURCE / "config/run_table.jsonl")
    source_scenes = read_json(SOURCE / "config/frozen_scenes.json")
    assert len(source_schedule) == 1500
    assert len(source_scenes) == 25

    config = copy.deepcopy(source_config)
    config.update({
        "experiment_version": VERSION,
        "scheduled_calls": 1750,
        "levels": ["L4"],
        "reference_experiment": str(SOURCE),
        "input_source_experiment": str(SOURCE),
        "baseline_mask_source": str(MASK_SOURCE),
        "task_definition": (
            "Choose one executable next action that makes the target directly graspable; "
            "evaluate native L4 outputs by direct Isaac Sim replay."
        ),
        "matrix_definition": (
            "C0-C6 crossed with D4/D8, 25 scenes, five fixed seeds; corrected v19 "
            "images and geometry are held fixed."
        ),
        "source_v19_task_note": (
            "Only the frozen inputs are inherited from v19. The native L4 task and output "
            "schema are action selection, not L3 blocker-reason classification."
        ),
    })

    for relative in (
        "prompts/system_en.txt",
        "prompts/l1_en.txt",
        "prompts/l2_en.txt",
        "prompts/l3_en.txt",
        "prompts/l4_en.txt",
        "schemas/l1.json",
        "schemas/l2.json",
        "schemas/l3.json",
        "schemas/l4_d4.json",
        "schemas/l4_d8.json",
    ):
        copy_file(SOURCE / relative, DEST / relative)

    for scene in source_scenes:
        for relative in (
            scene["numbered_rgb"],
            scene["id_mapping"],
            scene["scene_geometry"],
            scene["ground_truth"],
        ):
            copy_file(SOURCE / relative, DEST / relative)
        for geometry in scene["geometry_files"].values():
            copy_file(SOURCE / geometry["path"], DEST / geometry["path"])
        gt = read_json(DEST / scene["ground_truth"])
        assert gt["L4"]["evaluation"] == "direct_isaac_sim_replay"

    mask_paths = sorted((MASK_SOURCE / "audit/masks").glob("*.png"))
    assert len(mask_paths) == 50
    for path in mask_paths:
        copy_file(path, DEST / "audit/masks" / path.name)
    common.write_json(DEST / "config/frozen_scenes.json", source_scenes)

    schema_paths = {
        resolution: f"schemas/l4_{resolution.lower()}.json"
        for resolution in RESOLUTIONS
    }
    schedule = []
    for original in source_schedule:
        resolutions = RESOLUTIONS if original["direction_resolution"] is None else (
            original["direction_resolution"],
        )
        for resolution in resolutions:
            row = copy.deepcopy(original)
            row["source_l3_v19_run_id"] = original["run_id"]
            row.pop("single_valid_reference_run_id", None)
            row["experiment_version"] = VERSION
            row["level"] = "L4"
            row["direction_resolution"] = resolution
            row["schema"] = schema_paths[resolution]
            row["schema_sha256"] = common.sha256_path(DEST / row["schema"])
            suffix = f"r{int(row['repeat_index']):02d}_s{int(row['seed'])}"
            condition = f"{row['geometry_condition']}_{resolution}"
            row["run_id"] = f"v24__{row['scene_id']}__L4__{condition}__{suffix}"
            row["reference_run_id"] = (
                f"v1__{row['scene_id']}__L4__{condition}__{suffix}"
            )
            schedule.append(row)

    assert len(schedule) == len({row["run_id"] for row in schedule}) == 1750
    assert Counter(condition_key(row) for row in schedule) == Counter({
        f"C{condition}_{resolution}": 125
        for condition in range(7)
        for resolution in RESOLUTIONS
    })
    random.Random(20260918).shuffle(schedule)

    common.write_json(DEST / "config/experiment_config.json", config)
    run_table = DEST / "config/run_table.jsonl"
    write_jsonl(run_table, schedule)

    frozen_hashes = {
        str(path.relative_to(DEST)): common.sha256_path(path)
        for folder in ("config", "prompts", "schemas", "geometry", "images", "audit/masks", "audit/scene_geometry")
        for path in (DEST / folder).rglob("*")
        if path.is_file()
    }
    common.write_json(DEST / "audit/frozen_file_hashes.json", frozen_hashes)
    common.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass",
        "checked_at": now(),
        "experiment_version": VERSION,
        "source_experiment": str(SOURCE),
        "source_run_table_sha256": common.sha256_path(SOURCE / "config/run_table.jsonl"),
        "run_table_sha256": common.sha256_path(run_table),
        "scheduled_calls": 1750,
        "scene_count": 25,
        "seed_count": 5,
        "condition_count": 14,
        "condition_sizes": dict(sorted(Counter(condition_key(row) for row in schedule).items())),
        "frozen_file_count": len(frozen_hashes),
        "native_l4_action_schema": True,
        "direct_isaac_sim_replay_required": True,
        "failures": [],
    })
    config, schedule = verify()
    print(json.dumps({
        "status": "prepared",
        "experiment": str(DEST),
        "version": config["experiment_version"],
        "scheduled_calls": len(schedule),
        "condition_count": 14,
    }, indent=2))


if __name__ == "__main__":
    main()
