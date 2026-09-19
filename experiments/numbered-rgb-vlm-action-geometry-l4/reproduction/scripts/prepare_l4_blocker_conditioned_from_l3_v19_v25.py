#!/usr/bin/env python3
"""Prepare two-stage L3 blocker selection -> L4 blocker action experiment."""

from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l4_full_from_v19_v24"
L3_SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l4_blocker_conditioned_v25"
VERSION = "l4_blocker_conditioned_from_l3_v19_v25"

ACTION_PROMPT = """The perception stage has already applied the L3 strongest-blocker rule.
Its frozen output for this exact scene, geometry condition, and seed is:

- blocking_reason: {selected_blocking_reason}
- selected blocker_id: {selected_blocker_id}

Choose exactly one next action from:
RETRIEVE, TRANSLATE, ROTATE, LIFT_AND_RELOCATE.

Follow these constraints exactly:

1. If blocking_reason is NONE and blocker_id is null, output RETRIEVE with the
   target object ID. Do not output any other action.
2. Otherwise, the selected blocker_id is the single visible non-target blocker
   chosen by L3. Output TRANSLATE, ROTATE, or LIFT_AND_RELOCATE for exactly that
   blocker_id. Never select the target and never select a different object.
3. Select the action that can be executed without contacting any unselected
   object and that makes the target directly graspable after the action.
4. TRANSLATE: output a shelf direction, an integer distance in 1 cm increments,
   and the matching qualitative distance level.
5. ROTATE: rotate around the vertical axis through the selected blocker's
   footprint center. Output CW or CCW as viewed from above, an angle in 5-degree
   increments, and the matching qualitative angle level.
6. LIFT_AND_RELOCATE: do not output a destination, lift height, or path.
7. Output one action only, not a sequence.

Allowed translation directions:
{direction_list}"""


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
    assert all("selected_blocking_reason" in row for row in schedule)
    assert Counter(condition_key(row) for row in schedule) == Counter({
        f"C{condition}_{resolution}": 125
        for condition in range(7)
        for resolution in ("D4", "D8")
    })
    for relative, digest in read_json(DEST / "audit/frozen_file_hashes.json").items():
        assert common.sha256_path(DEST / relative) == digest, relative
    return config, schedule


def main() -> None:
    if (DEST / "audit/preflight_report.json").exists():
        config, schedule = verify()
        print(json.dumps({
            "status": "already_prepared",
            "version": config["experiment_version"],
            "scheduled_calls": len(schedule),
        }, indent=2))
        return
    if DEST.exists():
        raise RuntimeError(f"Destination exists without a frozen preflight: {DEST}")

    source_config = read_json(SOURCE / "config/experiment_config.json")
    source_schedule = read_jsonl(SOURCE / "config/run_table.jsonl")
    source_scenes = read_json(SOURCE / "config/frozen_scenes.json")
    l3_rows = {row["run_id"]: row for row in read_jsonl(L3_SOURCE / "logs/runs.jsonl")}
    assert len(source_schedule) == 1750
    assert len(l3_rows) == 1500

    config = copy.deepcopy(source_config)
    config.update({
        "experiment_version": VERSION,
        "scheduled_calls": 1750,
        "levels": ["L4"],
        "reference_experiment": str(SOURCE),
        "l3_selection_source_experiment": str(L3_SOURCE),
        "task_definition": (
            "Two-stage evaluation: reuse the exact L3 v19 strongest-blocker output, "
            "then choose one executable L4 action for exactly that selected blocker."
        ),
        "stage_policy": {
            "stage_1": "Frozen L3 v19 blocking_reason and strongest blocker_id",
            "stage_2": "L4 action selection conditioned on the frozen blocker",
            "none_case": "blocking_reason NONE requires RETRIEVE target",
            "blocked_case": "non-NONE requires non-target action on selected blocker only",
        },
        "comparison_experiment": str(SOURCE),
    })

    for relative in (
        "prompts/system_en.txt", "prompts/l1_en.txt", "prompts/l2_en.txt",
        "prompts/l3_en.txt", "schemas/l1.json", "schemas/l2.json", "schemas/l3.json",
        "schemas/l4_d4.json", "schemas/l4_d8.json",
    ):
        copy_file(SOURCE / relative, DEST / relative)
    (DEST / "prompts").mkdir(parents=True, exist_ok=True)
    (DEST / "prompts/l4_en.txt").write_text(ACTION_PROMPT + "\n", encoding="utf-8")

    for scene in source_scenes:
        for relative in (
            scene["numbered_rgb"], scene["id_mapping"], scene["scene_geometry"],
            scene["ground_truth"],
        ):
            copy_file(SOURCE / relative, DEST / relative)
        for geometry in scene["geometry_files"].values():
            copy_file(SOURCE / geometry["path"], DEST / geometry["path"])
    for path in sorted((SOURCE / "audit/masks").glob("*.png")):
        copy_file(path, DEST / "audit/masks" / path.name)
    common.write_json(DEST / "config/frozen_scenes.json", source_scenes)

    schedule = []
    source_l3_reason_counts = Counter()
    source_l3_joint = Counter()
    for original in source_schedule:
        l3_run_id = original["source_l3_v19_run_id"]
        l3 = l3_rows[l3_run_id]
        parsed = l3["parsed_response"]
        assert isinstance(parsed, dict)
        reason = parsed["blocking_reason"]
        blocker = parsed["blocker_id"]
        assert reason == "NONE" or isinstance(blocker, int)
        assert reason != "NONE" or blocker is None
        row = copy.deepcopy(original)
        row["source_direct_l4_run_id"] = original["run_id"]
        row["experiment_version"] = VERSION
        row["run_id"] = "v25__" + original["run_id"].removeprefix("v24__")
        row["l3_source_run_id"] = l3_run_id
        row["selected_blocking_reason"] = reason
        row["selected_blocker_id"] = blocker
        row["l3_source_response"] = parsed
        row["l3_source_reason_correct"] = bool(
            l3["scoring"]["blocking_reason_correct"]
        )
        row["l3_source_blocker_correct"] = bool(
            l3["scoring"]["blocker_valid_set_member"]
        )
        row["l3_source_joint_correct"] = bool(l3["scoring"]["task_correct"])
        source_l3_reason_counts[reason] += 1
        source_l3_joint[row["l3_source_joint_correct"]] += 1
        schedule.append(row)

    assert len(schedule) == len({row["run_id"] for row in schedule}) == 1750
    common.write_json(DEST / "config/experiment_config.json", config)
    run_table = DEST / "config/run_table.jsonl"
    write_jsonl(run_table, schedule)

    frozen_hashes = {
        str(path.relative_to(DEST)): common.sha256_path(path)
        for folder in (
            "config", "prompts", "schemas", "geometry", "images",
            "audit/masks", "audit/scene_geometry",
        )
        for path in (DEST / folder).rglob("*")
        if path.is_file()
    }
    common.write_json(DEST / "audit/frozen_file_hashes.json", frozen_hashes)
    common.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass",
        "checked_at": now(),
        "experiment_version": VERSION,
        "scheduled_calls": 1750,
        "scene_count": 25,
        "seed_count": 5,
        "condition_count": 14,
        "run_table_sha256": common.sha256_path(run_table),
        "l3_source_run_table_sha256": common.sha256_path(
            L3_SOURCE / "config/run_table.jsonl"
        ),
        "l3_source_inference_sha256": common.sha256_path(
            L3_SOURCE / "logs/runs.jsonl"
        ),
        "source_l3_reason_counts": dict(sorted(source_l3_reason_counts.items())),
        "source_l3_joint_correct_counts": {
            str(key).lower(): value for key, value in sorted(source_l3_joint.items())
        },
        "condition_sizes": dict(sorted(Counter(condition_key(row) for row in schedule).items())),
        "frozen_file_count": len(frozen_hashes),
        "failures": [],
    })
    config, schedule = verify()
    print(json.dumps({
        "status": "prepared",
        "experiment": str(DEST),
        "version": config["experiment_version"],
        "scheduled_calls": len(schedule),
        "source_l3_reason_counts": dict(sorted(source_l3_reason_counts.items())),
        "source_l3_joint_correct_counts": {
            str(key).lower(): value for key, value in sorted(source_l3_joint.items())
        },
    }, indent=2))


if __name__ == "__main__":
    main()
