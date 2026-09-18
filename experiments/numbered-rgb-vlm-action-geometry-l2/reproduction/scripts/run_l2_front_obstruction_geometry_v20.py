#!/usr/bin/env python3
"""Run the agreed binary front-obstruction prompt on frozen v19 inputs."""
import argparse
import fcntl
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import run_l2_approach_access_geometry_v18 as engine

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_straight_access_geometry_v19"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_front_obstruction_geometry_v20"
VERSION = "l2_front_obstruction_geometry_v20"
SYSTEM = """Assess whether objects in front of the target obstruct its removal.
Numbered badges identify objects.
Return only a JSON object."""
USER_TEMPLATE = """Target object ID: {target_object_id}
Geometry information:
{geometry_json}

Does any object in front of target {target_object_id}
obstruct pulling the target out of the shelf, even slightly?

Return {"decision":"BLOCKED"} or {"decision":"CLEAR"}."""
SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["BLOCKED", "CLEAR"]}},
    "required": ["decision"],
    "additionalProperties": False,
}


def render(row):
    geometry = "null" if row["geometry_json_raw"] is None else json.dumps(
        row["geometry_json_raw"], ensure_ascii=False, indent=2, sort_keys=True)
    return USER_TEMPLATE.replace("{target_object_id}", str(row["target_object_id"])).replace("{geometry_json}", geometry)


def prepare():
    reference = engine.read(SOURCE / "config/experiment_config.json")
    config = {
        **{k: reference[k] for k in ("scene_count", "scheduled_calls", "seeds", "run_order_seed",
                                   "model_id", "model_path", "generation", "runtime", "conditions", "geometry_policy")},
        "experiment_version": VERSION, "reference_experiment": str(SOURCE),
        "reference_config_sha256": engine.digest(SOURCE / "config/experiment_config.json"),
        "reference_run_table_sha256": engine.digest(SOURCE / "config/run_table.jsonl"),
        "reference_logs_sha256": engine.digest(SOURCE / "logs/runs.jsonl"),
        "inference_implementation": str(Path(engine.__file__).resolve()),
        "inference_implementation_sha256": engine.digest(Path(engine.__file__)),
        "system_prompt": SYSTEM, "user_prompt_template": USER_TEMPLATE,
        "schema": SCHEMA,
        "schema_sha256": engine.text_digest(json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        "gripper_information_supplied": False,
        "task_definition": "Any object in front obstructing pulling target out, even slightly; binary BLOCKED/CLEAR, not visual occlusion degree.",
        "evaluation": "descriptive_front_obstruction_predictions_no_front_obstruction_execution_ground_truth",
        "comparison_note": "Question, system wording and labels changed from v19; CLEAR is not a complete retrieval feasibility guarantee.",
    }
    config_path = DEST / "config/experiment_config.json"
    prior_schedule = engine.rows(SOURCE / "config/run_table.jsonl")
    if config_path.exists():
        assert engine.read(config_path) == config, "Frozen config changed"
        for relative, sha in engine.read(DEST / "audit/frozen_file_hashes.json").items():
            assert engine.digest(DEST / relative) == sha, relative
        schedule = engine.rows(DEST / "config/run_table.jsonl")
    else:
        assert not (DEST / "logs/runs.jsonl").exists(), "Refuse to overwrite inference"
        scenes = engine.read(SOURCE / "config/frozen_scenes.json")
        assert len(scenes) == 25
        for scene in scenes:
            assert engine.digest(Path(scene["image_path"])) == scene["image_sha256"]
            assert engine.digest(Path(scene["id_mapping_path"])) == scene["id_mapping_sha256"]
            assert engine.read(Path(scene["id_mapping_path"]))["target_object_id"] == scene["target_object_id"]
        schedule, copied = [], set()
        for prior in prior_schedule:
            row = dict(prior)
            assert prior["run_id"].startswith("v19__")
            row["run_id"] = prior["run_id"].replace("v19__", "v20__", 1)
            if prior["geometry_path"]:
                source_path = Path(prior["geometry_path"])
                relative = source_path.relative_to(SOURCE)
                payload_path = DEST / relative
                if relative not in copied:
                    assert engine.digest(source_path) == prior["geometry_sha256"]
                    payload_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_path, payload_path)
                    copied.add(relative)
                row["geometry_path"] = str(payload_path)
                assert engine.read(payload_path) == row["geometry_json_raw"]
            row["user_prompt"] = render(row)
            row["prompt_sha256"] = engine.text_digest(SYSTEM + "\n\n" + row["user_prompt"])
            schedule.append(row)
        engine.write_json(DEST / "config/frozen_scenes.json", scenes)
        engine.write_json(DEST / "schemas/l2.json", SCHEMA)
        (DEST / "prompts").mkdir(parents=True, exist_ok=True)
        (DEST / "logs").mkdir(parents=True, exist_ok=True)
        (DEST / "prompts/system_en.txt").write_text(SYSTEM + "\n", encoding="utf-8")
        (DEST / "prompts/l2_template_en.txt").write_text(USER_TEMPLATE + "\n", encoding="utf-8")
        with (DEST / "config/run_table.jsonl").open("w", encoding="utf-8") as handle:
            for row in schedule:
                engine.append(handle, row)
        engine.write_json(config_path, config)
        frozen = [Path("config/experiment_config.json"), Path("config/frozen_scenes.json"),
                  Path("config/run_table.jsonl"), Path("schemas/l2.json"),
                  Path("prompts/system_en.txt"), Path("prompts/l2_template_en.txt"), *sorted(copied)]
        engine.write_json(DEST / "audit/frozen_file_hashes.json", {str(p): engine.digest(DEST / p) for p in frozen})
    assert len(schedule) == len({r["run_id"] for r in schedule}) == len(prior_schedule) == 1500
    groups, checked_paths = defaultdict(list), set()
    for row, prior in zip(schedule, prior_schedule):
        for k in set(prior) - {"run_id", "geometry_path", "user_prompt", "prompt_sha256"}:
            assert row[k] == prior[k], (row["run_id"], k)
        assert row["user_prompt"] == render(row)
        assert row["prompt_sha256"] == engine.text_digest(SYSTEM + "\n\n" + row["user_prompt"])
        groups[(row["scene_id"], row["condition_key"])].append(row["seed"])
        for path_key, hash_key in (("image_path", "image_sha256"), ("id_mapping_path", "id_mapping_sha256"), ("geometry_path", "geometry_sha256")):
            p = row[path_key]
            if p and p not in checked_paths:
                assert engine.digest(Path(p)) == row[hash_key], p
                checked_paths.add(p)
    assert len(groups) == 300
    assert all(len(seeds) == 5 and set(seeds) == set(config["seeds"]) for seeds in groups.values())
    engine.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass", "checked_at": engine.now(), "scheduled_calls": len(schedule),
        "unique_run_ids": len({r["run_id"] for r in schedule}), "scene_count": 25,
        "condition_counts": dict(Counter(r["condition_key"] for r in schedule)),
        "identical_v19_images_geometry_model_settings_seeds_order": True,
        "geometry_payloads_unchanged": len({r["geometry_path"] for r in schedule if r["geometry_path"]}),
        "schema_labels": ["BLOCKED", "CLEAR"], "agreed_question_verbatim": True,
        "run_table_sha256": engine.digest(DEST / "config/run_table.jsonl"), "failures": [],
    })
    return config, schedule


def validate(config, schedule, records):
    assert len(records) == len({r["run_id"] for r in records}) == 1500
    assert [r["run_id"] for r in records] == [r["run_id"] for r in schedule]
    for row, planned in zip(records, schedule):
        for k, value in planned.items():
            assert row[k] == value, (row["run_id"], k)
        for k in ("experiment_version", "model_id", "model_path", "generation", "system_prompt", "schema", "schema_sha256"):
            assert row[k] == config[k], (row["run_id"], k)
        parsed = json.loads(row["raw_response"])
        assert row["schema_valid"] and parsed == row["parsed_response"]
        assert set(parsed) == {"decision"} and parsed["decision"] in ("BLOCKED", "CLEAR")
        assert row["finish_reason"] == "stop"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, schedule = prepare()
        if args.prepare_only:
            print(json.dumps(engine.read(DEST / "audit/preflight_report.json"), indent=2))
            return
        engine.DEST, engine.VERSION, engine.SYSTEM, engine.SCHEMA = DEST, VERSION, SYSTEM, SCHEMA
        records = engine.rows(DEST / "logs/runs.jsonl") if args.verify_only else engine.infer(config, schedule)
        validate(config, schedule, records)
        engine.verify_completion(config, schedule, records)


if __name__ == "__main__":
    main()
