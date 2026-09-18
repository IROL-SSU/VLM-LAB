#!/usr/bin/env python3
"""Fresh inference using exact Notion v16 prompts, schema, and frozen inputs."""
import argparse
import fcntl
import json
from collections import Counter, defaultdict
from pathlib import Path

import run_l2_approach_access_geometry_v18 as engine

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_v16_prompt_replay_v21"
VERSION = "l2_v16_prompt_replay_v21"


def prepare():
    original = engine.read(SOURCE / "config/experiment_config.json")
    notion_path = DEST / "config/notion_prompt_source.json"
    notion = engine.read(notion_path)
    for field in ("system_prompt", "user_prompt_template", "schema"):
        assert notion[field] == original[field], f"Notion and v16 differ: {field}"
    config = {
        **{k: original[k] for k in ("scene_count", "scheduled_calls", "seeds", "run_order_seed", "model_id",
                                   "model_path", "generation", "runtime", "conditions", "geometry_policy",
                                   "system_prompt", "user_prompt_template", "schema", "schema_sha256")},
        "experiment_version": VERSION, "reference_experiment": str(SOURCE),
        "reference_config_sha256": engine.digest(SOURCE / "config/experiment_config.json"),
        "reference_run_table_sha256": engine.digest(SOURCE / "config/run_table.jsonl"),
        "reference_logs_sha256": engine.digest(SOURCE / "logs/runs.jsonl"),
        "notion_source_sha256": engine.digest(notion_path), "notion_url": notion["url"],
        "notion_page_last_edited_at": notion["page_last_edited_at"],
        "inference_implementation": str(Path(engine.__file__).resolve()),
        "inference_implementation_sha256": engine.digest(Path(engine.__file__)),
        "gripper_information_supplied": False,
        "evaluation": "response_counts_and_original_v16_scene_design_agreement_not_physical_accuracy",
        "comparison_note": "Fresh 1500-call repeat of exact v16 prompts and inputs; not reuse of prior responses. Original image/geometry files are hash-verified and referenced read-only.",
    }
    prior = engine.rows(SOURCE / "config/run_table.jsonl")
    schedule = []
    checked, groups = set(), defaultdict(list)
    for previous in prior:
        row = dict(previous)
        assert row["run_id"].startswith("v16__")
        row["run_id"] = row["run_id"].replace("v16__", "v21__", 1)
        geometry = "null" if row["geometry_json_raw"] is None else json.dumps(row["geometry_json_raw"], ensure_ascii=False, indent=2, sort_keys=True)
        prompt = config["user_prompt_template"].replace("{target_object_id}", str(row["target_object_id"])).replace("{geometry_json}", geometry)
        assert row["user_prompt"] == prompt
        assert row["prompt_sha256"] == engine.text_digest(config["system_prompt"] + "\n\n" + prompt)
        for pkey, hkey in (("image_path", "image_sha256"), ("id_mapping_path", "id_mapping_sha256"), ("geometry_path", "geometry_sha256")):
            p = row[pkey]
            if p and p not in checked:
                assert engine.digest(Path(p)) == row[hkey], p
                if pkey == "geometry_path":
                    assert engine.read(Path(p)) == row["geometry_json_raw"]
                if pkey == "id_mapping_path":
                    assert engine.read(Path(p))["target_object_id"] == row["target_object_id"]
                checked.add(p)
        groups[(row["scene_id"], row["condition_key"])].append(row["seed"])
        schedule.append(row)
    assert len(schedule) == len({r["run_id"] for r in schedule}) == 1500
    assert len(groups) == 300 and all(len(s) == 5 and set(s) == set(config["seeds"]) for s in groups.values())
    assert len({r["scene_id"] for r in schedule}) == 25
    assert set(Counter(r["condition_key"] for r in schedule).values()) == {125}
    path = DEST / "config/experiment_config.json"
    if path.exists():
        assert engine.read(path) == config, "Frozen configuration changed"
        assert engine.rows(DEST / "config/run_table.jsonl") == schedule
        for relative, sha in engine.read(DEST / "audit/frozen_file_hashes.json").items():
            assert engine.digest(DEST / relative) == sha, relative
    else:
        assert not (DEST / "logs/runs.jsonl").exists()
        engine.write_json(path, config)
        engine.write_json(DEST / "config/frozen_scenes.json", engine.read(SOURCE / "config/frozen_scenes.json"))
        engine.write_json(DEST / "schemas/l2.json", config["schema"])
        (DEST / "prompts").mkdir(exist_ok=True)
        (DEST / "logs").mkdir(exist_ok=True)
        (DEST / "prompts/system_en.txt").write_text(config["system_prompt"] + "\n", encoding="utf-8")
        (DEST / "prompts/l2_template_en.txt").write_text(config["user_prompt_template"] + "\n", encoding="utf-8")
        with (DEST / "config/run_table.jsonl").open("w", encoding="utf-8") as handle:
            for row in schedule:
                engine.append(handle, row)
        frozen = ["config/experiment_config.json", "config/notion_prompt_source.json", "config/frozen_scenes.json",
                  "config/run_table.jsonl", "schemas/l2.json", "prompts/system_en.txt", "prompts/l2_template_en.txt"]
        engine.write_json(DEST / "audit/frozen_file_hashes.json", {p: engine.digest(DEST / p) for p in frozen})
    engine.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass", "scheduled_calls": 1500, "scene_condition_groups": 300,
        "notion_prompts_and_schema_exactly_match_v16": True,
        "all_inputs_prompts_settings_seeds_order_identical_to_v16": True,
        "new_inference_not_reused_responses": True, "checked_at": engine.now(),
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
        assert set(parsed) == {"decision"} and parsed["decision"] in ("RETRIEVE_NOW", "REARRANGE_FIRST")
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
        engine.DEST, engine.VERSION = DEST, VERSION
        engine.SYSTEM, engine.SCHEMA = config["system_prompt"], config["schema"]
        records = engine.rows(DEST / "logs/runs.jsonl") if args.verify_only else engine.infer(config, schedule)
        validate(config, schedule, records)
        engine.verify_completion(config, schedule, records)


if __name__ == "__main__":
    main()
