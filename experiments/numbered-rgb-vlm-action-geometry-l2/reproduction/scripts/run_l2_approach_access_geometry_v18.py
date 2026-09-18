#!/usr/bin/env python3
"""Run the agreed concise approach-path prompt on the frozen v16 inputs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_approach_access_geometry_v18"
VERSION = "l2_approach_access_geometry_v18"
SYSTEM = """Assess whether the target is accessible from the shelf opening.
Numbered badges identify objects.
Return only a JSON object. Do not include explanations."""
USER_TEMPLATE = """Target object ID: {target_object_id}

Geometry information:
{geometry_json}

Is there a clear approach path from the shelf opening to the target
without moving or contacting other objects?

Choose:
- ACCESSIBLE: there is a clear approach path.
- BLOCKED: other objects block the approach.

Return exactly one JSON object with the field "decision"."""
SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["ACCESSIBLE", "BLOCKED"]}},
    "required": ["decision"],
    "additionalProperties": False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def prepare():
    reference = read(SOURCE / "config/experiment_config.json")
    config = {
        **{key: reference[key] for key in (
            "scene_count", "scheduled_calls", "seeds", "run_order_seed", "model_id",
            "model_path", "generation", "runtime", "conditions", "geometry_policy",
        )},
        "experiment_version": VERSION,
        "reference_experiment": str(SOURCE),
        "reference_config_sha256": digest(SOURCE / "config/experiment_config.json"),
        "reference_run_table_sha256": digest(SOURCE / "config/run_table.jsonl"),
        "reference_logs_sha256": digest(SOURCE / "logs/runs.jsonl"),
        "system_prompt": SYSTEM,
        "user_prompt_template": USER_TEMPLATE,
        "schema": SCHEMA,
        "schema_sha256": text_digest(json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        "gripper_information_supplied": False,
        "task_definition": "Clear approach path from shelf opening to target without moving or contacting other objects; no grasp or removal judgment.",
        "evaluation": "descriptive_accessibility_predictions_no_approach_execution_gt",
        "v16_comparison": "RETRIEVE_NOW to ACCESSIBLE and REARRANGE_FIRST to BLOCKED is a semantic response comparison, not correctness or accuracy.",
    }
    config_path = DEST / "config/experiment_config.json"
    if config_path.exists():
        assert read(config_path) == config, "Frozen config changed"
        audit = read(DEST / "audit/preflight_report.json")
        assert audit["status"] == "pass"
        for relative, sha in read(DEST / "audit/frozen_file_hashes.json").items():
            assert digest(DEST / relative) == sha, f"Frozen file changed: {relative}"
        schedule = rows(DEST / "config/run_table.jsonl")
    else:
        assert not (DEST / "logs/runs.jsonl").exists(), "Existing inference without frozen config"
        scenes = read(SOURCE / "config/frozen_scenes.json")
        assert len(scenes) == 25
        for scene in scenes:
            assert digest(Path(scene["image_path"])) == scene["image_sha256"]
            assert digest(Path(scene["id_mapping_path"])) == scene["id_mapping_sha256"]
            assert read(Path(scene["id_mapping_path"]))["target_object_id"] == scene["target_object_id"]
        schedule = []
        copied = set()
        for prior in rows(SOURCE / "config/run_table.jsonl"):
            row = dict(prior)
            row["run_id"] = prior["run_id"].replace("v16__", "v18__", 1)
            if prior["geometry_path"]:
                source_path = Path(prior["geometry_path"])
                relative = source_path.relative_to(SOURCE)
                payload_path = DEST / relative
                if relative not in copied:
                    assert digest(source_path) == prior["geometry_sha256"]
                    payload_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_path, payload_path)
                    copied.add(relative)
                assert read(payload_path) == prior["geometry_json_raw"]
                row["geometry_path"] = str(payload_path)
            payload = row["geometry_json_raw"]
            geometry = "null" if payload is None else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            row["user_prompt"] = USER_TEMPLATE.replace("{target_object_id}", str(row["target_object_id"])).replace("{geometry_json}", geometry)
            row["prompt_sha256"] = text_digest(SYSTEM + "\n\n" + row["user_prompt"])
            schedule.append(row)
        write_json(DEST / "config/frozen_scenes.json", scenes)
        write_json(DEST / "schemas/l2.json", SCHEMA)
        (DEST / "prompts").mkdir(parents=True, exist_ok=True)
        (DEST / "prompts/system_en.txt").write_text(SYSTEM + "\n", encoding="utf-8")
        (DEST / "prompts/l2_template_en.txt").write_text(USER_TEMPLATE + "\n", encoding="utf-8")
        (DEST / "logs").mkdir(parents=True, exist_ok=True)
        with (DEST / "config/run_table.jsonl").open("w", encoding="utf-8") as handle:
            for row in schedule:
                append(handle, row)
        frozen_paths = [
            Path("config/frozen_scenes.json"), Path("config/run_table.jsonl"),
            Path("schemas/l2.json"), Path("prompts/system_en.txt"), Path("prompts/l2_template_en.txt"),
            *sorted(copied),
        ]
        write_json(DEST / "audit/frozen_file_hashes.json", {str(p): digest(DEST / p) for p in frozen_paths})
        write_json(DEST / "audit/preflight_report.json", {
            "status": "pass", "checked_at": now(), "scheduled_calls": len(schedule),
            "unique_run_ids": len({r["run_id"] for r in schedule}),
            "scene_count": len(scenes), "condition_counts": dict(Counter(r["condition_key"] for r in schedule)),
            "identical_v16_images_geometry_settings_seeds_and_run_order": True,
            "geometry_payloads_copied_without_changes": len(copied),
            "system_and_user_prompts_match_agreed_text": True,
            "run_table_sha256": digest(DEST / "config/run_table.jsonl"), "failures": [],
        })
        write_json(config_path, config)
    assert len(schedule) == len({r["run_id"] for r in schedule}) == 1500
    assert len({r["scene_id"] for r in schedule}) == 25
    assert len(Counter(r["condition_key"] for r in schedule)) == 12
    assert set(Counter(r["condition_key"] for r in schedule).values()) == {125}
    for row in schedule:
        assert digest(Path(row["image_path"])) == row["image_sha256"]
        if row["geometry_path"]:
            assert digest(Path(row["geometry_path"])) == row["geometry_sha256"]
    return config, schedule


def infer(config, schedule):
    import run_l2_direct_retrieval_single_image_v15 as single

    log_path = DEST / "logs/runs.jsonl"
    records = rows(log_path)
    done = {r["run_id"]: r for r in records}
    planned = {r["run_id"]: r for r in schedule}
    assert len(done) == len(records) and set(done).issubset(planned)
    for run_id, row in done.items():
        assert row["prompt_sha256"] == planned[run_id]["prompt_sha256"]
        assert row["schema"] == SCHEMA and row["generation"] == config["generation"]
    pending = [r for r in schedule if r["run_id"] not in done]
    if not pending:
        return records
    images = {}
    for row in pending:
        if row["scene_id"] not in images:
            with single.Image.open(row["image_path"]) as image:
                images[row["scene_id"]] = image.convert("RGB")
    print(f"Preflight passed: 25 scenes x 12 conditions x 5 seeds; pending={len(pending)}", flush=True)
    started = time.perf_counter()
    processor = single.AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
    engine = single.LLM(
        model=config["model_path"], runner="generate", trust_remote_code=True,
        dtype="auto", tensor_parallel_size=1,
        gpu_memory_utilization=config["runtime"]["gpu_memory_utilization"],
        max_model_len=config["runtime"]["max_model_len"], max_num_seqs=1,
        enable_prefix_caching=True, moe_backend=config["runtime"]["moe_backend"],
        limit_mm_per_prompt={"image": 1},
    )
    runtime = {
        "python": single.platform.python_version(), "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "model_load_seconds": round(time.perf_counter() - started, 3), "started_at": now(),
    }
    write_json(DEST / "logs/runtime.json", runtime)
    structured = single.StructuredOutputsParams(json=SCHEMA)
    chats = {}
    with log_path.open("a", encoding="utf-8") as handle, (DEST / "logs/attempts.jsonl").open("a", encoding="utf-8") as attempts:
        for row in pending:
            key = row["prompt_sha256"]
            if key not in chats:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
                        {"type": "text", "text": row["user_prompt"]},
                    ]},
                ]
                chats[key] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            request = {"prompt": chats[key], "multi_modal_data": {"image": images[row["scene_id"]]}, "multi_modal_uuids": {"image": row["image_sha256"]}}
            sampling = single.SamplingParams(n=1, seed=row["seed"], structured_outputs=structured, **config["generation"])
            for attempt in range(1, 4):
                started_at, started = now(), time.perf_counter()
                try:
                    output = engine.generate([request], sampling_params=[sampling], use_tqdm=False)[0]
                    completion = output.outputs[0]
                    break
                except Exception as exc:
                    append(attempts, {"run_id": row["run_id"], "seed": row["seed"], "attempt": attempt, "status": "infrastructure_error", "error": repr(exc), "started_at": started_at, "finished_at": now()})
                    if attempt == 3:
                        raise
            latency = time.perf_counter() - started
            raw, parsed, parse_error = completion.text, None, None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
            valid = isinstance(parsed, dict) and set(parsed) == {"decision"} and parsed["decision"] in SCHEMA["properties"]["decision"]["enum"]
            record = {
                **row, "experiment_version": VERSION, "model_id": config["model_id"],
                "model_path": config["model_path"], "generation": config["generation"],
                "system_prompt": SYSTEM, "schema": SCHEMA, "schema_sha256": config["schema_sha256"],
                "runtime": runtime, "gripper_information_supplied": False,
                "raw_response": raw, "parsed_response": parsed, "parse_error": parse_error,
                "schema_valid": valid, "finish_reason": completion.finish_reason, "attempt": attempt,
                "input_tokens": len(output.prompt_token_ids), "output_tokens": len(completion.token_ids),
                "latency_seconds": round(latency, 6), "started_at": started_at,
                "finished_at": now(), "task_correct": None,
            }
            assert record["input_tokens"] + config["generation"]["max_tokens"] <= config["runtime"]["max_model_len"]
            append(attempts, {"run_id": row["run_id"], "seed": row["seed"], "attempt": attempt, "status": "response_received", "raw_response": raw, "started_at": started_at, "finished_at": record["finished_at"]})
            append(handle, record)
            records.append(record)
            if len(records) == 1 or len(records) % 50 == 0:
                print(f"[{len(records)}/1500] {row['condition_key']} {row['scene_id']} seed={row['seed']} {raw}", flush=True)
                write_json(DEST / "logs/progress.json", {"completed": len(records), "expected": 1500, "updated_at": now()})
    return records


def verify_completion(config, schedule, records):
    assert len(records) == len({r["run_id"] for r in records}) == len(schedule) == 1500
    assert {r["run_id"] for r in records} == {r["run_id"] for r in schedule}
    for scene in {r["scene_id"] for r in records}:
        for condition in {r["condition_key"] for r in records}:
            group = [r for r in records if r["scene_id"] == scene and r["condition_key"] == condition]
            assert len(group) == 5 and {r["seed"] for r in group} == set(config["seeds"])
    report = {
        "status": "pass", "completed": len(records), "unique_run_ids": len(records),
        "schema_valid_calls": sum(r["schema_valid"] for r in records),
        "condition_counts": dict(Counter(r["condition_key"] for r in records)),
        "all_scene_condition_seed_sets_complete": True,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
        "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in rows(DEST / "logs/attempts.jsonl")),
        "total_generation_seconds": round(sum(r["latency_seconds"] for r in records), 3),
        "logs_sha256": digest(DEST / "logs/runs.jsonl"), "finished_at": now(),
    }
    write_json(DEST / "audit/completion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


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
            print(json.dumps(read(DEST / "audit/preflight_report.json"), ensure_ascii=False, indent=2))
            return
        records = rows(DEST / "logs/runs.jsonl") if args.verify_only else infer(config, schedule)
        verify_completion(config, schedule, records)


if __name__ == "__main__":
    main()
