#!/usr/bin/env python3
"""Extend the frozen v15 single-image prompt to all 25 source images."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict

import run_l2_direct_retrieval_single_image_v15 as single


ROOT = single.ROOT
SOURCE = single.SOURCE
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_25scenes_v15"
ORDER_SEED = 20260917


def main() -> None:
    log_path = DEST / "logs/runs.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior results: {log_path}")
    reference_path = single.DEST / "config/experiment_config.json"
    reference = json.loads(reference_path.read_text())
    system = reference["system_prompt"]
    assert reference["user_prompt"].count("Target object ID: 43") == 1
    template = reference["user_prompt"].replace("Target object ID: 43", "Target object ID: {target_object_id}", 1)
    source_manifest = SOURCE / "config/frozen_scenes.json"
    source_scenes = json.loads(source_manifest.read_text())
    assert len(source_scenes) == 25
    assert len({scene["scene_id"] for scene in source_scenes}) == 25
    assert Counter(scene["scene_family"] for scene in source_scenes) == {
        "TRANSLATE": 5, "ROTATE": 5, "LIFT_AND_RELOCATE": 5, "FC_CLEAR": 5, "FC_BLOCKED": 5,
    }
    scenes, schedule, images = [], [], {}
    for source in sorted(source_scenes, key=lambda scene: scene["scene_id"]):
        image_path = SOURCE / source["numbered_rgb"]
        mapping_path = SOURCE / source["id_mapping"]
        image_hash = single.common.sha256_path(image_path)
        mapping_hash = single.common.sha256_path(mapping_path)
        assert image_hash == source["numbered_rgb_sha256"], source["scene_id"]
        assert mapping_hash == source["id_mapping_sha256"], source["scene_id"]
        mapping = json.loads(mapping_path.read_text())
        assert mapping["target_object_id"] == source["target_object_id"]
        with single.Image.open(image_path) as image:
            images[source["scene_id"]] = image.convert("RGB")
        scene = {
            "scene_id": source["scene_id"], "scene_family": source["scene_family"],
            "target_object_id": source["target_object_id"],
            "image_path": str(image_path), "image_sha256": image_hash,
            "image_size": list(images[source["scene_id"]].size),
            "id_mapping_path": str(mapping_path), "id_mapping_sha256": mapping_hash,
        }
        scenes.append(scene)
        user = template.replace("{target_object_id}", str(scene["target_object_id"]))
        for seed in single.SEEDS:
            schedule.append({
                **scene, "run_id": f"v15_all25__{scene['scene_id']}__L2_C0__s{seed}",
                "seed": seed, "user_prompt": user,
                "prompt_sha256": single.common.sha256_text(system + "\n\n" + user),
            })
    assert len(schedule) == 125 and len({row["run_id"] for row in schedule}) == 125
    probe = next(row for row in schedule if row["scene_id"] == single.SCENE)
    assert probe["user_prompt"] == reference["user_prompt"]
    assert probe["image_sha256"] == reference["image_sha256"]
    random.Random(ORDER_SEED).shuffle(schedule)
    config = {
        "experiment_version": "l2_direct_retrieval_25scenes_v15",
        "reference_config": str(reference_path),
        "reference_config_sha256": single.common.sha256_path(reference_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": single.common.sha256_path(source_manifest),
        "scene_count": 25, "scheduled_calls": 125, "seeds": list(single.SEEDS),
        "run_order_seed": ORDER_SEED,
        "model_id": reference["model_id"], "model_path": reference["model_path"],
        "generation": reference["generation"], "runtime": reference["runtime"],
        "system_prompt": system, "user_prompt_template": template,
        "schema": reference["schema"], "schema_sha256": reference["schema_sha256"],
        "geometry": None, "gripper_information_supplied": False,
        "evaluation": "descriptive_predictions_no_retrieval_execution_gt",
    }
    assert config["schema"] == single.SCHEMA
    single.common.write_json(DEST / "config/experiment_config.json", config)
    single.common.write_json(DEST / "config/frozen_scenes.json", scenes)
    single.common.write_json(DEST / "schemas/l2.json", config["schema"])
    prompt_dir = DEST / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system_en.txt").write_text(system + "\n")
    (prompt_dir / "l2_template_en.txt").write_text(template + "\n")
    schedule_path = DEST / "config/run_table.jsonl"
    with schedule_path.open("w", encoding="utf-8") as handle:
        for row in schedule:
            single.append(handle, row)
    single.common.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass", "checked_at": single.now(), "scene_count": 25,
        "run_table_rows": 125, "unique_run_ids": 125,
        "source_image_and_mapping_hashes_verified": True,
        "single_scene_prompt_image_and_settings_preserved": True,
        "run_table_sha256": single.common.sha256_path(schedule_path), "failures": [],
    })
    print(f"Preflight passed: 25 scenes x 5 seeds; loading {config['model_id']}", flush=True)
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
        "python": single.platform.python_version(),
        "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "model_load_seconds": round(time.perf_counter() - started, 3),
    }
    single.common.write_json(DEST / "logs/runtime.json", runtime)
    structured = single.StructuredOutputsParams(json=config["schema"])
    records, chats = [], {}
    with log_path.open("x", encoding="utf-8") as handle, (DEST / "logs/attempts.jsonl").open("x", encoding="utf-8") as attempts:
        for row in schedule:
            scene_id = row["scene_id"]
            if scene_id not in chats:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
                        {"type": "text", "text": row["user_prompt"]},
                    ]},
                ]
                chats[scene_id] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            request = {"prompt": chats[scene_id], "multi_modal_data": {"image": images[scene_id]}, "multi_modal_uuids": {"image": row["image_sha256"]}}
            sampling = single.SamplingParams(n=1, seed=row["seed"], structured_outputs=structured, **config["generation"])
            for attempt in range(1, 4):
                started_at = single.now()
                started = time.perf_counter()
                try:
                    output = engine.generate([request], sampling_params=[sampling], use_tqdm=False)[0]
                    completion = output.outputs[0]
                    break
                except Exception as exc:
                    single.append(attempts, {"run_id": row["run_id"], "seed": row["seed"], "attempt": attempt, "status": "infrastructure_error", "error": repr(exc), "started_at": started_at, "finished_at": single.now()})
                    if attempt == 3:
                        raise
            latency = time.perf_counter() - started
            raw, parsed, parse_error = completion.text, None, None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
            schema_valid = isinstance(parsed, dict) and set(parsed) == {"decision"} and parsed["decision"] in ("RETRIEVE_NOW", "REARRANGE_FIRST")
            record = {
                **row, "experiment_version": config["experiment_version"],
                "level": "L2", "geometry_condition": "C0", "geometry": None,
                "gripper_information_supplied": False,
                "model_id": config["model_id"], "model_path": config["model_path"],
                "generation": config["generation"], "system_prompt": system,
                "schema": config["schema"], "schema_sha256": config["schema_sha256"],
                "runtime": runtime, "attempt": attempt, "raw_response": raw,
                "parsed_response": parsed, "parse_error": parse_error, "schema_valid": schema_valid,
                "finish_reason": completion.finish_reason,
                "input_tokens": len(output.prompt_token_ids), "output_tokens": len(completion.token_ids),
                "latency_seconds": round(latency, 6), "started_at": started_at,
                "finished_at": single.now(), "task_correct": None,
            }
            single.append(attempts, {"run_id": row["run_id"], "seed": row["seed"], "attempt": attempt, "status": "response_received", "started_at": started_at, "finished_at": record["finished_at"], "raw_response": raw})
            single.append(handle, record)
            records.append(record)
            if len(records) == 1 or len(records) % 25 == 0:
                print(f"[{len(records)}/125] {scene_id} seed={row['seed']} {raw}", flush=True)
                single.common.write_json(DEST / "logs/progress.json", {"completed": len(records), "expected": 125, "updated_at": single.now()})
    by_scene, by_family = defaultdict(list), defaultdict(list)
    for record in records:
        by_scene[record["scene_id"]].append(record)
        by_family[record["scene_family"]].append(record)
    def counts(rows):
        tally = Counter(row["parsed_response"]["decision"] for row in rows if row["schema_valid"])
        return {"RETRIEVE_NOW": tally["RETRIEVE_NOW"], "REARRANGE_FIRST": tally["REARRANGE_FIRST"], "INVALID": sum(not row["schema_valid"] for row in rows)}
    scene_results = []
    for scene_id, rows in sorted(by_scene.items()):
        tally = counts(rows)
        responses = [{"seed": row["seed"], "decision": row["parsed_response"]["decision"] if row["schema_valid"] else "INVALID"} for row in sorted(rows, key=lambda row: row["seed"])]
        scene_results.append({"scene_id": scene_id, "scene_family": rows[0]["scene_family"], "target_object_id": rows[0]["target_object_id"], **tally, "unanimous": max(tally["RETRIEVE_NOW"], tally["REARRANGE_FIRST"]) == 5, "responses": responses})
    summary = {
        "model_id": config["model_id"], "scene_count": len(by_scene),
        "completed_calls": len(records), "scheduled_calls": 125,
        "schema_valid_calls": sum(record["schema_valid"] for record in records),
        "decision_counts": counts(records),
        "unanimous_scenes": sum(scene["unanimous"] for scene in scene_results),
        "mixed_scenes": [scene["scene_id"] for scene in scene_results if not scene["unanimous"]],
        "by_family": {family: counts(rows) for family, rows in sorted(by_family.items())},
        "by_scene": scene_results, "retrieval_execution_gt_available": False,
        "accuracy": None, "model_load_seconds": runtime["model_load_seconds"],
        "total_generation_seconds": round(sum(record["latency_seconds"] for record in records), 3),
        "finished_at": single.now(),
    }
    single.common.write_json(DEST / "results/summary.json", summary)
    with (DEST / "results/scene_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", "target_id", "RETRIEVE_NOW", "REARRANGE_FIRST", "INVALID", "unanimous"] + [str(seed) for seed in single.SEEDS])
        for scene in scene_results:
            writer.writerow([scene["scene_id"], scene["target_object_id"], scene["RETRIEVE_NOW"], scene["REARRANGE_FIRST"], scene["INVALID"], scene["unanimous"]] + [response["decision"] for response in scene["responses"]])
    lines = [
        "# L2 direct retrieval — v15, all 25 scenes", "",
        "Same system/user prompt, model, sampling settings, schema, and source image family as the single-scene v15 test. Only the target ID changes between prompts. No geometry or gripper information is supplied.", "",
        f"- Model: {config['model_id']}",
        f"- Calls: {len(records)}/125; seeds: {list(single.SEEDS)}; temperature: {config['generation']['temperature']}",
        f"- Decision counts: {summary['decision_counts']}",
        f"- Unanimous scenes: {summary['unanimous_scenes']}/25",
        "- These are model predictions, not measured retrieval accuracy. No matching execution GT was used.", "",
        "| Scene | Target | RETRIEVE_NOW | REARRANGE_FIRST | Invalid | Unanimous |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scene in scene_results:
        lines.append(f"| {scene['scene_id']} | {scene['target_object_id']} | {scene['RETRIEVE_NOW']} | {scene['REARRANGE_FIRST']} | {scene['INVALID']} | {scene['unanimous']} |")
    (DEST / "results/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "by_scene"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
