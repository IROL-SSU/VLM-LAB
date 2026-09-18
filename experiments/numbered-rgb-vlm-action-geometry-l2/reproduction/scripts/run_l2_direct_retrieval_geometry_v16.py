#!/usr/bin/env python3
"""Frozen v15 retrieval prompt with the original twelve L2 geometry conditions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_l2_direct_retrieval_single_image_v15 as single


ROOT = single.ROOT
REFERENCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_25scenes_v15"
GEOMETRY_SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_maskfix_validation"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16"
VERSION = "l2_direct_retrieval_geometry_v16"
CONDITIONS = [
    ("C0", None, None, "RGB only"),
    ("C1", None, "r_numeric", "R Numeric"),
    ("C2", "D4", "r_qualitative_d4", "R Qualitative D4"),
    ("C2", "D8", "r_qualitative_d8", "R Qualitative D8"),
    ("C3", "D4", "f_numeric_d4", "F Numeric D4"),
    ("C3", "D8", "f_numeric_d8", "F Numeric D8"),
    ("C4", "D4", "f_qualitative_d4", "F Qualitative D4"),
    ("C4", "D8", "f_qualitative_d8", "F Qualitative D8"),
    ("C5", "D4", "m_numeric_d4", "M Numeric D4"),
    ("C5", "D8", "m_numeric_d8", "M Numeric D8"),
    ("C6", "D4", "m_qualitative_d4", "M Qualitative D4"),
    ("C6", "D8", "m_qualitative_d8", "M Qualitative D8"),
]
FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def identity(mapping):
    return sorted((row["display_id"], row["simulator_instance_id"], row["visible"]) for row in mapping["instances"])


def prepare():
    reference_path = REFERENCE / "config/experiment_config.json"
    reference = read(reference_path)
    template = reference["user_prompt_template"]
    assert template.count("Geometry information:\nnull") == 1
    template = template.replace("Geometry information:\nnull", "Geometry information:\n{geometry_json}")
    config = {
        "experiment_version": VERSION, "reference_config": str(reference_path),
        "reference_config_sha256": single.common.sha256_path(reference_path),
        "geometry_source": str(GEOMETRY_SOURCE),
        "scene_count": 25, "scheduled_calls": 1500, "seeds": list(single.SEEDS),
        "run_order_seed": 20260917,
        "model_id": reference["model_id"], "model_path": reference["model_path"],
        "generation": reference["generation"], "runtime": reference["runtime"],
        "system_prompt": reference["system_prompt"], "user_prompt_template": template,
        "schema": reference["schema"], "schema_sha256": reference["schema_sha256"],
        "conditions": [{"condition": c, "resolution": r, "directory": d, "name": n} for c, r, d, n in CONDITIONS],
        "gripper_information_supplied": False,
        "geometry_policy": "Original R/F/M payloads; ascending visible-object records only. R qualitative retains original 5cm clearance categories, not gripper dimensions.",
        "evaluation": "predictions_and_scene_design_agreement_not_physical_retrieval_accuracy",
    }
    config_path = DEST / "config/experiment_config.json"
    if config_path.exists():
        assert read(config_path) == config, "Frozen config changed"
        schedule = read_lines(DEST / "config/run_table.jsonl")
        audit = read(DEST / "audit/preflight_report.json")
        assert audit["run_table_sha256"] == single.common.sha256_path(DEST / "config/run_table.jsonl")
        assert len(schedule) == 1500
        return config, schedule
    scenes = read(REFERENCE / "config/frozen_scenes.json")
    assert len(scenes) == 25 and len({s["scene_id"] for s in scenes}) == 25
    geom_manifest = {scene["scene_id"]: scene for scene in read(GEOMETRY_SOURCE / "config/frozen_scenes.json")}
    previous_rows = {(row["scene_id"], row["seed"]): row for row in read_lines(REFERENCE / "logs/runs.jsonl")}
    schedule, payload_audit = [], []
    for scene in scenes:
        scene_id = scene["scene_id"]
        assert single.common.sha256_path(Path(scene["image_path"])) == scene["image_sha256"]
        assert single.common.sha256_path(Path(scene["id_mapping_path"])) == scene["id_mapping_sha256"]
        mapping = read(Path(scene["id_mapping_path"]))
        gm = geom_manifest[scene_id]
        assert single.common.sha256_path(GEOMETRY_SOURCE / gm["numbered_rgb"]) == scene["image_sha256"]
        assert identity(read(GEOMETRY_SOURCE / gm["id_mapping"])) == identity(mapping)
        assert mapping["target_object_id"] == scene["target_object_id"] == gm["target_object_id"]
        visible = {row["display_id"] for row in mapping["instances"] if row["visible"]}
        all_ids = {row["display_id"] for row in mapping["instances"]}
        target = scene["target_object_id"]
        assert target in visible
        for condition, resolution, directory, name in CONDITIONS:
            key = condition + ("_" + resolution if resolution else "")
            payload, source_path, payload_path = None, None, None
            source_hash, payload_hash, dropped = None, single.common.sha256_text("null"), []
            if directory:
                source_path = GEOMETRY_SOURCE / f"geometry/{directory}/{scene_id}.json"
                source_hash = single.common.sha256_path(source_path)
                payload = read(source_path)
                expected_kind = directory.upper().removesuffix("_D4").removesuffix("_D8")
                assert payload["condition"] == expected_kind
                if resolution:
                    assert payload["direction_resolution"] == resolution
                records_key = "relations" if condition in {"C1", "C2"} else "objects"
                original_rows = payload[records_key]
                assert all(row["object_id"] in all_ids for row in original_rows)
                dropped = sorted(row["object_id"] for row in original_rows if row["object_id"] not in visible)
                payload[records_key] = sorted((row for row in original_rows if row["object_id"] in visible), key=lambda row: row["object_id"])
                expected_ids = visible - {target} if records_key == "relations" else visible
                assert {row["object_id"] for row in payload[records_key]} == expected_ids
                assert len(payload[records_key]) == len(expected_ids)
                if records_key == "relations":
                    assert payload["target_id"] == target
                if condition == "C2":
                    assert all(row["clearance_zone"] in {"CLEARANCE_ZONE_OVERLAP", "CLEARANCE_ZONE_CLEAR"} for row in payload[records_key])
                serialized = json.dumps(payload).lower()
                assert not any(word in serialized for word in ("gripper", "finger_placement", "direct_graspable", "blocker_id", "best_action", "retrievable"))
                payload_path = DEST / f"geometry/{directory}/{scene_id}.json"
                single.common.write_json(payload_path, payload)
                payload_hash = single.common.sha256_path(payload_path)
            geometry_text = "null" if payload is None else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            user = template.replace("{target_object_id}", str(target)).replace("{geometry_json}", geometry_text)
            payload_audit.append({"scene_id": scene_id, "condition_key": key, "source_path": str(source_path) if source_path else None, "source_sha256": source_hash, "payload_path": str(payload_path) if payload_path else None, "payload_sha256": payload_hash, "excluded_fully_occluded_object_ids": dropped})
            for seed in single.SEEDS:
                if condition == "C0":
                    prior = previous_rows[(scene_id, seed)]
                    assert prior["user_prompt"] == user and prior["system_prompt"] == config["system_prompt"]
                    assert prior["generation"] == config["generation"]
                    assert prior["schema"] == config["schema"] and prior["image_sha256"] == scene["image_sha256"]
                schedule.append({
                    **scene, "run_id": f"v16__{scene_id}__L2__{key}__s{seed}", "seed": seed,
                    "level": "L2", "condition_key": key, "condition_name": name,
                    "geometry_condition": condition, "direction_resolution": resolution,
                    "geometry_path": str(payload_path) if payload_path else None,
                    "geometry_sha256": payload_hash, "geometry_json_raw": payload,
                    "user_prompt": user,
                    "prompt_sha256": single.common.sha256_text(config["system_prompt"] + "\n\n" + user),
                })
    assert len(schedule) == 1500 and len({row["run_id"] for row in schedule}) == 1500
    assert set(Counter(row["condition_key"] for row in schedule).values()) == {125}
    random.Random(config["run_order_seed"]).shuffle(schedule)
    single.common.write_json(DEST / "config/frozen_scenes.json", scenes)
    single.common.write_json(DEST / "schemas/l2.json", config["schema"])
    prompt_dir = DEST / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system_en.txt").write_text(config["system_prompt"] + "\n")
    (prompt_dir / "l2_template_en.txt").write_text(template + "\n")
    schedule_path = DEST / "config/run_table.jsonl"
    with schedule_path.open("w", encoding="utf-8") as handle:
        for row in schedule:
            single.append(handle, row)
    single.common.write_json(DEST / "audit/geometry_payloads.json", payload_audit)
    single.common.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass", "checked_at": single.now(), "scene_count": 25,
        "scheduled_calls": 1500, "unique_run_ids": 1500,
        "condition_counts": dict(Counter(row["condition_key"] for row in schedule)),
        "run_table_sha256": single.common.sha256_path(schedule_path),
        "c0_exact_input_checks_passed": 125,
        "geometry_image_hashes_and_instance_correspondence_verified": True,
        "geometry_policy": config["geometry_policy"],
        "filtered_payload_count": sum(bool(item["excluded_fully_occluded_object_ids"]) for item in payload_audit),
        "failures": [],
    })
    single.common.write_json(config_path, config)
    return config, schedule


def infer(config, schedule):
    log_path = DEST / "logs/runs.jsonl"
    records = read_lines(log_path)
    done = {row["run_id"]: row for row in records}
    assert len(done) == len(records)
    planned = {row["run_id"]: row for row in schedule}
    assert set(done).issubset(planned)
    for run_id, row in done.items():
        assert row["prompt_sha256"] == planned[run_id]["prompt_sha256"]
    pending = [row for row in schedule if row["run_id"] not in done]
    if not pending:
        return records
    images = {}
    for row in pending:
        if row["scene_id"] not in images:
            assert single.common.sha256_path(Path(row["image_path"])) == row["image_sha256"]
            with single.Image.open(row["image_path"]) as image:
                images[row["scene_id"]] = image.convert("RGB")
        if row["geometry_path"]:
            assert single.common.sha256_path(Path(row["geometry_path"])) == row["geometry_sha256"]
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
        "model_load_seconds": round(time.perf_counter() - started, 3), "started_at": single.now(),
    }
    single.common.write_json(DEST / "logs/runtime.json", runtime)
    structured = single.StructuredOutputsParams(json=config["schema"])
    chats = {}
    with log_path.open("a", encoding="utf-8") as handle, (DEST / "logs/attempts.jsonl").open("a", encoding="utf-8") as attempts:
        for row in pending:
            prompt_key = row["prompt_sha256"]
            if prompt_key not in chats:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": config["system_prompt"]}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
                        {"type": "text", "text": row["user_prompt"]},
                    ]},
                ]
                chats[prompt_key] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            request = {"prompt": chats[prompt_key], "multi_modal_data": {"image": images[row["scene_id"]]}, "multi_modal_uuids": {"image": row["image_sha256"]}}
            sampling = single.SamplingParams(n=1, seed=row["seed"], structured_outputs=structured, **config["generation"])
            for attempt in range(1, 4):
                started_at, started = single.now(), time.perf_counter()
                try:
                    output = engine.generate([request], sampling_params=[sampling], use_tqdm=False)[0]
                    completion = output.outputs[0]
                    break
                except Exception as exc:
                    single.append(attempts, {"run_id": row["run_id"], "seed": row["seed"], "attempt": attempt, "status": "infrastructure_error", "error": repr(exc), "started_at": started_at, "finished_at": single.now()})
                    if attempt == 3:
                        raise
            raw, parsed, parse_error = completion.text, None, None
            latency = time.perf_counter() - started
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
            valid = isinstance(parsed, dict) and set(parsed) == {"decision"} and parsed["decision"] in ("RETRIEVE_NOW", "REARRANGE_FIRST")
            record = {
                **row, "experiment_version": VERSION, "model_id": config["model_id"],
                "model_path": config["model_path"], "generation": config["generation"],
                "system_prompt": config["system_prompt"], "schema": config["schema"],
                "schema_sha256": config["schema_sha256"], "runtime": runtime,
                "gripper_information_supplied": False, "raw_response": raw,
                "parsed_response": parsed, "parse_error": parse_error, "schema_valid": valid,
                "finish_reason": completion.finish_reason, "attempt": attempt,
                "input_tokens": len(output.prompt_token_ids), "output_tokens": len(completion.token_ids),
                "latency_seconds": round(latency, 6), "started_at": started_at,
                "finished_at": single.now(), "task_correct": None,
            }
            assert record["input_tokens"] + config["generation"]["max_tokens"] <= config["runtime"]["max_model_len"]
            single.append(attempts, {"run_id": row["run_id"], "seed": row["seed"], "attempt": attempt, "status": "response_received", "raw_response": raw, "started_at": started_at, "finished_at": record["finished_at"]})
            single.append(handle, record)
            records.append(record)
            if len(records) == 1 or len(records) % 100 == 0:
                print(f"[{len(records)}/1500] {row['condition_key']} {row['scene_id']} seed={row['seed']} {raw}", flush=True)
                single.common.write_json(DEST / "logs/progress.json", {"completed": len(records), "expected": 1500, "updated_at": single.now()})
    return records


def counts(rows):
    tally = Counter(row["parsed_response"]["decision"] for row in rows if row["schema_valid"])
    return {"RETRIEVE_NOW": tally["RETRIEVE_NOW"], "REARRANGE_FIRST": tally["REARRANGE_FIRST"], "INVALID": sum(not row["schema_valid"] for row in rows)}


def expected_design(row):
    return "RETRIEVE_NOW" if row["scene_family"] == "FC_CLEAR" else "REARRANGE_FIRST"


def decision(row):
    return row["parsed_response"]["decision"] if row["schema_valid"] else "INVALID"


def analyze(config, schedule, records):
    assert len(records) == len(schedule) == 1500
    assert {row["run_id"] for row in records} == {row["run_id"] for row in schedule}
    by_condition, by_scene = defaultdict(list), defaultdict(list)
    for row in records:
        by_condition[row["condition_key"]].append(row)
        by_scene[(row["scene_id"], row["condition_key"])].append(row)
    baseline = {(row["scene_id"], row["seed"]): row for row in by_condition["C0"]}
    prior = {(row["scene_id"], row["seed"]): row for row in read_lines(REFERENCE / "logs/runs.jsonl")}
    prior_changes = [key for key, row in baseline.items() if decision(row) != decision(prior[key])]
    condition_results, scene_results, paired_changes = [], [], []
    for c, r, _, name in CONDITIONS:
        key = c + ("_" + r if r else "")
        rows = by_condition[key]
        assert len(rows) == 125
        families = {family: counts([row for row in rows if row["scene_family"] == family]) for family in FAMILIES}
        mismatch = sum(decision(row) != expected_design(row) for row in rows)
        unanimous = 0
        for scene_id in sorted({row["scene_id"] for row in rows}):
            scene_rows = by_scene[(scene_id, key)]
            assert len(scene_rows) == 5 and {row["seed"] for row in scene_rows} == set(single.SEEDS)
            tally = counts(scene_rows)
            unanimous += max(tally["RETRIEVE_NOW"], tally["REARRANGE_FIRST"]) == 5
            scene_results.append({"scene_id": scene_id, "condition_key": key, "target_object_id": scene_rows[0]["target_object_id"], "scene_family": scene_rows[0]["scene_family"], **tally, "responses": [{"seed": row["seed"], "decision": decision(row)} for row in sorted(scene_rows, key=lambda row: row["seed"])]})
        changed, to_design, away_design = 0, 0, 0
        for row in rows:
            base = baseline[(row["scene_id"], row["seed"])]
            if decision(row) != decision(base):
                changed += 1
                to_design += decision(row) == expected_design(row)
                away_design += decision(base) == expected_design(row)
                paired_changes.append({"scene_id": row["scene_id"], "seed": row["seed"], "condition_key": key, "c0_decision": decision(base), "condition_decision": decision(row), "scene_design_reference": expected_design(row)})
        condition_results.append({
            "condition_key": key, "condition_name": name, "n": 125, **counts(rows),
            "by_family": families, "unanimous_scenes": unanimous,
            "changed_vs_c0": changed, "changes_toward_scene_design": to_design,
            "changes_away_from_scene_design": away_design,
            "scene_design_agreement_count": 125 - mismatch,
            "scene_design_agreement_rate": (125 - mismatch) / 125,
        })
    summary = {
        "experiment_version": VERSION, "model_id": config["model_id"],
        "completed_calls": 1500, "scene_count": 25, "condition_count": 12,
        "schema_valid_calls": sum(row["schema_valid"] for row in records),
        "infrastructure_errors": sum(row["status"] == "infrastructure_error" for row in read_lines(DEST / "logs/attempts.jsonl")),
        "retrieval_execution_gt_available": False, "accuracy": None,
        "scene_design_reference": "FC_CLEAR => RETRIEVE_NOW; other families => REARRANGE_FIRST. Advisory scene intent, not execution GT.",
        "condition_results": condition_results,
        "c0_changes_vs_previous_v15": [{"scene_id": key[0], "seed": key[1]} for key in prior_changes],
        "total_generation_seconds": round(sum(row["latency_seconds"] for row in records), 3),
        "input_token_range": [min(row["input_tokens"] for row in records), max(row["input_tokens"] for row in records)],
        "finished_at": single.now(),
    }
    single.common.write_json(DEST / "results/summary.json", summary)
    single.common.write_json(DEST / "results/scene_results.json", scene_results)
    single.common.write_json(DEST / "results/paired_changes_vs_c0.json", paired_changes)
    keys = [c + ("_" + r if r else "") for c, r, _, _ in CONDITIONS]
    results_by_key = {(row["scene_id"], row["condition_key"]): row for row in scene_results}
    with (DEST / "results/retrieve_counts_by_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_id", "target_object_id"] + keys)
        for scene_id in sorted({row["scene_id"] for row in records}):
            writer.writerow([scene_id, results_by_key[(scene_id, "C0")]["target_object_id"]] + [results_by_key[(scene_id, key)]["RETRIEVE_NOW"] for key in keys])
    with (DEST / "results/condition_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        fields = ["condition_key", "condition_name", "RETRIEVE_NOW", "REARRANGE_FIRST", "INVALID", "unanimous_scenes", "changed_vs_c0", "changes_toward_scene_design", "changes_away_from_scene_design", "scene_design_agreement_count"]
        writer.writerow(fields)
        for row in condition_results:
            writer.writerow([row[field] for field in fields])
    lines = [
        "# L2 direct retrieval v16 — geometry conditions", "",
        "25 identical numbered images × 12 conditions × 5 seeds = 1,500 independent calls. The v15 system prompt, task wording, model, sampling settings and output schema are unchanged; only the Geometry information block changes. No gripper data is supplied.", "",
        "Geometry uses the original R/F/M definitions from maskfix_validation, matched by image hash and simulator/display-ID correspondence. Only visible-object records are supplied. R-Qualitative includes the original 5 cm clearance-zone categories. The single fully hidden object (lift_v04, ID 16) is excluded from payload records; measurements of visible objects are otherwise reused unchanged.", "",
        "These are predictions. Scene-design agreement is descriptive agreement with FC_CLEAR→RETRIEVE_NOW and other scene families→REARRANGE_FIRST, NOT physical retrieval accuracy. In particular, FC_BLOCKED scenes were designed with finger-clearance assumptions that are not included in this prompt.", "",
        f"- Schema-valid calls: {summary['schema_valid_calls']}/1500",
        f"- Infrastructure errors: {summary['infrastructure_errors']}",
        f"- C0 responses changed versus previous v15: {len(prior_changes)}/125", "",
        "| Condition | Retrieve | Rearrange | FC_CLEAR retrieve /25 | FC_BLOCKED rearrange /25 | Other families rearrange /75 | Unanimous scenes /25 | Changed vs C0 /125 | Scene-design agreement /125 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in condition_results:
        family = row["by_family"]
        other = sum(family[name]["REARRANGE_FIRST"] for name in ("TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"))
        lines.append(f"| {row['condition_key']} {row['condition_name']} | {row['RETRIEVE_NOW']} | {row['REARRANGE_FIRST']} | {family['FC_CLEAR']['RETRIEVE_NOW']} | {family['FC_BLOCKED']['REARRANGE_FIRST']} | {other} | {row['unanimous_scenes']} | {row['changed_vs_c0']} | {row['scene_design_agreement_count']} |")
    lines += ["", "## Per-scene RETRIEVE_NOW counts (out of five seeds)", "", "| Scene | Target | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
    for scene_id in sorted({row["scene_id"] for row in records}):
        target = results_by_key[(scene_id, "C0")]["target_object_id"]
        lines.append(f"| {scene_id} | {target} | " + " | ".join(str(results_by_key[(scene_id, key)]["RETRIEVE_NOW"]) for key in keys) + " |")
    (DEST / "results/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    single.common.write_json(DEST / "audit/completion_report.json", {
        "status": "pass", "completed": len(records), "unique_run_ids": len({row["run_id"] for row in records}),
        "condition_counts": dict(Counter(row["condition_key"] for row in records)),
        "all_scene_condition_seed_sets_complete": True,
        "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
        "logs_sha256": single.common.sha256_path(DEST / "logs/runs.jsonl"),
        "finished_at": single.now(),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config, schedule = prepare()
    if args.prepare_only:
        print(json.dumps(read(DEST / "audit/preflight_report.json"), indent=2))
        return
    records = read_lines(DEST / "logs/runs.jsonl") if args.analyze_only else infer(config, schedule)
    analyze(config, schedule, records)


if __name__ == "__main__":
    main()
