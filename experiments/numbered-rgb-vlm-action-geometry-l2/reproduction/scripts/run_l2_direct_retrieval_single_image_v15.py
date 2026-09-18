#!/usr/bin/env python3
"""Run the agreed gripper-free L2 prompt on one exact source image, five seeds."""

from __future__ import annotations

import json
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10"
DEST = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_v15"
SCENE = "scene_lift_and_relocate_v05"
IMAGE = SOURCE / f"images/numbered_rgb/{SCENE}.png"
SEEDS = (28101, 28102, 28103, 28104, 28105)
SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE_NOW", "REARRANGE_FIRST"]}
    },
    "required": ["decision"],
    "additionalProperties": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> None:
    log_path = DEST / "logs/runs.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior results: {log_path}")
    previous = json.loads((ROOT / "experiments/vlm_action_geometry_single_info_l2_simple_obstruction_score_v14/config/experiment_config.json").read_text())
    mapping_path = SOURCE / f"images/id_mappings/{SCENE}.json"
    mapping = json.loads(mapping_path.read_text())
    assert mapping["target_object_id"] == 43
    system = (DEST / "prompts/system_en.txt").read_text().strip()
    user = (DEST / "prompts/l2_en.txt").read_text().strip()
    with Image.open(IMAGE) as source_image:
        rgb = source_image.convert("RGB")
    config = {
        "experiment_version": "l2_direct_retrieval_v15",
        "scene_id": SCENE,
        "target_object_id": 43,
        "image_path": str(IMAGE),
        "image_sha256": common.sha256_path(IMAGE),
        "image_size": list(rgb.size),
        "id_mapping_path": str(mapping_path),
        "id_mapping_sha256": common.sha256_path(mapping_path),
        "geometry": None,
        "gripper_information_supplied": False,
        "scheduled_calls": len(SEEDS),
        "seeds": list(SEEDS),
        "model_id": previous["model_id"],
        "model_path": previous["model_path"],
        "generation": previous["generation"],
        "runtime": previous["runtime"],
        "system_prompt": system,
        "user_prompt": user,
        "prompt_sha256": common.sha256_text(system + "\n\n" + user),
        "schema": SCHEMA,
        "schema_sha256": common.sha256_text(common.canonical_json(SCHEMA)),
        "evaluation": "descriptive_predictions_no_retrieval_execution_gt",
    }
    common.write_json(DEST / "config/experiment_config.json", config)
    common.write_json(DEST / "schemas/l2.json", SCHEMA)
    common.write_json(DEST / "audit/preflight_report.json", {
        "status": "pass", "checked_at": now(), "scheduled_calls": len(SEEDS),
        "image_sha256": config["image_sha256"], "target_id_verified_from_mapping": 43,
        "prompt_sha256": config["prompt_sha256"], "failures": [],
    })
    print(f"Loading {config['model_id']}; image={IMAGE}; target=43; calls={len(SEEDS)}", flush=True)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
    engine = LLM(
        model=config["model_path"], runner="generate", trust_remote_code=True,
        dtype="auto", tensor_parallel_size=1,
        gpu_memory_utilization=config["runtime"]["gpu_memory_utilization"],
        max_model_len=config["runtime"]["max_model_len"], max_num_seqs=1,
        enable_prefix_caching=True, moe_backend=config["runtime"]["moe_backend"],
        limit_mm_per_prompt={"image": 1},
    )
    load_seconds = time.perf_counter() - started
    runtime = {
        "python": platform.python_version(),
        "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "model_load_seconds": round(load_seconds, 3),
    }
    common.write_json(DEST / "logs/runtime.json", runtime)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
            {"type": "text", "text": user},
        ]},
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    structured = StructuredOutputsParams(json=SCHEMA)
    records = []
    with log_path.open("x", encoding="utf-8") as handle, (DEST / "logs/attempts.jsonl").open("x", encoding="utf-8") as attempts:
        for seed in SEEDS:
            sampling = SamplingParams(n=1, seed=seed, structured_outputs=structured, **config["generation"])
            request = {"prompt": chat, "multi_modal_data": {"image": rgb}, "multi_modal_uuids": {"image": config["image_sha256"]}}
            for attempt in range(1, 4):
                started_at = now()
                started = time.perf_counter()
                try:
                    output = engine.generate([request], sampling_params=[sampling], use_tqdm=False)[0]
                    completion = output.outputs[0]
                    break
                except Exception as exc:
                    append(attempts, {"seed": seed, "attempt": attempt, "started_at": started_at, "finished_at": now(), "status": "infrastructure_error", "error": repr(exc)})
                    if attempt == 3:
                        raise
            latency = time.perf_counter() - started
            raw = completion.text
            parsed, parse_error = None, None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
            schema_valid = isinstance(parsed, dict) and set(parsed) == {"decision"} and parsed["decision"] in SCHEMA["properties"]["decision"]["enum"]
            record = {
                **config, "run_id": f"v15__{SCENE}__L2_C0__s{seed}",
                "seed": seed, "attempt": attempt, "runtime_versions": runtime,
                "raw_response": raw, "parsed_response": parsed,
                "parse_error": parse_error, "schema_valid": schema_valid,
                "finish_reason": completion.finish_reason,
                "input_tokens": len(output.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "latency_seconds": round(latency, 6),
                "started_at": started_at, "finished_at": now(), "task_correct": None,
            }
            append(attempts, {"seed": seed, "attempt": attempt, "status": "response_received", "started_at": started_at, "finished_at": record["finished_at"], "raw_response": raw})
            append(handle, record)
            records.append(record)
            print(json.dumps({"seed": seed, "response": parsed, "schema_valid": schema_valid}, ensure_ascii=False), flush=True)
    summary = {
        "model_id": config["model_id"], "target_object_id": 43,
        "image_path": str(IMAGE), "image_sha256": config["image_sha256"],
        "completed_calls": len(records), "scheduled_calls": len(SEEDS),
        "schema_valid_calls": sum(record["schema_valid"] for record in records),
        "decision_counts": dict(Counter(record["parsed_response"]["decision"] for record in records if record["schema_valid"])),
        "responses": [{"seed": record["seed"], "raw_response": record["raw_response"]} for record in records],
        "retrieval_execution_gt_available": False, "accuracy": None,
        "model_load_seconds": round(load_seconds, 3),
        "total_generation_seconds": round(sum(record["latency_seconds"] for record in records), 6),
        "finished_at": now(),
    }
    common.write_json(DEST / "results/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
