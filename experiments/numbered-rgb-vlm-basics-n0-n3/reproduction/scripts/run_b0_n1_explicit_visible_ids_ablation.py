#!/usr/bin/env python3
"""Run the frozen N1 explicit-visible-ID 3x2x2 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/b0_n1_explicit_visible_ids_ablation_60scenes_20260827"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def completed_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or run_id in rows:
                raise ValueError(f"invalid or duplicate run_id at line {line_number}: {run_id!r}")
            rows[run_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args()

    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    scenes = read_json(experiment / "config/frozen_scenes.json")
    schemas = {
        False: read_json(experiment / "config/json_schema_direct.json"),
        True: read_json(experiment / "config/json_schema_enumeration.json"),
    }
    system_prompt = (experiment / "prompts/system_prompt_en.txt").read_text(encoding="utf-8").strip()
    common_prompts = {
        False: (experiment / "prompts/direct_common_prompt_en.txt").read_text(encoding="utf-8").strip(),
        True: (experiment / "prompts/enumeration_common_prompt_en.txt").read_text(encoding="utf-8").strip(),
    }
    suffixes = {
        (False, "TEXT"): (experiment / "prompts/text_suffix_direct_en.txt").read_text(encoding="utf-8").strip(),
        (True, "TEXT"): (experiment / "prompts/text_suffix_enumeration_en.txt").read_text(encoding="utf-8").strip(),
        (False, "JSON_FORCED"): (experiment / "prompts/json_suffix_direct_en.txt").read_text(encoding="utf-8").strip(),
        (True, "JSON_FORCED"): (experiment / "prompts/json_suffix_enumeration_en.txt").read_text(encoding="utf-8").strip(),
    }
    output_path = (args.output or experiment / "logs/runs.jsonl").expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_name(output_path.stem + "_progress.json")
    completed = completed_records(output_path)

    schedule: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(config["main_seeds"], 1):
        for scene in scenes:
            for condition in config["conditions"]:
                schedule.append(
                    {
                        "run_id": f"{condition['condition_id']}__{scene['scene_id']}__r{repeat_index:02d}_s{seed}",
                        "scene": scene,
                        "condition": condition,
                        "repeat_index": repeat_index,
                        "seed": int(seed),
                    }
                )
    if len(schedule) != int(config["scheduled_main_calls"]):
        raise ValueError("schedule mismatch")
    pending = [row for row in schedule if row["run_id"] not in completed]
    if not pending:
        print(f"Nothing to run; completed={len(completed)} expected={len(schedule)}", flush=True)
        return

    generation = config["generation"]
    runtime_config = config["runtime"]
    model_path = Path(config["model_path"])
    gpu_memory = float(args.gpu_memory_utilization or runtime_config["gpu_memory_utilization"])
    print(f"Loading {model_path}; pending={len(pending)} completed={len(completed)} output={output_path}", flush=True)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    engine = LLM(
        model=str(model_path),
        runner="generate",
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory,
        max_model_len=int(runtime_config["max_model_len"]),
        max_num_seqs=1,
        enable_prefix_caching=True,
        moe_backend=runtime_config["moe_backend"],
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=str(experiment),
    )
    load_seconds = time.perf_counter() - load_started
    print(f"Model/engine loaded in {load_seconds:.1f}s", flush=True)

    runtime = {
        "python": platform.python_version(),
        "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "backend": "vllm_offline",
        "batch_size": 1,
        "max_model_len": int(runtime_config["max_model_len"]),
        "gpu_memory_utilization": gpu_memory,
        "moe_backend": runtime_config["moe_backend"],
    }
    image_cache: dict[Path, Image.Image] = {}
    prompt_cache: dict[tuple[bool, str], tuple[str, str]] = {}

    def image_for(path: Path) -> Image.Image:
        if path not in image_cache:
            image_cache[path] = Image.open(path).convert("RGB")
        return image_cache[path]

    def chat_for(enumeration: bool, output_format: str) -> tuple[str, str]:
        key = (enumeration, output_format)
        if key in prompt_cache:
            return prompt_cache[key]
        user_prompt = common_prompts[enumeration] + "\n\n" + suffixes[key]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": "file:///placeholder.png"}, {"type": "text", "text": user_prompt}]},
        ]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_cache[key] = (chat, sha256_text(system_prompt + "\n\n" + user_prompt))
        return prompt_cache[key]

    invocation_started = time.perf_counter()
    processed = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        for row in pending:
            scene = row["scene"]
            condition = row["condition"]
            output_format = condition["output_format"]
            enumeration = bool(condition["enumeration"])
            temperature = float(condition["temperature"])
            chat, prompt_hash = chat_for(enumeration, output_format)
            image_path = experiment / scene["image"]
            request = {
                "prompt": chat,
                "multi_modal_data": {"image": image_for(image_path)},
                "multi_modal_uuids": {"image": scene["image_sha256"]},
            }
            structured = StructuredOutputsParams(json=schemas[enumeration]) if output_format == "JSON_FORCED" else None
            sampling = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                presence_penalty=float(generation["presence_penalty"]),
                frequency_penalty=float(generation["frequency_penalty"]),
                max_tokens=int(generation["max_tokens"]),
                seed=int(row["seed"]),
                structured_outputs=structured,
            )
            started = time.perf_counter()
            outputs = engine.generate([request], sampling_params=[sampling], use_tqdm=False)
            wall_seconds = time.perf_counter() - started
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected number of outputs")
            request_output = outputs[0]
            completion = request_output.outputs[0]
            raw = completion.text.strip()
            record = {
                "run_id": row["run_id"],
                "phase": "full",
                "condition_id": condition["condition_id"],
                "temperature": temperature,
                "enumeration": enumeration,
                "output_format": output_format,
                "mode": output_format,
                "scene_id": scene["scene_id"],
                "level": scene["level"],
                "factors": scene["factors"],
                "ground_truth": scene["ground_truth"],
                "repeat_index": row["repeat_index"],
                "seed": row["seed"],
                "model_id": config["model_id"],
                "model_path": config["model_path"],
                "generation": {**generation, "temperature": temperature},
                "runtime": runtime,
                "image_path": scene["image"],
                "image_sha256": scene["image_sha256"],
                "prompt_sha256": prompt_hash,
                "json_schema_sha256": sha256_text(json.dumps(schemas[enumeration], sort_keys=True)) if output_format == "JSON_FORCED" else None,
                "raw_response": raw,
                "input_tokens": len(request_output.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "context_tokens_if_full_cap": len(request_output.prompt_token_ids) + int(generation["max_tokens"]),
                "wall_seconds": round(wall_seconds, 4),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if record["context_tokens_if_full_cap"] > int(runtime_config["max_model_len"]):
                raise RuntimeError(f"context budget violation for {row['run_id']}")
            output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output_handle.flush()
            completed[row["run_id"]] = record
            processed += 1
            progress = {
                "expected": len(schedule),
                "completed": len(completed),
                "processed_this_invocation": processed,
                "elapsed_seconds": round(time.perf_counter() - invocation_started, 3),
                "last_run_id": row["run_id"],
                "last_finish_reason": completion.finish_reason,
                "last_output_tokens": len(completion.token_ids),
            }
            progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            if processed == 1 or processed % 25 == 0 or len(completed) == len(schedule):
                print(
                    f"[{len(completed)}/{len(schedule)}] {row['run_id']} finish={completion.finish_reason} "
                    f"tokens={len(request_output.prompt_token_ids)}+{len(completion.token_ids)} wall={wall_seconds:.2f}s",
                    flush=True,
                )

    print(json.dumps({"completed": len(completed), "expected": len(schedule), "load_seconds": round(load_seconds, 3), "generation_seconds": round(time.perf_counter() - invocation_started, 3), "output": str(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
