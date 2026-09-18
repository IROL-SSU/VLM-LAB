#!/usr/bin/env python3
"""Run the frozen 6,250-call Qwen3-VL action/geometry v1 experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def completed_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    output = {}
    for line_number, row in enumerate(read_jsonl(path), 1):
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id in output:
            raise ValueError(f"Invalid or duplicate run_id at {path}:{line_number}: {run_id!r}")
        output[run_id] = row
    return output


def append_jsonl(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def build_user_prompt(row: dict, geometry: object, task_prompt: str) -> str:
    geometry_text = "null" if geometry is None else json.dumps(
        geometry, ensure_ascii=False, indent=2, sort_keys=True
    )
    task = task_prompt
    if "candidate_description" in row:
        task = task.replace("{candidate_description}", row["candidate_description"])
    if row.get("natural_language_only"):
        return task
    if row["level"] == "L4":
        directions = json.dumps(list(common.DIRECTIONS[row["direction_resolution"]]))
        task = task.replace("{direction_list}", directions)
    return (
        f"Target object ID: {row['target_object_id']}\n\n"
        f"Geometry information:\n{geometry_text}\n\n{task}"
    )


def semantic_validate(level: str, parsed: object, row: dict, mapping: dict) -> list[str]:
    errors = []
    if not isinstance(parsed, dict):
        return ["response is not an object"]
    visible_ids = {
        int(instance["display_id"])
        for instance in mapping["instances"]
        if instance["visible"]
    }
    target_id = int(row["target_object_id"])
    if level == "L1":
        if parsed.get("target_found") is False and parsed.get("visibility") != "NOT_VISIBLE":
            errors.append("target_found=false requires visibility=NOT_VISIBLE")
        if parsed.get("target_found") is True and parsed.get("visibility") == "NOT_VISIBLE":
            errors.append("visibility=NOT_VISIBLE requires target_found=false")
    elif level == "L2" and "grasp_obstruction_assessments_left_to_right" in parsed:
        visible_rows_ltr = sorted(
            (instance for instance in mapping["instances"] if instance["visible"]),
            key=lambda instance: (
                instance["visible_centroid_xy"][0], int(instance["display_id"])
            ),
        )
        assessment_ids_ltr = [
            int(instance["display_id"])
            for instance in visible_rows_ltr
            if int(instance["display_id"]) != target_id
        ]
        assessments = parsed.get("grasp_obstruction_assessments_left_to_right")
        if parsed.get("target_id") != target_id:
            errors.append("target_id does not equal target_object_id")
        if not isinstance(assessments, list):
            errors.append("grasp_obstruction_assessments_left_to_right is not an array")
        else:
            output_ids = [
                item.get("object_id") for item in assessments if isinstance(item, dict)
            ]
            if len(output_ids) != len(assessments):
                errors.append("every grasp-obstruction assessment must be an object")
            if output_ids != assessment_ids_ltr:
                errors.append("grasp-obstruction assessments do not match left-to-right IDs")
            for item in assessments:
                if isinstance(item, dict) and not isinstance(item.get("obstructs_grasp"), bool):
                    errors.append("obstructs_grasp must be Boolean")
    elif level == "L2" and "occlusion_assessments_left_to_right" in parsed:
        visible_rows_ltr = sorted(
            (instance for instance in mapping["instances"] if instance["visible"]),
            key=lambda instance: (
                instance["visible_centroid_xy"][0], int(instance["display_id"])
            ),
        )
        visible_ids_ltr = [int(instance["display_id"]) for instance in visible_rows_ltr]
        assessment_ids_ltr = [
            object_id for object_id in visible_ids_ltr if object_id != target_id
        ]
        output_visible_ids = parsed.get("visible_object_ids_left_to_right")
        assessments = parsed.get("occlusion_assessments_left_to_right")
        if output_visible_ids != visible_ids_ltr:
            errors.append("visible object IDs are not in the required left-to-right order")
        if parsed.get("target_id") != target_id:
            errors.append("target_id does not equal target_object_id")
        if not isinstance(assessments, list):
            errors.append("occlusion_assessments_left_to_right is not an array")
        else:
            output_ids = [
                item.get("object_id")
                for item in assessments
                if isinstance(item, dict)
            ]
            if len(output_ids) != len(assessments):
                errors.append("every occlusion assessment must be an object")
            if output_ids != assessment_ids_ltr:
                errors.append("occlusion assessments do not match non-target left-to-right IDs")
            for item in assessments:
                if isinstance(item, dict) and not isinstance(item.get("occludes_target"), bool):
                    errors.append("occludes_target must be Boolean")
    elif level == "L2" and "instance_assessments" in parsed:
        assessments = parsed.get("instance_assessments")
        if parsed.get("target_id") != target_id:
            errors.append("target_id does not equal target_object_id")
        if not isinstance(assessments, list):
            errors.append("instance_assessments is not an array")
        else:
            output_ids = [
                item.get("object_id")
                for item in assessments
                if isinstance(item, dict)
            ]
            if len(output_ids) != len(assessments):
                errors.append("every assessment must be an object")
            if len(assessments) != len(visible_ids):
                errors.append("assessment count does not equal numbered object count")
            if len(output_ids) != len(set(output_ids)):
                errors.append("object_id values are not unique")
            if set(output_ids) != visible_ids:
                errors.append("assessment object_id set does not equal visible display ID set")
            if output_ids != sorted(output_ids):
                errors.append("assessments are not in ascending object_id order")
            for item in assessments:
                if not isinstance(item, dict):
                    continue
                object_id = item.get("object_id")
                relation = item.get("relation_to_target")
                if object_id == target_id and relation != "TARGET":
                    errors.append("target object relation must be TARGET")
                if object_id != target_id and relation not in {
                    "INTERFERES", "DOES_NOT_INTERFERE"
                }:
                    errors.append("non-target relation is invalid")
    elif level == "L2" and "object_to_remove_id" in parsed:
        blocker = parsed.get("object_to_remove_id")
        if blocker is not None and not isinstance(blocker, int):
            errors.append("object_to_remove_id must be an integer or null")
        elif isinstance(blocker, int):
            if blocker not in visible_ids:
                errors.append("object_to_remove_id is not a visible display ID")
            if blocker == target_id:
                errors.append("object_to_remove_id cannot equal target_object_id")
    elif level == "L3" and "blocker_ids" in parsed:
        reason = parsed.get("blocking_reason")
        blockers = parsed.get("blocker_ids")
        if not isinstance(blockers, list):
            errors.append("blocker_ids must be an array")
        else:
            integer_blockers = [blocker for blocker in blockers if type(blocker) is int]
            if len(integer_blockers) != len(blockers):
                errors.append("every blocker_ids item must be an integer")
            if len(integer_blockers) != len(set(integer_blockers)):
                errors.append("blocker_ids values are not unique")
            if len(integer_blockers) == len(blockers) and blockers != sorted(blockers):
                errors.append("blocker_ids are not in ascending order")
            for blocker in blockers:
                if type(blocker) is not int:
                    continue
                if blocker not in visible_ids:
                    errors.append("blocker_ids contains a non-visible display ID")
                if blocker == target_id:
                    errors.append("blocker_ids cannot contain target_object_id")
            if reason == "NONE" and blockers:
                errors.append("blocking_reason=NONE requires blocker_ids=[]")
            if reason != "NONE" and not blockers:
                errors.append("non-NONE blocking_reason requires non-empty blocker_ids")
    elif level == "L3":
        reason = parsed.get("blocking_reason")
        blocker = parsed.get("blocker_id")
        if reason == "NONE" and blocker is not None:
            errors.append("blocking_reason=NONE requires blocker_id=null")
        if reason != "NONE" and blocker is None:
            errors.append("non-NONE blocking_reason requires integer blocker_id")
        if isinstance(blocker, int):
            if blocker not in visible_ids:
                errors.append("blocker_id is not a visible display ID")
            if blocker == target_id:
                errors.append("blocker_id cannot equal target_object_id")
    elif level == "L4":
        action = parsed.get("action")
        object_id = parsed.get("object_id")
        if isinstance(object_id, int) and object_id not in visible_ids:
            errors.append("object_id is not a visible display ID")
        if action == "RETRIEVE" and object_id != target_id:
            errors.append("RETRIEVE object_id must equal target_object_id")
        if action in {"TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"} and object_id == target_id:
            errors.append(f"{action} cannot manipulate the target in this benchmark")
        if action == "TRANSLATE":
            distance = parsed.get("distance_cm")
            label = parsed.get("distance_level")
            if isinstance(distance, int) and common.translation_level(distance) != label:
                errors.append("distance_level is inconsistent with distance_cm")
        if action == "ROTATE":
            angle = parsed.get("angle_deg")
            label = parsed.get("angle_level")
            if isinstance(angle, int) and common.rotation_level(angle) != label:
                errors.append("angle_level is inconsistent with angle_deg")
    return errors


def score_non_action(level: str, parsed: object, gt: dict) -> dict:
    if not isinstance(parsed, dict):
        return {"task_correct": False}
    if level == "L1":
        target_ok = parsed.get("target_found") == gt["L1"]["target_found"]
        visibility_ok = parsed.get("visibility") == gt["L1"]["visibility"]
        return {
            "target_found_correct": target_ok,
            "visibility_correct": visibility_ok,
            "task_correct": target_ok and visibility_ok,
        }
    if level == "L2":
        if "obstruction_score" in parsed:
            return {
                "task_correct": None,
                "evaluation": "descriptive_score_without_action_level_scalar_gt",
            }
        if "obstructs_grasp" in gt["L2"]:
            correct = parsed.get("obstructs_grasp") == gt["L2"]["obstructs_grasp"]
            return {
                "obstructs_grasp_correct": correct,
                "task_correct": correct,
            }
        if "grasp_obstruction_assessments_left_to_right" in gt["L2"]:
            expected = {
                int(item["object_id"]): bool(item["obstructs_grasp"])
                for item in gt["L2"]["grasp_obstruction_assessments_left_to_right"]
            }
            actual_rows = parsed.get("grasp_obstruction_assessments_left_to_right")
            actual = {}
            if isinstance(actual_rows, list):
                for item in actual_rows:
                    if isinstance(item, dict) and isinstance(item.get("object_id"), int):
                        actual[item["object_id"]] = item.get("obstructs_grasp")
            correct_count = sum(
                actual.get(object_id) == value for object_id, value in expected.items()
            )
            exact = actual == expected and parsed.get("target_id") == gt["L2"]["target_id"]
            return {
                "grasp_obstruction_exact_match": exact,
                "correct_grasp_obstruction_count": correct_count,
                "grasp_obstruction_candidate_count": len(expected),
                "task_correct": exact,
            }
        if "object_to_remove_id" in gt["L2"]:
            expected = gt["L2"]["object_to_remove_id"]
            actual = parsed.get("object_to_remove_id")
            correct = actual == expected
            return {
                "object_to_remove_id_correct": correct,
                "blocker_presence_correct": (actual is None) == (expected is None),
                "task_correct": correct,
            }
        if "occlusion_assessments_left_to_right" in gt["L2"]:
            expected_rows = gt["L2"]["occlusion_assessments_left_to_right"]
            expected = {
                int(item["object_id"]): bool(item["occludes_target"])
                for item in expected_rows
            }
            actual_rows = parsed.get("occlusion_assessments_left_to_right")
            actual = {}
            if isinstance(actual_rows, list):
                for item in actual_rows:
                    if isinstance(item, dict) and isinstance(item.get("object_id"), int):
                        actual[item["object_id"]] = item.get("occludes_target")
            correct_count = sum(
                actual.get(object_id) == value for object_id, value in expected.items()
            )
            exact = (
                actual == expected
                and parsed.get("target_id") == gt["L2"]["target_id"]
                and parsed.get("visible_object_ids_left_to_right")
                == gt["L2"]["visible_object_ids_left_to_right"]
            )
            expected_blockers = {
                object_id for object_id, value in expected.items() if value
            }
            actual_blockers = {
                object_id for object_id, value in actual.items() if value is True
            }
            return {
                "occlusion_instance_exact_match": exact,
                "correct_occlusion_count": correct_count,
                "occlusion_candidate_count": len(expected),
                "occluder_set_exact_match": actual_blockers == expected_blockers,
                "task_correct": exact,
            }
        if "instance_assessments" in gt["L2"]:
            expected = {
                int(item["object_id"]): item["relation_to_target"]
                for item in gt["L2"]["instance_assessments"]
            }
            actual = {}
            assessments = parsed.get("instance_assessments")
            if isinstance(assessments, list):
                for item in assessments:
                    if isinstance(item, dict) and isinstance(item.get("object_id"), int):
                        actual[item["object_id"]] = item.get("relation_to_target")
            correct_count = sum(
                actual.get(object_id) == relation
                for object_id, relation in expected.items()
            )
            exact = actual == expected and parsed.get("target_id") == gt["L2"]["target_id"]
            expected_blockers = {
                object_id for object_id, relation in expected.items()
                if relation == "INTERFERES"
            }
            actual_blockers = {
                object_id for object_id, relation in actual.items()
                if relation == "INTERFERES"
            }
            return {
                "instance_exact_match": exact,
                "correct_instance_count": correct_count,
                "instance_count": len(expected),
                "blocker_set_exact_match": actual_blockers == expected_blockers,
                "task_correct": exact,
            }
        if "potential_grasp_interference" in gt["L2"]:
            correct = (
                parsed.get("potential_grasp_interference")
                == gt["L2"]["potential_grasp_interference"]
            )
            return {
                "potential_grasp_interference_correct": correct,
                "task_correct": correct,
            }
        if "target_access_blocked" in gt["L2"]:
            correct = (
                parsed.get("target_access_blocked")
                == gt["L2"]["target_access_blocked"]
            )
            return {
                "target_access_blocked_correct": correct,
                "task_correct": correct,
            }
        if "occluded_by_other_object" in gt["L2"]:
            correct = (
                parsed.get("occluded_by_other_object")
                == gt["L2"]["occluded_by_other_object"]
            )
            return {
                "occluded_by_other_object_correct": correct,
                "task_correct": correct,
            }
        correct = parsed.get("direct_graspable") == gt["L2"]["direct_graspable"]
        return {"direct_graspable_correct": correct, "task_correct": correct}
    if level == "L3":
        reason_ok = parsed.get("blocking_reason") == gt["L3"]["blocking_reason"]
        if "all_blocker_ids" in gt["L3"]:
            expected = set(gt["L3"]["all_blocker_ids"])
            blockers = parsed.get("blocker_ids")
            actual = {
                blocker for blocker in blockers
                if type(blocker) is int
            } if isinstance(blockers, list) else set()
            true_positive = len(actual & expected)
            precision = (
                true_positive / len(actual)
                if actual else float(not expected)
            )
            recall = (
                true_positive / len(expected)
                if expected else float(not actual)
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            exact = actual == expected and isinstance(blockers, list)
            return {
                "blocking_reason_correct": reason_ok,
                "blocker_set_exact_match": exact,
                "blocker_presence_correct": bool(actual) == bool(expected),
                "blocker_true_positive_count": true_positive,
                "blocker_predicted_count": len(actual),
                "blocker_expected_count": len(expected),
                "blocker_precision": precision,
                "blocker_recall": recall,
                "blocker_f1": f1,
                "task_correct": reason_ok and exact,
            }
        blocker = parsed.get("blocker_id")
        if gt["L3"]["blocking_reason"] == "NONE":
            blocker_ok = blocker is None
        else:
            blocker_ok = blocker in gt["L3"]["valid_blocker_ids"]
        return {
            "blocking_reason_correct": reason_ok,
            "blocker_valid_set_member": blocker_ok,
            "task_correct": reason_ok and blocker_ok,
        }
    return {"task_correct": None, "replay_pending": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument(
        "--level",
        action="append",
        choices=("L1", "L2", "L3", "L4"),
        help="Run only the selected level; repeat to select multiple levels.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    preflight = read_json(experiment / "audit/preflight_report.json")
    if preflight.get("status") != "pass":
        raise RuntimeError("Preparation preflight did not pass")
    run_table_path = experiment / "config/run_table.jsonl"
    if common.sha256_path(run_table_path) != preflight["run_table_sha256"]:
        raise RuntimeError("Frozen run table hash changed after preflight")
    full_schedule = read_jsonl(run_table_path)
    if len(full_schedule) != int(config["scheduled_calls"]):
        raise RuntimeError(f"Schedule mismatch: {len(full_schedule)}")
    selected_levels = set(args.level or [])
    schedule = [
        row for row in full_schedule
        if not selected_levels or row["level"] in selected_levels
    ]
    if not schedule:
        raise RuntimeError("Run filters selected no scheduled calls")

    output_path = (args.output or experiment / "logs/runs.jsonl").expanduser().resolve()
    attempts_path = output_path.with_name("attempts.jsonl")
    progress_path = output_path.with_name("progress.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_records(output_path)
    pending = [row for row in schedule if row["run_id"] not in completed]
    if args.max_new_calls is not None:
        if args.max_new_calls <= 0:
            raise ValueError("--max-new-calls must be positive")
        pending = pending[: args.max_new_calls]
    if not pending:
        print(f"Nothing to run; completed={len(completed)} expected={len(schedule)}", flush=True)
        return

    generation = config["generation"]
    runtime_config = config["runtime"]
    model_path = Path(config["model_path"])
    gpu_memory = float(args.gpu_memory_utilization or runtime_config["gpu_memory_utilization"])
    system_prompt = (experiment / config["prompts"]["system"]).read_text(encoding="utf-8").strip()
    task_prompts = {
        level: (experiment / config["prompts"][level.lower()]).read_text(encoding="utf-8").strip()
        for level in ("L1", "L2", "L3", "L4")
    }
    print(
        f"Loading {model_path}; pending_this_invocation={len(pending)} "
        f"completed={len(completed)} expected={len(schedule)}",
        flush=True,
    )
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    engine = LLM(
        model=str(model_path), runner="generate", trust_remote_code=True,
        dtype="auto", tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory,
        max_model_len=int(runtime_config["max_model_len"]),
        max_num_seqs=1, enable_prefix_caching=True,
        moe_backend=runtime_config["moe_backend"],
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=str(experiment),
    )
    load_seconds = time.perf_counter() - load_started
    runtime = {
        "python": platform.python_version(),
        "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "backend": "vllm_offline", "batch_size": 1, "max_num_seqs": 1,
        "max_model_len": int(runtime_config["max_model_len"]),
        "gpu_memory_utilization": gpu_memory,
        "moe_backend": runtime_config["moe_backend"],
        "structured_output": runtime_config["structured_output"],
    }
    common.write_json(output_path.parent / "runtime.json", {
        **runtime, "model_load_seconds": round(load_seconds, 3),
        "invocation_started_at": utc_now(), "model_id": config["model_id"],
        "model_path": str(model_path),
    })
    print(f"Model loaded in {load_seconds:.1f}s", flush=True)

    image_cache: dict[Path, Image.Image] = {}
    mapping_cache: dict[str, dict] = {}
    gt_cache: dict[str, dict] = {}
    geometry_cache: dict[str, object] = {}
    schema_cache: dict[str, dict] = {}
    structured_cache: dict[str, StructuredOutputsParams] = {}
    invocation_started = time.perf_counter()
    processed = 0

    def image_for(path: Path) -> Image.Image:
        if path not in image_cache:
            image_cache[path] = Image.open(path).convert("RGB")
        return image_cache[path]

    def cached_json(relative_path: str, cache: dict[str, dict]) -> dict:
        if relative_path not in cache:
            cache[relative_path] = read_json(experiment / relative_path)
        return cache[relative_path]

    with output_path.open("a", encoding="utf-8") as run_handle, attempts_path.open("a", encoding="utf-8") as attempt_handle:
        for row in pending:
            geometry = None
            if row["geometry_json"]:
                geometry = cached_json(row["geometry_json"], geometry_cache)
            mapping = cached_json(row["id_mapping"], mapping_cache)
            gt = cached_json(row["ground_truth"], gt_cache)
            schema = cached_json(row["schema"], schema_cache)
            if row["schema"] not in structured_cache:
                structured_cache[row["schema"]] = StructuredOutputsParams(json=schema)
            user_prompt = build_user_prompt(row, geometry, task_prompts[row["level"]])
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_path = experiment / row["numbered_rgb"]
            request = {
                "prompt": chat,
                "multi_modal_data": {"image": image_for(image_path)},
                "multi_modal_uuids": {"image": row["numbered_rgb_sha256"]},
            }
            sampling = SamplingParams(
                n=1,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                presence_penalty=float(generation["presence_penalty"]),
                frequency_penalty=float(generation["frequency_penalty"]),
                max_tokens=int(generation["max_tokens"]),
                seed=int(row["seed"]),
                structured_outputs=structured_cache[row["schema"]],
            )

            last_error = None
            for attempt_index in range(1, args.max_attempts + 1):
                attempt_started_at = utc_now()
                started = time.perf_counter()
                try:
                    outputs = engine.generate([request], sampling_params=[sampling], use_tqdm=False)
                    wall_seconds = time.perf_counter() - started
                    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                        raise RuntimeError("vLLM returned an unexpected number of outputs")
                    request_output = outputs[0]
                    completion = request_output.outputs[0]
                    raw = completion.text.strip()
                    parse_error = None
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        parsed = None
                        parse_error = str(exc)
                    validation_errors = semantic_validate(row["level"], parsed, row, mapping) if parsed is not None else []
                    scoring = score_non_action(row["level"], parsed, gt)
                    input_tokens = len(request_output.prompt_token_ids)
                    output_tokens = len(completion.token_ids)
                    record = {
                        **row,
                        "model_id": config["model_id"],
                        "model_path": config["model_path"],
                        "generation": generation,
                        "runtime": runtime,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt_sha256": common.sha256_text(system_prompt + "\n\n" + user_prompt),
                        "geometry_json_raw": geometry,
                        "json_schema": schema,
                        "raw_response": raw,
                        "parsed_response": parsed,
                        "parse_error": parse_error,
                        "post_validation_errors": validation_errors,
                        "post_validation_pass": parsed is not None and not validation_errors,
                        "scoring": scoring,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "latency_seconds": round(wall_seconds, 6),
                        "input_token_usage": input_tokens,
                        "output_token_usage": output_tokens,
                        "started_at": attempt_started_at,
                        "finished_at": utc_now(),
                        "attempt_index": attempt_index,
                    }
                    if input_tokens + int(generation["max_tokens"]) > int(runtime_config["max_model_len"]):
                        raise RuntimeError(f"Context budget violation for {row['run_id']}")
                    append_jsonl(attempt_handle, {
                        "run_id": row["run_id"], "attempt_index": attempt_index,
                        "seed": row["seed"], "status": "success",
                        "started_at": attempt_started_at, "finished_at": record["finished_at"],
                        "latency_seconds": record["latency_seconds"],
                        "finish_reason": completion.finish_reason,
                        "raw_response": raw,
                    })
                    append_jsonl(run_handle, record)
                    completed[row["run_id"]] = record
                    processed += 1
                    progress = {
                        "expected": len(schedule), "completed": len(completed),
                        "processed_this_invocation": processed,
                        "elapsed_seconds": round(time.perf_counter() - invocation_started, 3),
                        "last_run_id": row["run_id"],
                        "last_finish_reason": completion.finish_reason,
                        "last_post_validation_pass": record["post_validation_pass"],
                        "updated_at": utc_now(),
                    }
                    common.write_json(progress_path, progress)
                    if processed == 1 or processed % 25 == 0 or len(completed) == len(schedule):
                        print(
                            f"[{len(completed)}/{len(schedule)}] {row['run_id']} "
                            f"finish={completion.finish_reason} tokens={input_tokens}+{output_tokens} "
                            f"valid={record['post_validation_pass']} wall={wall_seconds:.2f}s",
                            flush=True,
                        )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    append_jsonl(attempt_handle, {
                        "run_id": row["run_id"], "attempt_index": attempt_index,
                        "seed": row["seed"], "status": "infrastructure_error",
                        "started_at": attempt_started_at, "finished_at": utc_now(),
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    print(
                        f"[retry {attempt_index}/{args.max_attempts}] {row['run_id']}: "
                        f"{type(exc).__name__}: {exc}", flush=True,
                    )
            if last_error is not None:
                raise RuntimeError(
                    f"Infrastructure failure after {args.max_attempts} attempts for {row['run_id']}"
                ) from last_error

    summary = {
        "completed": len(completed), "expected": len(schedule),
        "processed_this_invocation": processed,
        "model_load_seconds": round(load_seconds, 3),
        "generation_seconds": round(time.perf_counter() - invocation_started, 3),
        "output": str(output_path), "finished_at": utc_now(),
    }
    common.write_json(output_path.parent / "last_invocation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
