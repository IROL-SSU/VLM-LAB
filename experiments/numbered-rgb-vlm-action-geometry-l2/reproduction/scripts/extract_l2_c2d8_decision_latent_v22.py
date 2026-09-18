#!/usr/bin/env python3
"""Extract one C2 condition's decision-boundary states from frozen L2 v16.

The readout is taken after the common answer prefix
`{"decision": "RE` and immediately before the first divergent token:
`TR` for RETRIEVE_NOW versus `ARR` for REARRANGE_FIRST.

Run with `.qwen3-vl/venv/bin/python`. The script does not regenerate or alter
the source v16 responses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXTRA_DEPS = ROOT / ".l1-latent/extraction-deps"
if EXTRA_DEPS.exists():
    sys.path.insert(0, str(EXTRA_DEPS))

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


SOURCE = ROOT / "experiments/vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16"
OUTPUT = ROOT / "experiments/vlm_action_geometry_single_info_l2_c2d8_decision_latent_v22"
FAMILY_ORDER = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
RETRIEVE = "RETRIEVE_NOW"
REARRANGE = "REARRANGE_FIRST"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expected_decision(family: str) -> str:
    return RETRIEVE if family == "FC_CLEAR" else REARRANGE


def collect_records(source: Path, condition_key: str) -> tuple[list[dict], dict]:
    config_path = source / "config/experiment_config.json"
    log_path = source / "logs/runs.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["condition_key"] == condition_key:
            groups[row["scene_id"]].append(row)
    assert len(groups) == 25, f"Expected 25 {condition_key} scenes, found {len(groups)}"

    geometry_condition, direction_resolution = condition_key.split("_", 1)
    assert geometry_condition == "C2", geometry_condition
    assert direction_resolution in {"D4", "D8"}, direction_resolution

    records = []
    source_hashes = {}
    order = {family: idx for idx, family in enumerate(FAMILY_ORDER)}
    for scene_id, runs in sorted(groups.items(), key=lambda item: (order[item[1][0]["scene_family"]], item[0])):
        runs.sort(key=lambda row: row["seed"])
        assert [row["seed"] for row in runs] == sorted(config["seeds"]), scene_id
        assert len(runs) == 5 and len({row["run_id"] for row in runs}) == 5
        representative = runs[0]
        shared = [
            "system_prompt", "user_prompt", "prompt_sha256", "image_path", "image_sha256",
            "id_mapping_path", "id_mapping_sha256", "geometry_path", "geometry_sha256",
            "schema_sha256", "target_object_id", "input_tokens", "model_id", "model_path",
        ]
        for field in shared:
            assert all(row[field] == representative[field] for row in runs), (scene_id, field)
        assert representative["condition_key"] == condition_key
        assert representative["geometry_condition"] == geometry_condition
        assert representative["direction_resolution"] == direction_resolution
        assert representative["model_id"] == config["model_id"]
        assert sha_text(representative["system_prompt"] + "\n\n" + representative["user_prompt"]) == representative["prompt_sha256"]

        for path_field, hash_field in [
            ("image_path", "image_sha256"),
            ("id_mapping_path", "id_mapping_sha256"),
            ("geometry_path", "geometry_sha256"),
        ]:
            path = Path(representative[path_field])
            actual = sha(path)
            assert actual == representative[hash_field], f"Source hash mismatch: {path}"
            source_hashes[str(path.relative_to(ROOT))] = actual

        decisions = [row["parsed_response"]["decision"] for row in runs]
        counts = Counter(decisions)
        expected = expected_decision(representative["scene_family"])
        correct_count = sum(decision == expected for decision in decisions)
        records.append({
            "index": len(records),
            "scene_id": scene_id,
            "scene_family": representative["scene_family"],
            "target_object_id": representative["target_object_id"],
            "image_path": str(Path(representative["image_path"]).relative_to(ROOT)),
            "image_sha256": representative["image_sha256"],
            "geometry_path": str(Path(representative["geometry_path"]).relative_to(ROOT)),
            "geometry_sha256": representative["geometry_sha256"],
            "system_prompt": representative["system_prompt"],
            "user_prompt": representative["user_prompt"],
            "prompt_sha256": representative["prompt_sha256"],
            "logged_input_tokens": representative["input_tokens"],
            "expected_decision": expected,
            "historical_predictions": [
                {
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "decision": decision,
                    "matches_scene_design": decision == expected,
                }
                for row, decision in zip(runs, decisions)
            ],
            "retrieve_count": counts[RETRIEVE],
            "rearrange_count": counts[REARRANGE],
            "correct_count": correct_count,
            "majority_decision": counts.most_common(1)[0][0],
            "outcome": "all_correct" if correct_count == 5 else "all_wrong" if correct_count == 0 else "mixed",
            "historical_backend": representative["runtime"],
        })
    assert sum(record["retrieve_count"] + record["rearrange_count"] for record in records) == 125
    return records, {
        "config": config,
        "source_config_sha256": sha(config_path),
        "source_log_sha256": sha(log_path),
        "source_hashes": source_hashes,
    }


def decision_boundary(tokenizer) -> dict:
    retrieve_text = '{"decision": "' + RETRIEVE
    rearrange_text = '{"decision": "' + REARRANGE
    retrieve_ids = tokenizer.encode(retrieve_text, add_special_tokens=False)
    rearrange_ids = tokenizer.encode(rearrange_text, add_special_tokens=False)
    common_length = 0
    for left, right in zip(retrieve_ids, rearrange_ids):
        if left != right:
            break
        common_length += 1
    assert 0 < common_length < min(len(retrieve_ids), len(rearrange_ids))
    common_ids = retrieve_ids[:common_length]
    common_text = tokenizer.decode(common_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    retrieve_branch_id = retrieve_ids[common_length]
    rearrange_branch_id = rearrange_ids[common_length]
    # Frozen v16 emits the exact prefix below. These checks make the intended
    # branch point explicit and fail fast if the tokenizer changes.
    assert common_text == '{"decision": "RE', repr(common_text)
    assert tokenizer.convert_ids_to_tokens(common_ids[-1]) == "RE"
    assert tokenizer.convert_ids_to_tokens(retrieve_branch_id) == "TR"
    assert tokenizer.convert_ids_to_tokens(rearrange_branch_id) == "ARR"
    return {
        "retrieve_output_prefix": retrieve_text,
        "rearrange_output_prefix": rearrange_text,
        "common_text": common_text,
        "common_token_ids": common_ids,
        "common_tokens": tokenizer.convert_ids_to_tokens(common_ids),
        "retrieve_branch_token_id": retrieve_branch_id,
        "retrieve_branch_token": tokenizer.convert_ids_to_tokens(retrieve_branch_id),
        "rearrange_branch_token_id": rearrange_branch_id,
        "rearrange_branch_token": tokenizer.convert_ids_to_tokens(rearrange_branch_id),
        "readout_definition": "Last common token RE, whose state predicts the first divergent TR versus ARR token.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--condition-key", choices=("C2_D4", "C2_D8"), default="C2_D8")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latent_states.npz").exists():
        raise FileExistsError("Extraction already exists; use a new output directory.")

    records, audit = collect_records(source, args.condition_key)
    model_path = Path(audit["config"]["model_path"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    boundary = decision_boundary(processor.tokenizer)
    prepared = []
    for record in records:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": record["system_prompt"]}]},
            {"role": "user", "content": [
                {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
                {"type": "text", "text": record["user_prompt"]},
            ]},
        ]
        rendered_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        boundary_prompt = rendered_chat + boundary["common_text"]
        with Image.open(ROOT / record["image_path"]) as raw_image:
            inputs = processor(
                text=[boundary_prompt], images=[raw_image.convert("RGB")], padding=True, return_tensors="pt"
            )
        assert inputs.input_ids.shape[0] == 1
        assert bool(inputs.attention_mask.all()), "This extraction expects one unpadded example."
        token_ids = inputs.input_ids[0].tolist()
        assert token_ids[-len(boundary["common_token_ids"]):] == boundary["common_token_ids"]
        assert inputs.input_ids.shape[1] == record["logged_input_tokens"] + len(boundary["common_token_ids"])
        record.update({
            "rendered_chat_sha256": sha_text(rendered_chat),
            "boundary_prompt_sha256": sha_text(boundary_prompt),
            "input_token_ids": token_ids,
            "input_token_count_at_boundary": inputs.input_ids.shape[1],
            "image_grid_thw": inputs.image_grid_thw.tolist(),
            "readout_token_position": inputs.input_ids.shape[1] - 1,
            "readout_token_id": int(inputs.input_ids[0, -1]),
            "readout_token_text": processor.tokenizer.decode(inputs.input_ids[0, -1:]),
            "pixel_values_sha256": hashlib.sha256(inputs.pixel_values.numpy().tobytes()).hexdigest(),
        })
        prepared.append(inputs)

    write_json(output / "records.json", records)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(source),
        "source_condition": args.condition_key,
        "n_scenes": len(records),
        "n_historical_responses": 125,
        "independent_latent_unit": "scene, not sampling seed",
        "model_path": str(model_path),
        "model_id": audit["config"]["model_id"],
        "source_config_sha256": audit["source_config_sha256"],
        "source_log_sha256": audit["source_log_sha256"],
        "source_file_hashes": audit["source_hashes"],
        "source_code_sha256": sha(Path(__file__)),
        "model_config_sha256": sha(model_path / "config.json"),
        "model_weight_index_sha256": sha(model_path / "model.safetensors.index.json"),
        "model_weight_files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(model_path.glob("*.safetensors"))
        ],
        "model_weights_hash_scope": "Config and weight index hashed; shard filenames and byte sizes recorded.",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": importlib.metadata.version("accelerate"),
            "dtype": "bfloat16",
            "attention": "sdpa",
            "batch_size": 1,
            "use_cache": False,
            "backend": "transformers_forward",
            "cuda_device": torch.cuda.get_device_name(),
        },
        "decision_boundary": boundary,
        "mid_state": "Output of decoder block 24 at the RE readout token, before final RMSNorm.",
        "final_state": "Same token after decoder block 48 and final RMSNorm, immediately before lm_head.",
        "branch_logits": "Raw lm_head logits for TR and ARR at the readout token; two-way softmax is descriptive.",
        "historical_overlay": (
            "Five original vLLM constrained-decoding responses per scene. They were not regenerated during "
            "this extraction; exact backend numerical equivalence is not claimed."
        ),
        "preflight": {
            "all_source_hashes_match": True,
            "all_prompt_hashes_match": True,
            "all_boundary_tokenizations_match": True,
            "one_latent_point_per_scene": True,
            "answer_after_branch_not_in_input": True,
            "historical_outcomes": dict(Counter(record["outcome"] for record in records)),
        },
    }
    write_json(output / "extraction_manifest.json", manifest)
    print(json.dumps({
        "status": "preflight_passed",
        "scenes": len(records),
        "boundary": boundary,
        "output": str(output),
    }, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return

    torch.manual_seed(20260917)
    torch.cuda.manual_seed_all(20260917)
    torch.set_num_threads(8)
    print("Loading local BF16 checkpoint onto cuda:0...", flush=True)
    started = time.perf_counter()
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)
    language = model.model.language_model
    assert len(language.layers) == 48
    captured = {}

    def capture(name):
        def hook(_module, _args, output_tensor):
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            captured[name] = output_tensor[0, -1].detach().float().cpu().numpy().copy()
        return hook

    handles = [
        language.layers[23].register_forward_hook(capture("mid_layer24")),
        language.norm.register_forward_hook(capture("final_norm")),
    ]
    mids, finals, branch_logits = [], [], []
    try:
        with torch.inference_mode():
            for idx, (record, cpu_inputs) in enumerate(zip(records, prepared)):
                captured.clear()
                tick = time.perf_counter()
                inputs = cpu_inputs.to("cuda:0")
                result = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
                assert set(captured) == {"mid_layer24", "final_norm"}
                mid = captured["mid_layer24"]
                final = captured["final_norm"]
                assert mid.shape == (2048,) and final.shape == (2048,)
                logits = result.logits[0, -1].float()
                pair = np.array([
                    logits[boundary["retrieve_branch_token_id"]].item(),
                    logits[boundary["rearrange_branch_token_id"]].item(),
                ], dtype=np.float32)
                assert np.isfinite(mid).all() and np.isfinite(final).all() and np.isfinite(pair).all()
                mids.append(mid)
                finals.append(final)
                branch_logits.append(pair)
                two_way = torch.softmax(torch.from_numpy(pair), dim=0).numpy()
                record.update({
                    "extraction_seconds": round(time.perf_counter() - tick, 3),
                    "branch_logit_retrieve_TR": float(pair[0]),
                    "branch_logit_rearrange_ARR": float(pair[1]),
                    "branch_logit_margin_retrieve_minus_rearrange": float(pair[0] - pair[1]),
                    "branch_two_way_probability_retrieve": float(two_way[0]),
                    "branch_two_way_probability_rearrange": float(two_way[1]),
                })
                np.savez_compressed(
                    output / "latent_states.partial.npz",
                    mid_layer24_states=np.stack(mids),
                    final_norm_states=np.stack(finals),
                    branch_logits=np.stack(branch_logits),
                    scene_ids=np.array([row["scene_id"] for row in records[:idx + 1]]),
                )
                print(json.dumps({
                    "completed": idx + 1,
                    "total": len(records),
                    "scene": record["scene_id"],
                    "seconds": record["extraction_seconds"],
                    "retrieve_probability_two_way": record["branch_two_way_probability_retrieve"],
                    "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }), flush=True)
                del result, logits, inputs
    finally:
        for handle in handles:
            handle.remove()

    (output / "latent_states.partial.npz").replace(output / "latent_states.npz")
    write_json(output / "records.json", records)
    manifest.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "mid_layer24_states_shape": list(np.stack(mids).shape),
        "final_norm_states_shape": list(np.stack(finals).shape),
        "branch_logits_shape": list(np.stack(branch_logits).shape),
        "latent_states_sha256": sha(output / "latent_states.npz"),
        "records_sha256": sha(output / "records.json"),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    })
    write_json(output / "extraction_manifest.json", manifest)
    print("Extraction complete.", flush=True)


if __name__ == "__main__":
    main()
