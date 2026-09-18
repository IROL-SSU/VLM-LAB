#!/usr/bin/env python3
"""Extract one deterministic C2/D4 pre-reason hidden state per v19 scene.

The model never samples an answer. A fixed assistant JSON prefix is appended
to every prompt, and the final prefix token is used as the readout position:

    {"blocking_reason": "<readout predicts the first reason token>

This mirrors the v23/L1 latent workflow while moving the readout to the
blocking_reason value decision point.
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


SOURCE_DEFAULT = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
OUTPUT_DEFAULT = ROOT / "experiments/l3_v19_c2_d4_reason_latent_20260918"
ASSISTANT_PREFIX = '{"blocking_reason": "'
PREFIX = {
    "FC_BLOCKED": "B",
    "FC_CLEAR": "C",
    "LIFT_AND_RELOCATE": "L",
    "ROTATE": "R",
    "TRANSLATE": "T",
}


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


def collect_records(source: Path) -> tuple[list[dict], dict]:
    config = json.loads((source / "config/experiment_config.json").read_text())
    groups = defaultdict(list)
    with (source / "logs/runs.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["geometry_condition"] == "C2" and row["direction_resolution"] == "D4":
                groups[row["scene_id"]].append(row)
    assert len(groups) == 25, f"Expected 25 scenes, found {len(groups)}"

    records = []
    source_hashes = {}
    for scene_id, runs in sorted(groups.items()):
        runs.sort(key=lambda row: row["seed"])
        assert [r["seed"] for r in runs] == sorted(config["main_seeds"]), scene_id
        assert len(runs) == 5 and len({r["run_id"] for r in runs}) == 5
        representative = runs[0]
        shared = [
            "system_prompt", "user_prompt", "prompt_sha256", "numbered_rgb_sha256",
            "geometry_json_sha256", "ground_truth_sha256", "id_mapping_sha256",
            "schema_sha256", "target_object_id", "input_token_usage", "model_id", "model_path",
        ]
        for field in shared:
            assert all(r[field] == representative[field] for r in runs), (scene_id, field)
        assert representative["geometry_json_raw"]["condition"] == "R_QUALITATIVE"
        assert representative["geometry_json_raw"]["direction_resolution"] == "D4"
        assert sha_text(representative["system_prompt"] + "\n\n" + representative["user_prompt"]) == representative["prompt_sha256"]

        for field, hash_field in [
            ("numbered_rgb", "numbered_rgb_sha256"),
            ("ground_truth", "ground_truth_sha256"),
            ("id_mapping", "id_mapping_sha256"),
            ("schema", "schema_sha256"),
        ]:
            path = source / representative[field]
            actual = sha(path)
            assert actual == representative[hash_field], f"Source hash mismatch: {path}"
            source_hashes[str(path.relative_to(ROOT))] = actual

        gt = json.loads((source / representative["ground_truth"]).read_text())["L3"]
        historical = []
        for run in runs:
            historical.append({
                "run_id": run["run_id"],
                "seed": run["seed"],
                "response": run["parsed_response"],
                "reason_correct": bool(run["scoring"]["blocking_reason_correct"]),
                "blocker_correct": bool(run["scoring"]["blocker_valid_set_member"]),
                "joint_correct": bool(run["scoring"]["task_correct"]),
            })
        reason_counts = Counter(x["response"]["blocking_reason"] for x in historical)
        blocker_counts = Counter(str(x["response"]["blocker_id"]) for x in historical)
        family = representative["scene_family"]
        records.append({
            "index": len(records),
            "scene_id": scene_id,
            "short_id": PREFIX[family] + str(int(scene_id.rsplit("v", 1)[1])),
            "scene_family": family,
            "family_clear": family == "FC_CLEAR",
            "semantic_clear": gt["blocking_reason"] == "NONE",
            "target_object_id": representative["target_object_id"],
            "image_path": str((source / representative["numbered_rgb"]).relative_to(ROOT)),
            "image_sha256": representative["numbered_rgb_sha256"],
            "ground_truth": gt,
            "geometry": representative["geometry_json_raw"],
            "system_prompt": representative["system_prompt"],
            "user_prompt": representative["user_prompt"],
            "prompt_sha256": representative["prompt_sha256"],
            "logged_input_tokens": representative["input_token_usage"],
            "historical_responses": historical,
            "reason_prediction_counts": dict(reason_counts),
            "blocker_prediction_counts": dict(blocker_counts),
            "reason_correct_count": sum(x["reason_correct"] for x in historical),
            "blocker_correct_count": sum(x["blocker_correct"] for x in historical),
            "joint_correct_count": sum(x["joint_correct"] for x in historical),
            "historical_backend": representative["runtime"],
        })
    return records, {
        "config": config,
        "source_hashes": source_hashes,
        "source_log_sha256": sha(source / "logs/runs.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    records, audit = collect_records(source)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "hidden_states.npz").exists():
        raise FileExistsError("Extraction already exists; use a new output directory.")

    model_path = Path(audit["config"]["model_path"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    prepared = []
    for record in records:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": record["system_prompt"]}]},
            {"role": "user", "content": [
                {"type": "image", "image": "file:///numbered_rgb_placeholder.png"},
                {"type": "text", "text": record["user_prompt"]},
            ]},
        ]
        base_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        decision_chat = base_chat + ASSISTANT_PREFIX
        with Image.open(ROOT / record["image_path"]) as raw_image:
            image = raw_image.convert("RGB")
            base_inputs = processor(text=[base_chat], images=[image], return_tensors="pt")
            inputs = processor(text=[decision_chat], images=[image], return_tensors="pt")
        assert base_inputs.input_ids.shape[0] == inputs.input_ids.shape[0] == 1
        assert base_inputs.input_ids.shape[1] == record["logged_input_tokens"], record["scene_id"]
        base_ids = base_inputs.input_ids[0].tolist()
        decision_ids = inputs.input_ids[0].tolist()
        assert decision_ids[:len(base_ids)] == base_ids, record["scene_id"]
        assert len(decision_ids) > len(base_ids)
        assert bool(inputs.attention_mask.all()), "This extraction expects one unpadded example."
        prefix_ids = decision_ids[len(base_ids):]
        record.update({
            "base_rendered_chat_sha256": sha_text(base_chat),
            "decision_rendered_chat_sha256": sha_text(decision_chat),
            "assistant_prefix": ASSISTANT_PREFIX,
            "assistant_prefix_token_ids": prefix_ids,
            "assistant_prefix_token_texts": [processor.tokenizer.decode([token]) for token in prefix_ids],
            "decision_input_tokens": len(decision_ids),
            "readout_token_position": len(decision_ids) - 1,
            "readout_token_id": int(decision_ids[-1]),
            "readout_token_text": processor.tokenizer.decode(decision_ids[-1:]),
            "image_grid_thw": inputs.image_grid_thw.tolist(),
            "pixel_values_sha256": hashlib.sha256(inputs.pixel_values.numpy().tobytes()).hexdigest(),
        })
        prepared.append(inputs)

    write_json(output / "records.json", records)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(source),
        "source_version": "l3_primary_blocker_v19",
        "level": "L3",
        "condition": "C2_D4",
        "n_scenes": len(records),
        "n_historical_responses": sum(len(r["historical_responses"]) for r in records),
        "independent_point_unit": "scene, NOT sampling seed",
        "model_path": str(model_path),
        "model_id": audit["config"]["model_id"],
        "source_log_sha256": audit["source_log_sha256"],
        "source_file_hashes": audit["source_hashes"],
        "source_code_sha256": sha(Path(__file__)),
        "model_config_sha256": sha(model_path / "config.json"),
        "model_weight_index_sha256": sha(model_path / "model.safetensors.index.json"),
        "model_weight_files": [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(model_path.glob("*.safetensors"))
        ],
        "model_weights_hash_scope": "Config and weight index hashed; weight shards recorded by filename and byte size.",
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
        "readout": "Final token of a fixed assistant prefix, immediately before the first blocking_reason value token.",
        "assistant_prefix": ASSISTANT_PREFIX,
        "answer_generation": False,
        "answer_or_gt_value_in_input": False,
        "sampling_seed_relevance": "None: no token sampling or generation occurs.",
        "block_states": "Outputs of decoder blocks 1..48 at the fixed pre-reason readout token, before final RMSNorm.",
        "final_state": "The same token after final RMSNorm, returned as last_hidden_state.",
        "historical_overlay": "Five existing v19 vLLM responses per scene are metadata only and were not generated during extraction.",
        "preflight": {
            "all_source_hashes_match": True,
            "all_base_input_token_counts_match": True,
            "one_point_per_scene": True,
            "reason_value_tokens_in_input": False,
            "family_counts": dict(Counter(r["scene_family"] for r in records)),
            "gt_reason_counts": dict(Counter(r["ground_truth"]["blocking_reason"] for r in records)),
            "family_clear_scene_count": sum(r["family_clear"] for r in records),
            "semantic_clear_scene_count": sum(r["semantic_clear"] for r in records),
            "readout_token_texts": sorted({r["readout_token_text"] for r in records}),
        },
    }
    write_json(output / "extraction_manifest.json", manifest)
    print(json.dumps({
        "status": "preflight_passed",
        "n": len(records),
        "output": str(output),
        "prefix_token_ids": records[0]["assistant_prefix_token_ids"],
        "prefix_token_texts": records[0]["assistant_prefix_token_texts"],
        "readout_token_text": records[0]["readout_token_text"],
    }, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return

    torch.manual_seed(20260918)
    torch.cuda.manual_seed_all(20260918)
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

    def hook_for(layer):
        def hook(_module, _args, output_tensor):
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            captured[layer] = output_tensor[0, -1].detach().float().cpu().numpy().copy()
        return hook

    handles = [block.register_forward_hook(hook_for(i + 1)) for i, block in enumerate(language.layers)]
    blocks, finals = [], []
    try:
        with torch.inference_mode():
            for idx, (record, cpu_inputs) in enumerate(zip(records, prepared)):
                captured.clear()
                tick = time.perf_counter()
                inputs = cpu_inputs.to("cuda:0")
                result = model.model(**inputs, use_cache=False, return_dict=True)
                assert set(captured) == set(range(1, 49))
                block_states = np.stack([captured[i] for i in range(1, 49)])
                final = result.last_hidden_state[0, -1].float().cpu().numpy()
                assert block_states.shape == (48, 2048) and final.shape == (2048,)
                assert np.isfinite(block_states).all() and np.isfinite(final).all()
                expected = language.norm(
                    torch.from_numpy(block_states[-1]).to("cuda:0", dtype=torch.bfloat16)
                ).float().cpu().numpy()
                np.testing.assert_allclose(expected, final, rtol=1e-4, atol=1e-5)
                blocks.append(block_states)
                finals.append(final)
                record["extraction_seconds"] = round(time.perf_counter() - tick, 3)
                np.savez_compressed(
                    output / "hidden_states.partial.npz",
                    block_states=np.stack(blocks),
                    final_norm_states=np.stack(finals),
                    scene_ids=np.array([r["scene_id"] for r in records[:idx + 1]]),
                )
                print(json.dumps({
                    "completed": idx + 1,
                    "total": len(records),
                    "scene": record["scene_id"],
                    "seconds": record["extraction_seconds"],
                    "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }), flush=True)
                del result, inputs
    finally:
        for handle in handles:
            handle.remove()

    (output / "hidden_states.partial.npz").replace(output / "hidden_states.npz")
    write_json(output / "records.json", records)
    manifest.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "hidden_states_shape": list(np.stack(blocks).shape),
        "final_norm_states_shape": list(np.stack(finals).shape),
        "hidden_states_sha256": sha(output / "hidden_states.npz"),
        "records_sha256": sha(output / "records.json"),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    })
    write_json(output / "extraction_manifest.json", manifest)
    print("Extraction complete.", flush=True)


if __name__ == "__main__":
    main()
