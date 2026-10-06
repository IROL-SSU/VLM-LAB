#!/usr/bin/env python3
"""Five-scene supervised overfit check, NOT a held-out/generalization result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from generate_translate_rgbd_pilot import ROOT, write_json
from translate_subgoal_model import (
    DIRECTION_NAMES, INPUT_FILES, TranslateSubgoalModel, decode,
    observation_batch, read_json, supervised_loss,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "experiments/translate_random_pilot_20261001_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/translate_train_pilot_20261001_v1")
    parser.add_argument("--rgb-weights", type=Path, default=ROOT / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        raise ValueError("steps and lr must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    if not args.rgb_weights.is_file():
        raise FileNotFoundError("Official pretrained RGB weights are required; no random fallback")
    pretrained_hash = file_hash(args.rgb_weights)
    if not pretrained_hash.startswith("f37072fd"):
        raise ValueError("ResNet18 IMAGENET1K_V1 checksum mismatch")
    out, dataset = args.output.resolve(), args.dataset.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a run: {out}")
    from validate_translate_rgbd_pilot import validate
    validation = validate(dataset)
    out.mkdir(parents=True)
    write_json(out / "dataset_validation.json", validation)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read_json(dataset / "manifest.json")
    rows = manifest["samples"]
    if len(rows) != 5:
        raise ValueError("This experiment protocol expects exactly five training scenes")
    labels = [read_json(dataset / row["sample_dir"] / "label.json") for row in rows]
    y_dir = torch.tensor([DIRECTION_NAMES.index(label["direction"]) for label in labels], device=args.device)
    y_dist = torch.tensor([label["distance_m"] for label in labels], dtype=torch.float32, device=args.device)
    batch = {key: value.to(args.device) for key, value in observation_batch(dataset, rows, args.seed).items()}
    model = TranslateSubgoalModel(args.rgb_weights).to(args.device)
    rgb_initial_hash = state_hash(model.rgb_encoder)
    with torch.no_grad():
        rgb_features = model.rgb_encoder(batch["image"], batch["masks"]).detach()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    config = {
        "experiment": "five-scene-overfit-debug", "dataset": str(dataset), "seed": args.seed,
        "train_scene_ids": [row["scene_id"] for row in rows], "validation_scene_ids": [], "test_scene_ids": [],
        "steps": args.steps, "batch_size": len(rows), "optimizer": "Adam", "lr": args.lr,
        "scheduler": "cosine to 1e-5", "augmentation": "none", "point_sampling": "fixed per observation, 512 target + 512 blocker + 1024 scene",
        "input_files": list(INPUT_FILES), "rgb_input_hw": [480, 640],
        "rgb_encoder": "ResNet18 ImageNet V1; all conv stages; target/blocker/global soft-mask pooling; frozen weights AND BatchNorm; feature cache",
        "geometry_encoder": "PointNet-style shared MLP 5->64->128->128; role/global max+mean pooling; observed centers and relative center",
        "fusion": "RGB/geometry each projected to 128; concat->128->64",
        "output": "8 direction logits + 8 direction-conditioned distances (0..0.25m); no action/rotation head",
        "direction_names": list(DIRECTION_NAMES), "distance_scale_m": .25,
        "loss": "cross_entropy + 5 * smooth_l1(distance/0.25, label/0.25, beta=0.02); regression at GT direction only",
        "checkpoint_selection": "lowest TRAINING loss, evaluated every 10 steps; not validation selection",
        "rgb_weights_source": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "rgb_weights_sha256": pretrained_hash,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "limitations": "All evaluation scenes are training scenes. Four RIGHT, one LEFT; other six directions unvalidated. Static AABB proxy, not executor or grasp success.",
    }
    write_json(out / "config.json", config)
    fingerprints = {"manifest.json": file_hash(dataset / "manifest.json")}
    for row in rows:
        for filename in (*INPUT_FILES, "label.json"):
            relative = Path(row["sample_dir"]) / filename
            fingerprints[str(relative)] = file_hash(dataset / relative)
    write_json(out / "dataset_fingerprints.json", fingerprints)
    source_dir = out / "source"
    source_dir.mkdir()
    for name in ("translate_subgoal_model.py", "train_translate_pilot.py", "evaluate_translate_pilot.py",
                 "generate_translate_rgbd_pilot.py", "translate_pilot_geometry.py", "validate_translate_rgbd_pilot.py", "revise_translate_pilot_visibility.py"):
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    history, best_loss, best_state, best_step = [], float("inf"), None, None
    start = time.monotonic()

    @torch.no_grad()
    def record(step):
        nonlocal best_loss, best_state, best_step
        model.eval()
        output = model.from_features(rgb_features, batch["points"])
        total, classification, regression = supervised_loss(output, y_dir, y_dist)
        predicted_dir, predicted_dist, _ = decode(output)
        item = {"step": step, "loss": float(total), "direction_loss": float(classification),
                "distance_loss": float(regression), "direction_accuracy": float((predicted_dir == y_dir).float().mean()),
                "distance_mae_mm": float((predicted_dist - y_dist).abs().mean() * 1000),
                "distance_max_error_mm": float((predicted_dist - y_dist).abs().max() * 1000),
                "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - start}
        history.append(item)
        if step > 0 and item["loss"] < best_loss:
            best_loss, best_step = item["loss"], step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("rgb_encoder.")}
        if step % 100 == 0 or step == args.steps:
            print(f"[train] step={step} loss={item['loss']:.6f} direction={item['direction_accuracy']:.0%} MAE={item['distance_mae_mm']:.4f}mm max={item['distance_max_error_mm']:.4f}mm", flush=True)
        write_json(out / "progress.json", item)

    record(0)
    gradient_audit = {}
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model.from_features(rgb_features, batch["points"])
        loss, _, _ = supervised_loss(output, y_dir, y_dist)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at {step}")
        loss.backward()
        if step == 1:
            for prefix in ("rgb_encoder", "geometry_encoder", "rgb_projection", "geo_projection", "fusion", "direction_head", "distance_head"):
                params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                gradient_audit[prefix] = {"grad_l2": sum(float(p.grad.square().sum()) for p in params if p.grad is not None) ** .5,
                                          "parameters_with_grad": sum(p.grad is not None for p in params)}
            assert gradient_audit["rgb_encoder"]["parameters_with_grad"] == 0
            assert all(value["grad_l2"] > 0 for key, value in gradient_audit.items() if key != "rgb_encoder")
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5)
        optimizer.step()
        scheduler.step()
        if step % 10 == 0 or step == args.steps:
            record(step)
    rgb_final_hash = state_hash(model.rgb_encoder)
    assert rgb_initial_hash == rgb_final_hash, "Frozen RGB parameters / buffers changed"
    write_json(out / "gradient_audit.json", {"first_step": gradient_audit, "rgb_state_before": rgb_initial_hash,
                                            "rgb_state_after": rgb_final_hash, "frozen_rgb_unchanged": True})
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model_state": final_state, "config": config, "step": args.steps,
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict()}, out / "last.pt")
    final_state.update(best_state)
    torch.save({"model_state": final_state, "config": config, "step": best_step, "training_loss": best_loss}, out / "best.pt")
    write_json(out / "history.json", history)
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"[train] saved best.pt at training step {best_step}; starting independent checkpoint reload", flush=True)
    from evaluate_translate_pilot import evaluate
    result = evaluate(out / "best.pt", dataset, out / "evaluation", device=args.device)
    write_json(out / "completion.json", {"completed": True, "best_step": best_step, "best_training_loss": best_loss,
                                         "training_wall_seconds": history[-1]["elapsed_s"], "evaluation": result["summary"]})
    print(f"[train] COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
