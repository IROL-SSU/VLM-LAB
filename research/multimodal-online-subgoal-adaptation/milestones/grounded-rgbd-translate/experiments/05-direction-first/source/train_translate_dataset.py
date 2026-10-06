#!/usr/bin/env python3
"""Fixed-split RGB-D translation baseline; select on validation, test once.

Reuse the pilot architecture and observation-only loader. Frozen RGB features
are cached, while the geometry encoder and heads are trained in minibatches.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision

from evaluate_translate_pilot import constant_baseline, evaluation_preview, geometry_check
from generate_translate_rgbd_pilot import ROOT, write_json
from train_translate_pilot import file_hash, state_hash
from translate_pilot_geometry import DIRECTIONS
from translate_subgoal_model import (
    DIRECTION_NAMES, INPUT_FILES, TranslateSubgoalModel, decode,
    observation_batch, read_json, supervised_loss,
)


def validate_splits(manifest, manifests, split_ids):
    """Check identity, membership, path uniqueness, and disjointness, not labels."""
    all_rows = manifest["samples"]
    canonical = {r["scene_id"]: r for r in all_rows}
    if len(canonical) != len(all_rows):
        raise ValueError("Duplicate scene IDs in full manifest")
    paths = [r["sample_dir"] for r in all_rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate sample directories")
    seen = set()
    result = {}
    for name in ("train", "validation", "test"):
        rows = manifests[name]["samples"]
        ids = [r["scene_id"] for r in rows]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Empty, duplicate, or overlapping split: {name}")
        if set(ids) != set(split_ids[name]):
            raise ValueError(f"Split membership mismatch: {name}")
        for row in rows:
            if canonical.get(row["scene_id"]) != row:
                raise ValueError(f"Row differs from canonical manifest: {row['scene_id']}")
        seen.update(ids)
        result[name] = rows
    if seen != set(canonical):
        raise ValueError("Splits do not cover the dataset")
    return result


def labels_for(dataset, rows, device):
    labels = [read_json(dataset / row["sample_dir"] / "label.json") for row in rows]
    if any(x["direction"] not in ("LEFT", "RIGHT") or not 0 < x["distance_m"] < .25 for x in labels):
        raise ValueError("Expected LEFT/RIGHT labels with distances in (0, 0.25m)")
    return labels, torch.tensor([DIRECTION_NAMES.index(x["direction"]) for x in labels], device=device), torch.tensor(
        [x["distance_m"] for x in labels], dtype=torch.float32, device=device)


@torch.no_grad()
def cache_observations(model, dataset, rows, seed, device, output, split, batch_size=8):
    rgb, points = [], []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = observation_batch(dataset, rows[start:start + batch_size], seed)
        rgb.append(model.rgb_encoder(batch["image"].to(device), batch["masks"].to(device)).cpu())
        points.append(batch["points"])
        if start % 80 == 0 or start + batch_size >= len(rows):
            print(f"[cache {split}] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    result = {"rgb_features": torch.cat(rgb), "points": torch.cat(points)}
    torch.save({**result, "scene_ids": [r["scene_id"] for r in rows], "sampling_seed": seed,
                "rgb_state_sha256": state_hash(model.rgb_encoder), "input_files": list(INPUT_FILES)}, output / f"cache_{split}.pt")
    return {k: v.to(device) for k, v in result.items()}


def minibatches(count, batch_size, generator=None, shuffle=False):
    indices = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    return indices.split(batch_size)


@torch.no_grad()
def infer_cached(model, cache, batch_size):
    model.eval()
    outputs = []
    for idx in minibatches(len(cache["points"]), batch_size):
        outputs.append(model.from_features(cache["rgb_features"][idx], cache["points"][idx]))
    return {key: torch.cat([o[key] for o in outputs]) for key in outputs[0]}


def prediction_metrics(output, directions, distances):
    total, classification, regression = supervised_loss(output, directions, distances)
    pred_dir, pred_dist, _ = decode(output)
    gt_head = output["distances_m"].gather(1, directions[:, None]).squeeze(1)
    return {"loss": float(total), "direction_loss": float(classification), "distance_loss": float(regression),
            "direction_accuracy": float((pred_dir == directions).float().mean()),
            "distance_mae_mm": float((pred_dist - distances).abs().mean() * 1000),
            "gt_direction_head_distance_mae_mm": float((gt_head - distances).abs().mean() * 1000)}


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test(checkpoint, dataset, rows, train_labels, out, device, batch_size):
    """Reload full checkpoint, predict all test inputs, only then open test GT."""
    out.mkdir()
    digest = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    ids = {r["scene_id"] for r in rows}
    assert not ids.intersection(config["train_scene_ids"] + config["validation_scene_ids"])
    assert ids == set(config["test_scene_ids"])
    model = TranslateSubgoalModel().to(device)
    model.load_state_dict(state["model_state"], strict=True)
    cache = cache_observations(model, dataset, rows, config["seed"], device, out, "test")
    output = infer_cached(model, cache, batch_size)
    dirs, distances, confidences = [v.cpu().tolist() for v in decode(output)]
    raw = [{"scene_id": r["scene_id"], "direction": DIRECTION_NAMES[d], "distance_m": s, "confidence": c}
           for r, d, s, c in zip(rows, dirs, distances, confidences)]
    write_json(out / "raw_predictions_before_gt.json", raw)
    labels, gt_dir, gt_dist = labels_for(dataset, rows, device)
    predictions, previews = [], []
    for index, (row, p, label) in enumerate(zip(rows, raw, labels)):
        folder = dataset / row["sample_dir"]
        objects = read_json(folder / "objects.json")
        delta = np.asarray([*DIRECTIONS[p["direction"]], 0.]) * p["distance_m"]
        p = {**p, "ground_truth_direction": label["direction"], "ground_truth_distance_m": label["distance_m"],
             "direction_correct": p["direction"] == label["direction"],
             "distance_error_mm": abs(p["distance_m"] - label["distance_m"]) * 1000,
             "signed_distance_error_mm": (p["distance_m"] - label["distance_m"]) * 1000,
             "displacement_error_mm": float(np.linalg.norm(delta - label["delta_shelf_m"]) * 1000),
             "delta_shelf_m": delta.tolist(), "geometry": geometry_check(objects, p["direction"], p["distance_m"])}
        predictions.append(p)
        # Fixed first 20 in manifest order: no best-case selection.
        if index < 20:
            image = evaluation_preview(folder, objects, label, p, "HELD-OUT TEST LAYOUT")
            image.save(out / f"{row['scene_id']}.png")
            previews.append(image)
    summary = prediction_metrics(output, gt_dir, gt_dist)
    summary.update({"sample_count": len(rows), "checkpoint_epoch": state["epoch"], "checkpoint_sha256": digest,
                    "direction_correct_count": sum(p["direction_correct"] for p in predictions),
                    "distance_median_error_mm": float(np.median([p["distance_error_mm"] for p in predictions])),
                    "distance_p90_error_mm": float(np.percentile([p["distance_error_mm"] for p in predictions], 90)),
                    "distance_max_error_mm": max(p["distance_error_mm"] for p in predictions),
                    "displacement_mae_mm": float(np.mean([p["displacement_error_mm"] for p in predictions])),
                    "direction_and_5mm_count": sum(p["direction_correct"] and p["distance_error_mm"] <= 5 for p in predictions),
                    "proxy_valid_count": sum(p["geometry"]["proxy_valid"] for p in predictions),
                    "corridor_clear_count": sum(p["geometry"]["corridor_clear"] for p in predictions),
                    "collision_free_count": sum(p["geometry"]["collision_free_swept_aabb"] for p in predictions),
                    "constant_baseline": constant_baseline(train_labels, labels),
                    "protocol": "one final test evaluation; best checkpoint selected by validation loss; uncorrected predictions",
                    "limitations": "Same 15 assets and 50 role pairs across splits; only layouts held out. LEFT/RIGHT only. Static AABB goal proxy, not execution/grasp success."})
    summary["by_direction"] = {}
    for direction in ("LEFT", "RIGHT"):
        group = [p for p in predictions if p["ground_truth_direction"] == direction]
        summary["by_direction"][direction] = {"count": len(group), "correct": sum(p["direction_correct"] for p in group),
            "distance_mae_mm": float(np.mean([p["distance_error_mm"] for p in group]))}
    summary["confusion_matrix"] = {gt: {pred: sum(p["ground_truth_direction"] == gt and p["direction"] == pred for p in predictions)
                                               for pred in DIRECTION_NAMES} for gt in ("LEFT", "RIGHT")}
    assert digest == file_hash(checkpoint), "Checkpoint mutated during test"
    summary["checkpoint_unchanged"] = True
    write_json(out / "metrics.json", {"summary": summary, "predictions": predictions})
    save_csv(out / "predictions.csv", [{k: v for k, v in p.items() if k not in ("geometry", "delta_shelf_m")} for p in predictions])
    sheet = Image.new("RGB", (4 * 480, 5 * 360), "white")
    for index, image in enumerate(previews):
        sheet.paste(image.resize((480, 360)), ((index % 4) * 480, (index // 4) * 360))
    sheet.save(out / "first20_predictions.jpg", quality=92)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "experiments/translate_diverse_1000_20261001_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/translate_train_1000_20261001_v1")
    parser.add_argument("--rgb-weights", type=Path, default=ROOT / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("Positive epochs, patience, batch size and learning rate required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent fallback")
    dataset, out = args.dataset.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite run: {out}")
    manifest_hash = file_hash(dataset / "manifest.json")
    completion = read_json(dataset / "completion.json")
    if not completion["complete"] or completion["manifest_sha256"] != manifest_hash:
        raise ValueError("Dataset completion / audit fingerprint mismatch")
    pretrained_hash = file_hash(args.rgb_weights)
    if not pretrained_hash.startswith("f37072fd"):
        raise ValueError("Pretrained ImageNet ResNet18 checksum mismatch")
    splits = validate_splits(read_json(dataset / "manifest.json"),
        {s: read_json(dataset / f"manifest_{s}.json") for s in ("train", "validation", "test")}, read_json(dataset / "splits.json"))
    if {k: len(v) for k, v in splits.items()} != {"train": 800, "validation": 100, "test": 100}:
        raise ValueError("This registered experiment expects 800/100/100")
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = TranslateSubgoalModel(args.rgb_weights).to(args.device)
    rgb_hash = state_hash(model.rgb_encoder)
    trainable = [p for p in model.parameters() if p.requires_grad]
    config = {"experiment": "1000-scenes-held-out-layout-baseline-v1", "dataset": str(dataset), "seed": args.seed,
        **{f"{s}_scene_ids": [r["scene_id"] for r in rows] for s, rows in splits.items()},
        "epochs_max": args.epochs, "early_stopping_patience": args.patience, "batch_size": args.batch_size,
        "optimizer": "Adam", "lr": args.lr, "scheduler": "cosine to 1e-5 over maximum epochs",
        "checkpoint_selection": "strictly lowest validation CE + 5 * distance SmoothL1; no test tuning",
        "initialization": "fresh random trainable modules; ImageNet frozen RGB; no pilot checkpoint reuse",
        "augmentation": "none", "point_sampling": "fixed; 512 target + 512 blocker + 1024 scene; same seed for all scenes",
        "input_files": list(INPUT_FILES), "rgb_input_hw": [480, 640],
        "architecture": "unchanged pilot: frozen ResNet18 role/global pooling + trainable PointNet-style geometry + fusion MLP",
        "direction_names": list(DIRECTION_NAMES), "distance_scale_m": .25,
        "output": "8 direction logits + 8 direction-conditioned distances; labels LEFT/RIGHT only; six other directions unvalidated",
        "loss": "CE + 5 * SmoothL1(predicted GT-direction distance / .25, label / .25, beta=.02)",
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "rgb_weights_sha256": pretrained_hash, "rgb_state_sha256": rgb_hash, "manifest_sha256": manifest_hash,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "limitations": "Same assets and role pairs across splits; unseen layouts, not unseen objects; static proxy, no executor/rotation/grasp."}
    write_json(out / "config.json", config)
    source = out / "source"
    source.mkdir()
    for name in ("train_translate_dataset.py", "translate_subgoal_model.py", "train_translate_pilot.py", "evaluate_translate_pilot.py",
                 "generate_translate_rgbd_pilot.py", "translate_pilot_geometry.py", "test_translate_dataset_training.py"):
        shutil.copy2(Path(__file__).parent / name, source / name)
    fingerprints = {name: file_hash(dataset / name) for name in ("manifest.json", "manifest_train.json", "manifest_validation.json", "manifest_test.json", "splits.json", "validation.json")}
    rgb_seen = set()
    for rows in splits.values():
        for row in rows:
            for filename in INPUT_FILES:
                relative = Path(row["sample_dir"]) / filename
                fingerprints[str(relative)] = file_hash(dataset / relative)
            rgb_digest = fingerprints[str(Path(row["sample_dir"]) / "rgb.png")]
            if rgb_digest in rgb_seen:
                raise ValueError("Exact duplicate RGB observation across dataset")
            rgb_seen.add(rgb_digest)
    write_json(out / "input_fingerprints.json", fingerprints)
    caches, targets = {}, {}
    for split in ("train", "validation"):
        caches[split] = cache_observations(model, dataset, splits[split], args.seed, args.device, out, split)
        targets[split] = labels_for(dataset, splits[split], args.device)
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    shuffle_rng = torch.Generator().manual_seed(args.seed)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    start = time.monotonic()
    gradient_audit = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, idx in enumerate(minibatches(len(splits["train"]), args.batch_size, shuffle_rng, shuffle=True)):
            optimizer.zero_grad(set_to_none=True)
            output = model.from_features(caches["train"]["rgb_features"][idx], caches["train"]["points"][idx])
            loss, _, _ = supervised_loss(output, targets["train"][1][idx], targets["train"][2][idx])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if epoch == 1 and step == 0:
                for prefix in ("rgb_encoder", "geometry_encoder", "rgb_projection", "geo_projection", "fusion", "direction_head", "distance_head"):
                    params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                    gradient_audit[prefix] = {"grad_l2": sum(float(p.grad.square().sum()) for p in params if p.grad is not None) ** .5,
                                             "parameters_with_grad": sum(p.grad is not None for p in params)}
                assert gradient_audit["rgb_encoder"]["parameters_with_grad"] == 0
                assert all(v["grad_l2"] > 0 for k, v in gradient_audit.items() if k != "rgb_encoder")
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5)
            optimizer.step()
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - start}
        for split in ("train", "validation"):
            metrics = prediction_metrics(infer_cached(model, caches[split], args.batch_size), *targets[split][1:])
            record.update({f"{split}_{key}": value for key, value in metrics.items()})
        scheduler.step()
        if not np.isfinite(record["validation_loss"]):
            raise RuntimeError("Non-finite validation loss")
        improved = record["validation_loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale = record["validation_loss"], epoch, 0
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "epoch": epoch, "validation_loss": best_loss}, out / "best.pt")
        else:
            stale += 1
        record["selected_best"] = improved
        history.append(record)
        write_json(out / "history.json", history)
        write_json(out / "progress.json", {**record, "best_epoch": best_epoch, "stale_epochs": stale})
        print(f"[epoch {epoch}] train acc={record['train_direction_accuracy']:.1%} MAE={record['train_distance_mae_mm']:.2f}mm | val loss={record['validation_loss']:.4f} acc={record['validation_direction_accuracy']:.1%} MAE={record['validation_distance_mae_mm']:.2f}mm | best={best_epoch}", flush=True)
        if stale >= args.patience:
            print(f"[stop] No validation loss improvement for {stale} epochs", flush=True)
            break
    assert rgb_hash == state_hash(model.rgb_encoder), "Frozen RGB weights or BN buffers changed"
    write_json(out / "gradient_audit.json", {"first_step": gradient_audit, "frozen_rgb_unchanged": True, "rgb_state_sha256": rgb_hash})
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": config,
                "epoch": epoch, "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(), "shuffle_rng_state": shuffle_rng.get_state()}, out / "last.pt")
    save_csv(out / "history.csv", history)
    write_json(out / "selection_complete.json", {"best_epoch": best_epoch, "validation_loss": best_loss,
               "checkpoint_sha256": file_hash(out / "best.pt"), "test_evaluated": False})
    print(f"[test] Selection fixed at epoch {best_epoch}; reloading checkpoint for one test evaluation", flush=True)
    summary = evaluate_test(out / "best.pt", dataset, splits["test"], targets["train"][0], out / "test_evaluation", args.device, args.batch_size)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(axes, ("loss", "direction_accuracy", "distance_mae_mm"), ("Loss", "Direction accuracy", "Distance MAE (mm)")):
        for split in ("train", "validation"):
            ax.plot([h["epoch"] for h in history], [h[f"{split}_{key}"] for h in history], label=split)
        ax.axvline(best_epoch, linestyle="--", color="gray", label="selected epoch")
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    write_json(out / "completion.json", {"complete": True, "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_loss": best_loss, "training_seconds": history[-1]["elapsed_s"],
        "test_evaluations": 1, "test": summary})
    print(f"[COMPLETE] test direction={summary['direction_accuracy']:.1%}, distance MAE={summary['distance_mae_mm']:.2f}mm, static proxy={summary['proxy_valid_count']}/100\n{out}", flush=True)


if __name__ == "__main__":
    main()
