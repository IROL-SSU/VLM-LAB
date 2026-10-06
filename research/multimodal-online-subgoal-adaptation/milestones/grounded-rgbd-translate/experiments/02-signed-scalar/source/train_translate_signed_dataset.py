#!/usr/bin/env python3
"""Train a LEFT/RIGHT signed-displacement model with safe-interval targets.

The visual and geometry encoders match the registered 1,000-scene baseline.
Direction and distance are coupled as one signed shelf-X displacement.  Model
selection uses validation static-goal validity first, then label direction
accuracy and exact signed-displacement error.  Test is evaluated once after the
checkpoint is fixed.
"""
from __future__ import annotations

import argparse
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
from train_translate_dataset import (
    cache_observations,
    infer_cached,
    minibatches,
    save_csv,
    validate_splits,
)
from train_translate_pilot import file_hash, state_hash
from translate_pilot_geometry import DIRECTIONS
from translate_signed_subgoal_model import (
    DIRECTION_NAMES,
    INPUT_FILES,
    SignedTranslateSubgoalModel,
    decode,
    read_json,
    safe_interval_loss,
    safe_signed_bounds,
    signed_exact_target,
)


def labels_for(dataset, rows, device, safe_margin_m):
    labels = [read_json(dataset / row["sample_dir"] / "label.json") for row in rows]
    if any(
        x["direction"] not in DIRECTION_NAMES
        or not 0 < x["distance_m"] < .25
        or x["maximum_clear_translation_m"] <= x["distance_m"]
        for x in labels
    ):
        raise ValueError("Expected feasible LEFT/RIGHT intervals inside the 0.25m output range")
    direction = torch.tensor(
        [DIRECTION_NAMES.index(x["direction"]) for x in labels],
        dtype=torch.long,
        device=device,
    )
    minimum = torch.tensor(
        [x["distance_m"] for x in labels], dtype=torch.float32, device=device
    )
    maximum = torch.tensor(
        [x["maximum_clear_translation_m"] for x in labels],
        dtype=torch.float32,
        device=device,
    )
    lower, upper, edge_margin = safe_signed_bounds(
        direction, minimum, maximum, safe_margin_m
    )
    return {
        "labels": labels,
        "direction": direction,
        "minimum_distance_m": minimum,
        "maximum_clear_distance_m": maximum,
        "signed_exact_m": signed_exact_target(direction, minimum),
        "safe_lower_signed_m": lower,
        "safe_upper_signed_m": upper,
        "edge_margin_m": edge_margin,
    }


def target_subset(targets, idx):
    return {
        key: value[idx] if isinstance(value, torch.Tensor) else value
        for key, value in targets.items()
        if key != "labels"
    }


def prediction_metrics(output, targets):
    loss, violation = safe_interval_loss(
        output,
        targets["safe_lower_signed_m"],
        targets["safe_upper_signed_m"],
    )
    pred_direction, pred_distance, pred_signed = decode(output)
    interval_hit = (
        (pred_signed >= targets["safe_lower_signed_m"])
        & (pred_signed <= targets["safe_upper_signed_m"])
    )
    return {
        "loss": float(loss),
        "direction_accuracy": float(
            (pred_direction == targets["direction"]).float().mean()
        ),
        "direction_correct_count": int(
            (pred_direction == targets["direction"]).sum()
        ),
        "distance_mae_mm": float(
            (pred_distance - targets["minimum_distance_m"]).abs().mean() * 1000
        ),
        "signed_displacement_mae_mm": float(
            (pred_signed - targets["signed_exact_m"]).abs().mean() * 1000
        ),
        "safe_interval_hit_rate": float(interval_hit.float().mean()),
        "safe_interval_hit_count": int(interval_hit.sum()),
        "safe_interval_violation_mae_mm": float(violation.mean() * 1000),
    }


def geometry_metrics(output, object_sets):
    pred_direction, pred_distance, _ = decode(output)
    names = [DIRECTION_NAMES[index] for index in pred_direction.cpu().tolist()]
    distances = pred_distance.cpu().tolist()
    checks = [
        geometry_check(objects, direction, distance)
        for objects, direction, distance in zip(object_sets, names, distances)
    ]
    return {
        "proxy_valid_count": sum(x["proxy_valid"] for x in checks),
        "proxy_valid_rate": float(np.mean([x["proxy_valid"] for x in checks])),
        "corridor_clear_count": sum(x["corridor_clear"] for x in checks),
        "collision_free_count": sum(x["collision_free_swept_aabb"] for x in checks),
    }


def selection_score(metrics):
    """Lexicographic validation-only checkpoint selection."""
    return (
        metrics["proxy_valid_count"],
        metrics["direction_correct_count"],
        -metrics["signed_displacement_mae_mm"],
        -metrics["loss"],
    )


def evaluate_test(checkpoint, dataset, rows, train_labels, out, device, batch_size):
    """Reload fixed checkpoint, predict all test inputs, then open test GT."""
    out.mkdir()
    digest = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    ids = {row["scene_id"] for row in rows}
    assert not ids.intersection(
        config["train_scene_ids"] + config["validation_scene_ids"]
    )
    assert ids == set(config["test_scene_ids"])

    model = SignedTranslateSubgoalModel().to(device)
    model.load_state_dict(state["model_state"], strict=True)
    cache = cache_observations(
        model, dataset, rows, config["seed"], device, out, "test"
    )
    output = infer_cached(model, cache, batch_size)
    direction_idx, distances, signed = [
        value.cpu().tolist() for value in decode(output)
    ]
    raw = [
        {
            "scene_id": row["scene_id"],
            "direction": DIRECTION_NAMES[index],
            "distance_m": distance,
            "signed_displacement_m": signed_distance,
        }
        for row, index, distance, signed_distance in zip(
            rows, direction_idx, distances, signed
        )
    ]
    write_json(out / "raw_predictions_before_gt.json", raw)

    targets = labels_for(dataset, rows, device, config["safe_margin_max_m"])
    lower = targets["safe_lower_signed_m"].cpu().tolist()
    upper = targets["safe_upper_signed_m"].cpu().tolist()
    exact_signed = targets["signed_exact_m"].cpu().tolist()
    predictions, previews = [], []
    for index, (row, prediction, label, low, high, exact) in enumerate(
        zip(rows, raw, targets["labels"], lower, upper, exact_signed)
    ):
        folder = dataset / row["sample_dir"]
        objects = read_json(folder / "objects.json")
        delta = np.asarray([*DIRECTIONS[prediction["direction"]], 0.]) * prediction["distance_m"]
        interval_violation = max(
            low - prediction["signed_displacement_m"],
            prediction["signed_displacement_m"] - high,
            0,
        )
        enriched = {
            **prediction,
            "ground_truth_direction": label["direction"],
            "ground_truth_distance_m": label["distance_m"],
            "ground_truth_signed_displacement_m": exact,
            "safe_lower_signed_m": low,
            "safe_upper_signed_m": high,
            "direction_correct": prediction["direction"] == label["direction"],
            "distance_error_mm": abs(prediction["distance_m"] - label["distance_m"]) * 1000,
            "signed_displacement_error_mm": abs(prediction["signed_displacement_m"] - exact) * 1000,
            "safe_interval_hit": low <= prediction["signed_displacement_m"] <= high,
            "safe_interval_violation_mm": interval_violation * 1000,
            "delta_shelf_m": delta.tolist(),
            "geometry": geometry_check(
                objects, prediction["direction"], prediction["distance_m"]
            ),
        }
        predictions.append(enriched)
        if index < 20:
            image = evaluation_preview(
                folder, objects, label, enriched, "REUSED HELD-OUT TEST LAYOUT"
            )
            image.save(out / f"{row['scene_id']}.png")
            previews.append(image)

    summary = prediction_metrics(output, targets)
    summary.update({
        "sample_count": len(rows),
        "checkpoint_epoch": state["epoch"],
        "checkpoint_sha256": digest,
        "distance_median_error_mm": float(np.median([
            p["distance_error_mm"] for p in predictions
        ])),
        "distance_p90_error_mm": float(np.percentile([
            p["distance_error_mm"] for p in predictions
        ], 90)),
        "distance_max_error_mm": max(
            p["distance_error_mm"] for p in predictions
        ),
        "proxy_valid_count": sum(p["geometry"]["proxy_valid"] for p in predictions),
        "corridor_clear_count": sum(
            p["geometry"]["corridor_clear"] for p in predictions
        ),
        "collision_free_count": sum(
            p["geometry"]["collision_free_swept_aabb"] for p in predictions
        ),
        "direction_and_5mm_count": sum(
            p["direction_correct"] and p["distance_error_mm"] <= 5
            for p in predictions
        ),
        "constant_baseline": constant_baseline(
            train_labels, targets["labels"]
        ),
        "protocol": (
            "checkpoint selected only on validation proxy validity, direction "
            "accuracy, and signed displacement error; test evaluated once for this run"
        ),
        "test_reuse_warning": (
            "This split was already analyzed during baseline development, so it is a "
            "reused benchmark rather than an untouched final test."
        ),
        "limitations": (
            "Same 15 assets and 50 role pairs across splits; layouts only held out. "
            "LEFT/RIGHT and static AABB proxy only; no executor/grasp validation."
        ),
    })
    summary["by_direction"] = {}
    for direction in DIRECTION_NAMES:
        group = [
            p for p in predictions if p["ground_truth_direction"] == direction
        ]
        summary["by_direction"][direction] = {
            "count": len(group),
            "correct": sum(p["direction_correct"] for p in group),
            "distance_mae_mm": float(np.mean([
                p["distance_error_mm"] for p in group
            ])),
            "proxy_valid": sum(p["geometry"]["proxy_valid"] for p in group),
        }
    summary["confusion_matrix"] = {
        gt: {
            pred: sum(
                p["ground_truth_direction"] == gt and p["direction"] == pred
                for p in predictions
            )
            for pred in DIRECTION_NAMES
        }
        for gt in DIRECTION_NAMES
    }
    assert digest == file_hash(checkpoint), "Checkpoint mutated during test"
    summary["checkpoint_unchanged"] = True
    write_json(out / "metrics.json", {
        "summary": summary,
        "predictions": predictions,
    })
    save_csv(out / "predictions.csv", [
        {
            key: value
            for key, value in prediction.items()
            if key not in ("geometry", "delta_shelf_m")
        }
        for prediction in predictions
    ])
    sheet = Image.new("RGB", (4 * 480, 5 * 360), "white")
    for index, image in enumerate(previews):
        sheet.paste(
            image.resize((480, 360)),
            ((index % 4) * 480, (index // 4) * 360),
        )
    sheet.save(out / "first20_predictions.jpg", quality=92)
    return summary


def write_readme(out, config, validation_metrics, summary):
    baseline_path = ROOT / "experiments/translate_train_1000_20261001_v1/completion.json"
    baseline = read_json(baseline_path)["test"] if baseline_path.exists() else None
    comparison = ""
    if baseline:
        comparison = f"""
## Registered baseline comparison

| Metric | 8-output baseline | Signed safe-interval model | Change |
|---|---:|---:|---:|
| Direction accuracy | {baseline['direction_accuracy']:.1%} | {summary['direction_accuracy']:.1%} | {(summary['direction_accuracy'] - baseline['direction_accuracy'])*100:+.1f} pp |
| Exact-minimum distance MAE | {baseline['distance_mae_mm']:.2f} mm | {summary['distance_mae_mm']:.2f} mm | {summary['distance_mae_mm'] - baseline['distance_mae_mm']:+.2f} mm |
| Signed displacement MAE | {baseline['displacement_mae_mm']:.2f} mm | {summary['signed_displacement_mae_mm']:.2f} mm | {summary['signed_displacement_mae_mm'] - baseline['displacement_mae_mm']:+.2f} mm |
| Static proxy valid | {baseline['proxy_valid_count']}/100 | {summary['proxy_valid_count']}/100 | {summary['proxy_valid_count'] - baseline['proxy_valid_count']:+d} |
"""
    text = f"""# Signed displacement and safe-interval training result

The model predicts one signed shelf-X displacement: negative is LEFT and positive is RIGHT. It was trained against a signed collision-free interval with an adaptive edge margin capped at {config['safe_margin_max_m']*1000:.0f} mm.

## Selected checkpoint

- Epoch: {summary['checkpoint_epoch']}
- Validation static proxy valid: {validation_metrics['proxy_valid_count']}/100
- Validation direction accuracy: {validation_metrics['direction_accuracy']:.1%}
- Validation signed displacement MAE against the minimum label: {validation_metrics['signed_displacement_mae_mm']:.2f} mm

## Reused test benchmark

- Direction accuracy: {summary['direction_accuracy']:.1%}
- Exact-minimum distance MAE: {summary['distance_mae_mm']:.2f} mm
- Signed displacement MAE: {summary['signed_displacement_mae_mm']:.2f} mm
- Safe interval hit: {summary['safe_interval_hit_count']}/{summary['sample_count']}
- Static proxy valid: {summary['proxy_valid_count']}/{summary['sample_count']}
- Corridor clear: {summary['corridor_clear_count']}/{summary['sample_count']}
- Collision/boundary safe: {summary['collision_free_count']}/{summary['sample_count']}
{comparison}
## Interpretation limit

The existing test split had already been inspected while developing the baseline and motivating this change. These numbers are useful for controlled comparison on the same scenes, but a newly generated untouched test set is required for a final generalization claim.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "experiments/translate_diverse_1000_20261001_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/translate_signed_safe_1000_20261002_v1",
    )
    parser.add_argument(
        "--rgb-weights",
        type=Path,
        default=ROOT / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--safe-margin-m", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("Positive epochs, patience, batch size and learning rate required")
    if not 0 <= args.safe_margin_m < .25:
        raise ValueError("safe-margin-m must be in [0, 0.25)")
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
    splits = validate_splits(
        read_json(dataset / "manifest.json"),
        {
            split: read_json(dataset / f"manifest_{split}.json")
            for split in ("train", "validation", "test")
        },
        read_json(dataset / "splits.json"),
    )
    if {key: len(value) for key, value in splits.items()} != {
        "train": 800,
        "validation": 100,
        "test": 100,
    }:
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

    model = SignedTranslateSubgoalModel(args.rgb_weights).to(args.device)
    rgb_hash = state_hash(model.rgb_encoder)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config = {
        "experiment": "1000-scenes-signed-safe-interval-v1",
        "dataset": str(dataset),
        "seed": args.seed,
        **{
            f"{split}_scene_ids": [row["scene_id"] for row in rows]
            for split, rows in splits.items()
        },
        "epochs_max": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "lr": args.lr,
        "scheduler": "cosine to 1e-5 over maximum epochs",
        "checkpoint_selection": (
            "lexicographic maximum validation static proxy validity, direction "
            "correct count, negative signed displacement MAE, negative interval loss"
        ),
        "initialization": (
            "fresh random trainable modules; ImageNet frozen RGB; no baseline checkpoint reuse"
        ),
        "augmentation": "none",
        "point_sampling": (
            "fixed; 512 target + 512 blocker + 1024 scene; same seed for all scenes"
        ),
        "input_files": list(INPUT_FILES),
        "rgb_input_hw": [480, 640],
        "architecture": (
            "baseline frozen ResNet18 role/global pooling + trainable PointNet-style "
            "geometry + fusion MLP + one tanh signed-displacement head"
        ),
        "direction_names": list(DIRECTION_NAMES),
        "distance_scale_m": .25,
        "safe_margin_max_m": args.safe_margin_m,
        "safe_interval": (
            "label minimum plus adaptive edge margin through clearance maximum minus "
            "adaptive edge margin; edge margin=min(configured maximum, feasible slack/4)"
        ),
        "output": (
            "one signed shelf-X displacement; negative=LEFT, positive=RIGHT, magnitude=distance"
        ),
        "loss": (
            "SmoothL1 of normalized distance outside the signed feasible interval; zero inside"
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "rgb_weights_sha256": pretrained_hash,
        "rgb_state_sha256": rgb_hash,
        "manifest_sha256": manifest_hash,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "python": sys.version,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "limitations": (
            "Same assets and role pairs across splits; reused test benchmark; static proxy; "
            "no executor, rotation, or grasp."
        ),
    }
    write_json(out / "config.json", config)

    source = out / "source"
    source.mkdir()
    for name in (
        "train_translate_signed_dataset.py",
        "translate_signed_subgoal_model.py",
        "translate_subgoal_model.py",
        "train_translate_dataset.py",
        "train_translate_pilot.py",
        "evaluate_translate_pilot.py",
        "generate_translate_rgbd_pilot.py",
        "translate_pilot_geometry.py",
        "test_translate_signed_training.py",
    ):
        shutil.copy2(Path(__file__).parent / name, source / name)

    fingerprints = {
        name: file_hash(dataset / name)
        for name in (
            "manifest.json",
            "manifest_train.json",
            "manifest_validation.json",
            "manifest_test.json",
            "splits.json",
            "validation.json",
        )
    }
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

    caches, targets, object_sets = {}, {}, {}
    for split in ("train", "validation"):
        caches[split] = cache_observations(
            model, dataset, splits[split], args.seed, args.device, out, split
        )
        targets[split] = labels_for(
            dataset, splits[split], args.device, args.safe_margin_m
        )
        object_sets[split] = [
            read_json(dataset / row["sample_dir"] / "objects.json")
            for row in splits[split]
        ]

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    shuffle_rng = torch.Generator().manual_seed(args.seed)
    history, best_score, best_epoch, best_validation, stale = [], None, 0, None, 0
    start = time.monotonic()
    gradient_audit = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, idx in enumerate(
            minibatches(
                len(splits["train"]), args.batch_size, shuffle_rng, shuffle=True
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model.from_features(
                caches["train"]["rgb_features"][idx],
                caches["train"]["points"][idx],
            )
            batch_targets = target_subset(targets["train"], idx)
            loss, _ = safe_interval_loss(
                output,
                batch_targets["safe_lower_signed_m"],
                batch_targets["safe_upper_signed_m"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if epoch == 1 and step == 0:
                for prefix in (
                    "rgb_encoder",
                    "geometry_encoder",
                    "rgb_projection",
                    "geo_projection",
                    "fusion",
                    "displacement_head",
                ):
                    params = [
                        parameter
                        for name, parameter in model.named_parameters()
                        if name.startswith(prefix)
                    ]
                    gradient_audit[prefix] = {
                        "grad_l2": sum(
                            float(parameter.grad.square().sum())
                            for parameter in params
                            if parameter.grad is not None
                        ) ** .5,
                        "parameters_with_grad": sum(
                            parameter.grad is not None for parameter in params
                        ),
                    }
                assert gradient_audit["rgb_encoder"]["parameters_with_grad"] == 0
                assert all(
                    values["grad_l2"] > 0
                    for name, values in gradient_audit.items()
                    if name != "rgb_encoder"
                )
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5)
            optimizer.step()

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.monotonic() - start,
        }
        cached_outputs = {}
        for split in ("train", "validation"):
            cached_outputs[split] = infer_cached(
                model, caches[split], args.batch_size
            )
            metrics = prediction_metrics(cached_outputs[split], targets[split])
            record.update({
                f"{split}_{key}": value for key, value in metrics.items()
            })
        validation_geometry = geometry_metrics(
            cached_outputs["validation"], object_sets["validation"]
        )
        record.update({
            f"validation_{key}": value
            for key, value in validation_geometry.items()
        })
        scheduler.step()

        validation_metrics = {
            key.removeprefix("validation_"): value
            for key, value in record.items()
            if key.startswith("validation_")
        }
        score = selection_score(validation_metrics)
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_validation = validation_metrics.copy()
            stale = 0
            torch.save({
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "config": config,
                "epoch": epoch,
                "validation_metrics": best_validation,
                "selection_score": list(score),
            }, out / "best.pt")
        else:
            stale += 1
        record["selected_best"] = improved
        record["selection_score"] = list(score)
        history.append(record)
        write_json(out / "history.json", history)
        write_json(out / "progress.json", {
            **record,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation,
            "stale_epochs": stale,
        })
        print(
            f"[epoch {epoch}] train acc={record['train_direction_accuracy']:.1%} "
            f"signedMAE={record['train_signed_displacement_mae_mm']:.2f}mm | "
            f"val proxy={record['validation_proxy_valid_count']}/100 "
            f"acc={record['validation_direction_accuracy']:.1%} "
            f"signedMAE={record['validation_signed_displacement_mae_mm']:.2f}mm | "
            f"best={best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            print(
                f"[stop] No validation selection improvement for {stale} epochs",
                flush=True,
            )
            break

    assert rgb_hash == state_hash(model.rgb_encoder), (
        "Frozen RGB weights or BN buffers changed"
    )
    write_json(out / "gradient_audit.json", {
        "first_step": gradient_audit,
        "frozen_rgb_unchanged": True,
        "rgb_state_sha256": rgb_hash,
    })
    torch.save({
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "config": config,
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "shuffle_rng_state": shuffle_rng.get_state(),
    }, out / "last.pt")
    save_csv(out / "history.csv", history)
    write_json(out / "selection_complete.json", {
        "best_epoch": best_epoch,
        "validation_metrics": best_validation,
        "selection_score": list(best_score),
        "checkpoint_sha256": file_hash(out / "best.pt"),
        "test_evaluated": False,
    })

    print(
        f"[test] Selection fixed at epoch {best_epoch}; reloading checkpoint once",
        flush=True,
    )
    summary = evaluate_test(
        out / "best.pt",
        dataset,
        splits["test"],
        targets["train"]["labels"],
        out / "test_evaluation",
        args.device,
        args.batch_size,
    )

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    plots = (
        ("loss", "Safe interval loss"),
        ("direction_accuracy", "Direction accuracy"),
        ("signed_displacement_mae_mm", "Signed displacement MAE (mm)"),
    )
    for axis, (key, title) in zip(axes[:3], plots):
        for split in ("train", "validation"):
            axis.plot(
                [item["epoch"] for item in history],
                [item[f"{split}_{key}"] for item in history],
                label=split,
            )
        axis.axvline(best_epoch, linestyle="--", color="gray", label="selected")
        axis.set(xlabel="Epoch", title=title)
        axis.grid(alpha=.25)
        axis.legend()
    axes[3].plot(
        [item["epoch"] for item in history],
        [item["validation_proxy_valid_count"] for item in history],
        label="validation",
    )
    axes[3].axvline(best_epoch, linestyle="--", color="gray", label="selected")
    axes[3].set(xlabel="Epoch", title="Validation static proxy valid", ylim=(0, 100))
    axes[3].grid(alpha=.25)
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)

    completion_record = {
        "complete": True,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_metrics": best_validation,
        "training_seconds": history[-1]["elapsed_s"],
        "test_evaluations": 1,
        "test": summary,
    }
    write_json(out / "completion.json", completion_record)
    write_readme(out, config, best_validation, summary)
    print(
        f"[COMPLETE] test direction={summary['direction_accuracy']:.1%}, "
        f"signedMAE={summary['signed_displacement_mae_mm']:.2f}mm, "
        f"static proxy={summary['proxy_valid_count']}/100\n{out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
