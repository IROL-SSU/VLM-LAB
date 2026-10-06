#!/usr/bin/env python3
"""Train mirrored RGB-D observations with joint LEFT/RIGHT-distance actions.

Each output class is one executable direction-distance pair.  Supervision is a
set of every class that passes the static corridor and swept-AABB check, so the
network may learn any safe action but emits one argmax action at inference.
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
from train_translate_dataset import infer_cached, minibatches, save_csv, validate_splits
from train_translate_pilot import file_hash, state_hash
from translate_joint_action_model import (
    DIRECTION_NAMES,
    JointTranslateActionModel,
    decode,
    mirror_action_mask,
    mirror_points,
    preferred_action_mask,
    preferred_acceptable_set_loss,
)
from translate_pilot_geometry import DIRECTIONS
from translate_subgoal_model import INPUT_FILES, observation_batch, read_json


BASE_ACTION_STEP_MM = 5
BASE_ACTION_MIN_MM = 20
BASE_ACTION_MAX_MM = 250


def action_mask_for_objects(objects, action_distances_m):
    """Return LEFT block followed by RIGHT block of executable actions."""
    return torch.tensor([
        geometry_check(objects, direction, distance)["proxy_valid"]
        for direction in DIRECTION_NAMES
        for distance in action_distances_m
    ], dtype=torch.bool)


def build_action_catalog(dataset, rows):
    """Build a 5 mm grid and close training-only narrow-interval gaps.

    Some scenes have less than 5 mm between first corridor clearance and a
    collision.  For training scenes with no safe grid representative, add the
    registered canonical distance.  Validation and test never influence the
    catalog.
    """
    distances = {
        millimeters / 1000
        for millimeters in range(
            BASE_ACTION_MIN_MM,
            BASE_ACTION_MAX_MM + 1,
            BASE_ACTION_STEP_MM,
        )
    }
    additions = []
    for row in rows:
        folder = dataset / row["sample_dir"]
        objects = read_json(folder / "objects.json")
        label = read_json(folder / "label.json")
        catalog = sorted(distances)
        if not action_mask_for_objects(objects, catalog).any():
            distance = round(float(label["distance_m"]), 6)
            if not geometry_check(objects, label["direction"], distance)["proxy_valid"]:
                raise ValueError(f"Canonical action is not geometry-valid: {row['scene_id']}")
            distances.add(distance)
            additions.append({"scene_id": row["scene_id"], "distance_m": distance})
    catalog = sorted(distances)
    for row in rows:
        objects = read_json(dataset / row["sample_dir"] / "objects.json")
        if not action_mask_for_objects(objects, catalog).any():
            raise AssertionError(f"Action catalog does not cover {row['scene_id']}")
    return catalog, additions


def supervision_for(
    dataset,
    rows,
    action_distances_m,
    device,
    augment_mirror=False,
    preferred_margin_m=.01,
):
    labels = [read_json(dataset / row["sample_dir"] / "label.json") for row in rows]
    objects = [read_json(dataset / row["sample_dir"] / "objects.json") for row in rows]
    if any(
        label["direction"] not in DIRECTION_NAMES
        or not 0 < label["distance_m"] < .25
        for label in labels
    ):
        raise ValueError("Expected LEFT/RIGHT labels with distances in (0, 0.25m)")
    acceptable = torch.stack([
        action_mask_for_objects(item, action_distances_m) for item in objects
    ]).to(device)
    if not acceptable.any(dim=1).all():
        missing = [
            row["scene_id"]
            for row, mask in zip(rows, acceptable)
            if not bool(mask.any())
        ]
        raise ValueError(f"Scenes without a catalogued safe action: {missing}")
    preferred = preferred_action_mask(
        acceptable, action_distances_m, preferred_margin_m
    )
    canonical_direction = torch.tensor([
        DIRECTION_NAMES.index(label["direction"]) for label in labels
    ], dtype=torch.long, device=device)
    canonical_distance = torch.tensor([
        label["distance_m"] for label in labels
    ], dtype=torch.float32, device=device)
    if augment_mirror:
        acceptable = torch.cat([acceptable, mirror_action_mask(acceptable)], dim=0)
        preferred = torch.cat([preferred, mirror_action_mask(preferred)], dim=0)
        canonical_direction = torch.cat([
            canonical_direction, 1 - canonical_direction
        ])
        canonical_distance = torch.cat([
            canonical_distance, canonical_distance
        ])
    sign = torch.where(
        canonical_direction == 0,
        canonical_distance.new_tensor(-1.),
        canonical_distance.new_tensor(1.),
    )
    return {
        "labels": labels,
        "objects": objects,
        "acceptable_mask": acceptable,
        "preferred_mask": preferred,
        "canonical_direction": canonical_direction,
        "canonical_distance_m": canonical_distance,
        "canonical_signed_m": sign * canonical_distance,
    }


def target_subset(targets, idx):
    return {
        key: value[idx]
        for key, value in targets.items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def cache_joint_observations(
    model,
    dataset,
    rows,
    seed,
    device,
    output,
    split,
    augment_mirror=False,
    batch_size=8,
):
    original_rgb, original_points = [], []
    mirrored_rgb, mirrored_points = [], []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = observation_batch(dataset, rows[start:start + batch_size], seed)
        images = batch["image"].to(device)
        masks = batch["masks"].to(device)
        original_rgb.append(model.rgb_encoder(images, masks).cpu())
        original_points.append(batch["points"])
        if augment_mirror:
            mirrored_rgb.append(model.rgb_encoder(
                images.flip(-1), masks.flip(-1)
            ).cpu())
            mirrored_points.append(mirror_points(batch["points"]))
        if start % 80 == 0 or start + batch_size >= len(rows):
            suffix = " + mirrors" if augment_mirror else ""
            print(
                f"[cache {split}{suffix}] {min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    rgb_features = torch.cat(original_rgb)
    points = torch.cat(original_points)
    scene_ids = [row["scene_id"] for row in rows]
    if augment_mirror:
        rgb_features = torch.cat([rgb_features, torch.cat(mirrored_rgb)])
        points = torch.cat([points, torch.cat(mirrored_points)])
        scene_ids += [f"{row['scene_id']}::mirror" for row in rows]
    result = {"rgb_features": rgb_features, "points": points}
    torch.save({
        **result,
        "scene_ids": scene_ids,
        "sampling_seed": seed,
        "rgb_state_sha256": state_hash(model.rgb_encoder),
        "input_files": list(INPUT_FILES),
        "horizontal_mirror": augment_mirror,
    }, output / f"cache_{split}.pt")
    return {key: value.to(device) for key, value in result.items()}


def prediction_metrics(
    output,
    targets,
    action_distances_m,
    preferred_weight=.2,
):
    total, set_nll, ranking, preferred_nll = preferred_acceptable_set_loss(
        output,
        targets["acceptable_mask"],
        targets["preferred_mask"],
        preferred_weight=preferred_weight,
    )
    direction, distance, _, action_index = decode(output, action_distances_m)
    acceptable_hit = targets["acceptable_mask"].gather(
        1, action_index[:, None]
    ).squeeze(1)
    preferred_hit = targets["preferred_mask"].gather(
        1, action_index[:, None]
    ).squeeze(1)
    probabilities = output["action_logits"].softmax(dim=1)
    acceptable_mass = (
        probabilities * targets["acceptable_mask"].to(probabilities.dtype)
    ).sum(dim=1)
    sign = torch.where(
        direction == 0, distance.new_tensor(-1.), distance.new_tensor(1.)
    )
    signed = sign * distance
    direction_correct = direction == targets["canonical_direction"]
    return {
        "loss": float(total),
        "acceptable_set_nll": float(set_nll),
        "ranking_loss": float(ranking),
        "preferred_set_nll": float(preferred_nll),
        "acceptable_probability_mass": float(acceptable_mass.mean()),
        "safe_action_hit_rate": float(acceptable_hit.float().mean()),
        "safe_action_hit_count": int(acceptable_hit.sum()),
        "preferred_action_hit_rate": float(preferred_hit.float().mean()),
        "preferred_action_hit_count": int(preferred_hit.sum()),
        "direction_accuracy": float(direction_correct.float().mean()),
        "direction_correct_count": int(direction_correct.sum()),
        "distance_mae_mm": float(
            (distance - targets["canonical_distance_m"]).abs().mean() * 1000
        ),
        "signed_displacement_mae_mm": float(
            (signed - targets["canonical_signed_m"]).abs().mean() * 1000
        ),
        "mean_predicted_distance_mm": float(distance.mean() * 1000),
        "direction_and_5mm_count": int((
            direction_correct
            & ((distance - targets["canonical_distance_m"]).abs() <= .005)
        ).sum()),
    }


def selection_score(metrics):
    """Validation-only lexicographic selection."""
    return (
        metrics["safe_action_hit_count"],
        metrics["direction_correct_count"],
        -metrics["mean_predicted_distance_mm"],
        -metrics["loss"],
    )


def evaluate_test(checkpoint, dataset, rows, train_labels, out, device, batch_size):
    """Predict test observations before opening any test labels or geometry."""
    out.mkdir()
    digest = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    action_distances_m = config["action_distances_m"]
    ids = {row["scene_id"] for row in rows}
    assert not ids.intersection(
        config["train_scene_ids"] + config["validation_scene_ids"]
    )
    assert ids == set(config["test_scene_ids"])

    model = JointTranslateActionModel(action_distances_m).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    cache = cache_joint_observations(
        model, dataset, rows, config["seed"], device, out, "test"
    )
    output = infer_cached(model, cache, batch_size)
    direction, distances, confidences, action_indices = [
        value.cpu().tolist()
        for value in decode(output, action_distances_m)
    ]
    raw = [{
        "scene_id": row["scene_id"],
        "action_index": action_index,
        "direction": DIRECTION_NAMES[direction_index],
        "distance_m": distance,
        "confidence": confidence,
    } for row, action_index, direction_index, distance, confidence in zip(
        rows, action_indices, direction, distances, confidences
    )]
    write_json(out / "raw_predictions_before_gt.json", raw)

    targets = supervision_for(
        dataset,
        rows,
        action_distances_m,
        device,
        augment_mirror=False,
        preferred_margin_m=config["preferred_margin_max_m"],
    )
    metrics = prediction_metrics(
        output,
        targets,
        action_distances_m,
        preferred_weight=config["preferred_loss_weight"],
    )
    masks = targets["acceptable_mask"].cpu()
    preferred_masks = targets["preferred_mask"].cpu()
    predictions, previews = [], []
    for index, (row, prediction, label, objects) in enumerate(zip(
        rows, raw, targets["labels"], targets["objects"]
    )):
        geometry = geometry_check(
            objects, prediction["direction"], prediction["distance_m"]
        )
        acceptable = bool(masks[index, prediction["action_index"]])
        preferred = bool(preferred_masks[index, prediction["action_index"]])
        if acceptable != geometry["proxy_valid"]:
            raise AssertionError("Catalog supervision and geometry metric disagree")
        vector = np.asarray([*DIRECTIONS[prediction["direction"]], 0.])
        delta = vector * prediction["distance_m"]
        gt_delta = np.asarray(label["delta_shelf_m"])
        enriched = {
            **prediction,
            "ground_truth_direction": label["direction"],
            "ground_truth_distance_m": label["distance_m"],
            "direction_correct": prediction["direction"] == label["direction"],
            "distance_error_mm": abs(
                prediction["distance_m"] - label["distance_m"]
            ) * 1000,
            "displacement_error_mm": float(np.linalg.norm(delta - gt_delta) * 1000),
            "acceptable_action": acceptable,
            "preferred_action": preferred,
            "acceptable_action_count": int(masks[index].sum()),
            "opposite_direction_also_acceptable": bool(
                masks[index].reshape(2, -1)[
                    1 - DIRECTION_NAMES.index(label["direction"])
                ].any()
            ),
            "delta_shelf_m": delta.tolist(),
            "geometry": geometry,
        }
        predictions.append(enriched)
        if index < 20:
            folder = dataset / row["sample_dir"]
            image = evaluation_preview(
                folder, objects, label, enriched, "REUSED HELD-OUT TEST LAYOUT"
            )
            image.save(out / f"{row['scene_id']}.png")
            previews.append(image)

    metrics.update({
        "sample_count": len(rows),
        "checkpoint_epoch": state["epoch"],
        "checkpoint_sha256": digest,
        "distance_median_error_mm": float(np.median([
            prediction["distance_error_mm"] for prediction in predictions
        ])),
        "distance_p90_error_mm": float(np.percentile([
            prediction["distance_error_mm"] for prediction in predictions
        ], 90)),
        "distance_max_error_mm": max(
            prediction["distance_error_mm"] for prediction in predictions
        ),
        "displacement_mae_mm": float(np.mean([
            prediction["displacement_error_mm"] for prediction in predictions
        ])),
        "proxy_valid_count": sum(
            prediction["geometry"]["proxy_valid"] for prediction in predictions
        ),
        "corridor_clear_count": sum(
            prediction["geometry"]["corridor_clear"] for prediction in predictions
        ),
        "collision_free_count": sum(
            prediction["geometry"]["collision_free_swept_aabb"]
            for prediction in predictions
        ),
        "direction_and_proxy_valid_count": sum(
            prediction["direction_correct"] and prediction["geometry"]["proxy_valid"]
            for prediction in predictions
        ),
        "preferred_action_count": sum(
            prediction["preferred_action"] for prediction in predictions
        ),
        "opposite_direction_acceptable_scenes": sum(
            prediction["opposite_direction_also_acceptable"]
            for prediction in predictions
        ),
        "catalog_coverage_count": int(masks.any(dim=1).sum()),
        "constant_baseline": constant_baseline(train_labels, targets["labels"]),
        "protocol": (
            "checkpoint selected only on validation safe-action hits, canonical "
            "direction accuracy, mean movement distance, and loss; test evaluated once"
        ),
        "test_reuse_warning": (
            "This split was inspected in earlier experiments and is a reused comparison "
            "benchmark, not an untouched final test."
        ),
        "limitations": (
            "Mirroring adds symmetry supervision, not new physical layouts. Same assets "
            "and role pairs across splits; static AABB proxy only; no execution/grasp test."
        ),
    })
    if metrics["safe_action_hit_count"] != metrics["proxy_valid_count"]:
        raise AssertionError("Safe-action and geometry counts must match")
    metrics["by_direction"] = {}
    for direction_name in DIRECTION_NAMES:
        group = [
            prediction for prediction in predictions
            if prediction["ground_truth_direction"] == direction_name
        ]
        metrics["by_direction"][direction_name] = {
            "count": len(group),
            "correct": sum(item["direction_correct"] for item in group),
            "proxy_valid": sum(item["geometry"]["proxy_valid"] for item in group),
            "distance_mae_mm": float(np.mean([
                item["distance_error_mm"] for item in group
            ])),
        }
    metrics["confusion_matrix"] = {
        ground_truth: {
            predicted: sum(
                item["ground_truth_direction"] == ground_truth
                and item["direction"] == predicted
                for item in predictions
            )
            for predicted in DIRECTION_NAMES
        }
        for ground_truth in DIRECTION_NAMES
    }
    assert digest == file_hash(checkpoint), "Checkpoint mutated during test"
    metrics["checkpoint_unchanged"] = True
    write_json(out / "metrics.json", {
        "summary": metrics,
        "predictions": predictions,
    })
    save_csv(out / "predictions.csv", [{
        key: value
        for key, value in prediction.items()
        if key not in ("geometry", "delta_shelf_m")
    } for prediction in predictions])
    sheet = Image.new("RGB", (4 * 480, 5 * 360), "white")
    for index, image in enumerate(previews):
        sheet.paste(
            image.resize((480, 360)),
            ((index % 4) * 480, (index // 4) * 360),
        )
    sheet.save(out / "first20_predictions.jpg", quality=92)
    return metrics


def write_readme(out, config, validation_metrics, summary):
    baseline_path = ROOT / "experiments/translate_train_1000_20261001_v1/completion.json"
    signed_path = ROOT / "experiments/translate_signed_safe_1000_20261002_v1/completion.json"
    joint_path = ROOT / "experiments/translate_joint_bins_flip_1000_20261002_v1/completion.json"
    baseline = read_json(baseline_path)["test"] if baseline_path.exists() else None
    signed = read_json(signed_path)["test"] if signed_path.exists() else None
    joint = read_json(joint_path)["test"] if joint_path.exists() else None
    comparison_rows = []
    if baseline:
        comparison_rows.append((
            "8-output baseline",
            baseline["direction_accuracy"],
            baseline["distance_mae_mm"],
            baseline["proxy_valid_count"],
            baseline["corridor_clear_count"],
            baseline["collision_free_count"],
        ))
    if signed:
        comparison_rows.append((
            "Signed scalar + interval",
            signed["direction_accuracy"],
            signed["distance_mae_mm"],
            signed["proxy_valid_count"],
            signed["corridor_clear_count"],
            signed["collision_free_count"],
        ))
    if joint:
        comparison_rows.append((
            "Joint actions + mirror",
            joint["direction_accuracy"],
            joint["distance_mae_mm"],
            joint["proxy_valid_count"],
            joint["corridor_clear_count"],
            joint["collision_free_count"],
        ))
    comparison_rows.append((
        "Joint + mirror + preferred short-safe action",
        summary["direction_accuracy"],
        summary["distance_mae_mm"],
        summary["proxy_valid_count"],
        summary["corridor_clear_count"],
        summary["collision_free_count"],
    ))
    table = "\n".join(
        f"| {name} | {direction:.1%} | {distance:.2f} mm | {proxy}/100 | "
        f"{corridor}/100 | {collision}/100 |"
        for name, direction, distance, proxy, corridor, collision in comparison_rows
    )
    text = f"""# Joint actions with mirroring and a short-safe preference

The model classifies one action from {config['action_count']} joint LEFT/RIGHT and distance classes. Training uses original plus horizontally mirrored RGB, masks, and shelf-frame point clouds. Every catalog action passing the static geometry check remains acceptable. A {config['preferred_loss_weight']:.2f}-weight auxiliary target favors the shortest robust action after moving up to {config['preferred_margin_max_m']*1000:.0f} mm inward from both safe-interval edges.

## Selected checkpoint

- Epoch: {summary['checkpoint_epoch']}
- Validation safe action: {validation_metrics['safe_action_hit_count']}/100
- Validation canonical direction: {validation_metrics['direction_accuracy']:.1%}
- Validation preferred action: {validation_metrics['preferred_action_hit_count']}/100
- Validation mean predicted movement: {validation_metrics['mean_predicted_distance_mm']:.2f} mm

## Reused test benchmark

- Safe/static-proxy-valid action: {summary['proxy_valid_count']}/{summary['sample_count']}
- Canonical direction accuracy: {summary['direction_accuracy']:.1%}
- Canonical direction and proxy valid: {summary['direction_and_proxy_valid_count']}/{summary['sample_count']}
- Preferred short-safe action: {summary['preferred_action_count']}/{summary['sample_count']}
- Mean predicted movement: {summary['mean_predicted_distance_mm']:.2f} mm
- Exact-minimum distance MAE: {summary['distance_mae_mm']:.2f} mm
- Corridor clear: {summary['corridor_clear_count']}/{summary['sample_count']}
- Collision/boundary safe: {summary['collision_free_count']}/{summary['sample_count']}
- Test scenes where the opposite direction also has a safe catalog action: {summary['opposite_direction_acceptable_scenes']}/{summary['sample_count']}

## Controlled comparison

| Model | Canonical direction | Exact-min distance MAE | Static proxy valid | Corridor clear | Collision safe |
|---|---:|---:|---:|---:|---:|
{table}

## Interpretation limit

The test layouts were already inspected in earlier experiments. This is a controlled comparison on the same scenes, not an untouched final generalization result. Horizontal mirroring enforces symmetry but does not add genuinely new scene layouts.
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
        default=ROOT / "experiments/translate_joint_bins_flip_preferred_1000_20261002_v1",
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
    parser.add_argument("--preferred-margin-m", type=float, default=.01)
    parser.add_argument("--preferred-weight", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("Positive epochs, patience, batch size and learning rate required")
    if not 0 <= args.preferred_margin_m < .25 or args.preferred_weight < 0:
        raise ValueError("Invalid preferred margin or loss weight")
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

    action_distances_m, catalog_additions = build_action_catalog(
        dataset, splits["train"]
    )
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = JointTranslateActionModel(
        action_distances_m, args.rgb_weights
    ).to(args.device)
    rgb_hash = state_hash(model.rgb_encoder)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config = {
        "experiment": "1000-scenes-joint-actions-horizontal-mirror-preferred-v1",
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
            "lexicographic maximum validation safe-action hit count, canonical "
            "direction count, negative mean movement distance, negative loss"
        ),
        "initialization": (
            "fresh random trainable modules; frozen ImageNet RGB; no prior checkpoint reuse"
        ),
        "augmentation": (
            "training-only deterministic horizontal mirror of RGB and masks; shelf-X "
            "point reflection about x=0.28m; LEFT/RIGHT acceptable-action blocks swapped"
        ),
        "original_train_samples": len(splits["train"]),
        "augmented_train_samples": 2 * len(splits["train"]),
        "validation_augmentation": "none",
        "test_augmentation": "none",
        "point_sampling": (
            "fixed 512 target + 512 blocker + 1024 scene, then shelf-X reflection for mirror"
        ),
        "input_files": list(INPUT_FILES),
        "rgb_input_hw": [480, 640],
        "architecture": (
            "frozen ResNet18 role/global pooling + trainable PointNet-style geometry + "
            "fusion MLP + one joint-action classification head"
        ),
        "direction_names": list(DIRECTION_NAMES),
        "action_distances_m": action_distances_m,
        "action_count": 2 * len(action_distances_m),
        "base_action_grid": {
            "minimum_mm": BASE_ACTION_MIN_MM,
            "maximum_mm": BASE_ACTION_MAX_MM,
            "step_mm": BASE_ACTION_STEP_MM,
        },
        "training_only_catalog_additions": catalog_additions,
        "acceptable_targets": (
            "all joint actions passing corridor-clear and swept-AABB/boundary checks; "
            "both LEFT and RIGHT may be acceptable"
        ),
        "preferred_targets": (
            "shortest direction-level robust action after moving inward from both "
            "safe-interval edges by min(configured margin, discrete interval width/4)"
        ),
        "preferred_margin_max_m": args.preferred_margin_m,
        "preferred_loss_weight": args.preferred_weight,
        "loss": (
            "negative log total acceptable probability plus 0.25 times a 0.2-logit "
            "acceptable-vs-unacceptable argmax ranking hinge plus configured weight "
            "times negative log preferred-action probability"
        ),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
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
            "same assets/role pairs across splits; mirror is not a new layout; reused test; "
            "static AABB proxy only; no executor, rotation, or grasp"
        ),
    }
    write_json(out / "config.json", config)

    source = out / "source"
    source.mkdir()
    for name in (
        "train_translate_joint_actions.py",
        "translate_joint_action_model.py",
        "translate_subgoal_model.py",
        "train_translate_dataset.py",
        "train_translate_pilot.py",
        "evaluate_translate_pilot.py",
        "generate_translate_rgbd_pilot.py",
        "translate_pilot_geometry.py",
        "test_translate_joint_action_training.py",
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

    caches = {
        "train": cache_joint_observations(
            model,
            dataset,
            splits["train"],
            args.seed,
            args.device,
            out,
            "train",
            augment_mirror=True,
        ),
        "validation": cache_joint_observations(
            model,
            dataset,
            splits["validation"],
            args.seed,
            args.device,
            out,
            "validation",
        ),
    }
    targets = {
        "train": supervision_for(
            dataset,
            splits["train"],
            action_distances_m,
            args.device,
            augment_mirror=True,
            preferred_margin_m=args.preferred_margin_m,
        ),
        "validation": supervision_for(
            dataset,
            splits["validation"],
            action_distances_m,
            args.device,
            preferred_margin_m=args.preferred_margin_m,
        ),
    }
    if len(caches["train"]["points"]) != len(targets["train"]["acceptable_mask"]):
        raise AssertionError("Mirrored cache and targets are misaligned")

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
        for step, idx in enumerate(minibatches(
            len(caches["train"]["points"]),
            args.batch_size,
            shuffle_rng,
            shuffle=True,
        )):
            optimizer.zero_grad(set_to_none=True)
            output = model.from_features(
                caches["train"]["rgb_features"][idx],
                caches["train"]["points"][idx],
            )
            batch_targets = target_subset(targets["train"], idx)
            loss, _, _, _ = preferred_acceptable_set_loss(
                output,
                batch_targets["acceptable_mask"],
                batch_targets["preferred_mask"],
                preferred_weight=args.preferred_weight,
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
                    "action_head",
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
        for split in ("train", "validation"):
            split_output = infer_cached(model, caches[split], args.batch_size)
            metrics = prediction_metrics(
                split_output,
                targets[split],
                action_distances_m,
                preferred_weight=args.preferred_weight,
            )
            record.update({
                f"{split}_{key}": value for key, value in metrics.items()
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
            f"[epoch {epoch}] train safe={record['train_safe_action_hit_rate']:.1%} "
            f"dir={record['train_direction_accuracy']:.1%} | "
            f"val safe={record['validation_safe_action_hit_count']}/100 "
            f"dir={record['validation_direction_accuracy']:.1%} "
            f"preferred={record['validation_preferred_action_hit_count']}/100 "
            f"mean={record['validation_mean_predicted_distance_mm']:.1f}mm | "
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
        ("loss", "Set loss"),
        ("safe_action_hit_rate", "Safe action hit rate"),
        ("direction_accuracy", "Canonical direction accuracy"),
        ("mean_predicted_distance_mm", "Mean predicted movement (mm)"),
    )
    for axis, (key, title) in zip(axes, plots):
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
        f"[COMPLETE] test safe={summary['proxy_valid_count']}/100, "
        f"canonical direction={summary['direction_accuracy']:.1%}, "
        f"preferred={summary['preferred_action_count']}/100, "
        f"mean={summary['mean_predicted_distance_mm']:.1f}mm\n{out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
