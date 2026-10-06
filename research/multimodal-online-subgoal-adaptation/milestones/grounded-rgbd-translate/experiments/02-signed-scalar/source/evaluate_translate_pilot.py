#!/usr/bin/env python3
"""Reload checkpoint, predict from observations, THEN read labels / audit GT.

No label-dependent snapping, direction replacement or geometry-based correction
is applied to model outputs. Goal validity is a separate static-AABB check.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from generate_translate_rgbd_pilot import ROOT, font, project, write_json
from translate_pilot_geometry import DIRECTIONS, Rect, clearance_limit, target_corridor
from translate_subgoal_model import DIRECTION_NAMES, INPUT_FILES, TranslateSubgoalModel, decode, observation_batch, read_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constant_baseline(training_labels, evaluation_labels):
    directions = [r["direction"] for r in training_labels]
    majority = sorted(set(directions), key=lambda d: (-directions.count(d), d))[0]
    mean_distance = float(np.mean([r["distance_m"] for r in training_labels]))
    return {"fitted_on": "training labels only", "direction": majority,
            "direction_accuracy": sum(r["direction"] == majority for r in evaluation_labels) / len(evaluation_labels),
            "distance_m": mean_distance,
            "distance_mae_mm": float(np.mean([abs(r["distance_m"] - mean_distance) for r in evaluation_labels]) * 1000)}


def geometry_check(objects, direction_name, distance_m):
    rects = [Rect.from_list(obj["footprint_shelf_m"]) for obj in objects]
    direction = DIRECTIONS[direction_name]
    delta = np.asarray(direction) * distance_m
    goal = rects[1].moved(*delta)
    corridor = target_corridor(rects[0])
    overlap = goal.overlap(corridor)
    limit = clearance_limit(rects[1], direction, [rects[0], *rects[2:]])
    collision_free = distance_m < limit and goal.inside()
    return {"corridor_overlap_m2": float(overlap), "corridor_clear": bool(overlap <= 1e-10),
            "collision_free_swept_aabb": bool(collision_free), "clearance_limit_m": float(limit),
            "proxy_valid": bool(overlap <= 1e-10 and collision_free)}


def arrow(draw, start, end, color, width):
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    draw.polygon([end, (end[0] - 13 * math.cos(angle - .5), end[1] - 13 * math.sin(angle - .5)),
                  (end[0] - 13 * math.cos(angle + .5), end[1] - 13 * math.sin(angle + .5))], fill=color)


def evaluation_preview(folder, objects, label, prediction, split_caption="TRAINING SAMPLE (not held out)"):
    image = Image.open(folder / "rgb.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    calib = read_json(folder / "calibration.json")
    # GT reference position is used only as a visualization anchor, never input.
    anchor = np.asarray(objects[1]["reference_point_shelf_m"])
    start = project(anchor, calib)
    gt_end = project(anchor + np.asarray(label["delta_shelf_m"]), calib)
    pred_end = project(anchor + np.asarray(prediction["delta_shelf_m"]), calib)
    arrow(draw, start, gt_end, "#ffbc38", 10)
    arrow(draw, start, pred_end, "#30c8ff", 4)
    for filename, color in (("target_mask.png", "#30ff80"), ("blocker_mask.png", "#ffbc38")):
        yy, xx = np.where(np.array(Image.open(folder / filename)) > 0)
        draw.rectangle([int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())], outline=color, width=2)
    draw.rectangle([0, 0, image.width, 74], fill="#202020")
    draw.text((12, 6), f'{prediction["scene_id"]} | {split_caption}', font=font(21), fill="white")
    draw.text((12, 34), f'GT {label["direction"]} {label["distance_m"]*100:.2f} cm | PRED {prediction["direction"]} {prediction["distance_m"]*100:.2f} cm | err {prediction["distance_error_mm"]:.3f} mm', font=font(19), fill="white")
    draw.rectangle([0, image.height - 35, image.width, image.height], fill="#202020")
    draw.text((12, image.height - 29), f'Orange: GT | Cyan: prediction | static proxy valid: {prediction["geometry"]["proxy_valid"]}', font=font(19), fill="white")
    return image


def evaluate(checkpoint, dataset, output, device="cpu", sampling_seed=None):
    checkpoint, dataset, output = Path(checkpoint).resolve(), Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_hash_before = file_hash(checkpoint)
    config = state["config"]
    if tuple(config["direction_names"]) != DIRECTION_NAMES:
        raise ValueError("Checkpoint direction order mismatch")
    manifest = read_json(dataset / "manifest.json")
    rows = manifest["samples"]
    same_training_dataset = str(dataset) == config["dataset"]
    overlap = sorted(set(row["scene_id"] for row in rows) & set(config["train_scene_ids"]))
    training_root = Path(config["dataset"])
    training_manifest = read_json(training_root / "manifest.json")
    training_input_hashes = [{name: file_hash(training_root / row["sample_dir"] / name) for name in INPUT_FILES}
                             for row in training_manifest["samples"]]
    evaluation_input_hashes = [{name: file_hash(dataset / row["sample_dir"] / name) for name in INPUT_FILES} for row in rows]
    identical_observations = [row["scene_id"] for row, hashes in zip(rows, evaluation_input_hashes) if hashes in training_input_hashes]
    paired = manifest.get("paired_validation")
    held_out_layouts = bool(paired and not same_training_dataset and not overlap and not identical_observations)
    if paired:
        assert Path(paired["training_dataset"]).resolve() == training_root.resolve()
        assert held_out_layouts, "Paired validation contains training IDs or observations"
        assert read_json(dataset / "selection_complete.json")["manifest_sha256"] == file_hash(dataset / "manifest.json")
    split_caption = "HELD-OUT LAYOUT / NO RETRAIN" if held_out_layouts else "TRAINING SAMPLE (not held out)" if same_training_dataset else "EXTERNAL SAMPLE"
    model = TranslateSubgoalModel()
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    seed = config["seed"] if sampling_seed is None else sampling_seed
    batch = {key: value.to(device) for key, value in observation_batch(dataset, rows, seed).items()}
    with torch.no_grad():
        result = model(**batch)  # Full RGB + geometry inference; NOT cached training outputs.
        dirs, distances, confidences = decode(result)
    predicted = [{"direction": DIRECTION_NAMES[index], "distance_m": float(distance), "confidence": float(confidence)}
                 for index, distance, confidence in zip(dirs.cpu().tolist(), distances.cpu().tolist(), confidences.cpu().tolist())]
    # Ground-truth files are first opened only AFTER predictions have been made.
    if paired:
        from validate_translate_rgbd_pilot import validate
        validate(dataset)
    output.mkdir(parents=True)
    predictions, images, target_distances, target_directions = [], [], [], []
    for row, prediction in zip(rows, predicted):
        folder = dataset / row["sample_dir"]
        label, objects = read_json(folder / "label.json"), read_json(folder / "objects.json")
        target_distances.append(label["distance_m"])
        target_directions.append(label["direction"])
        vector = np.asarray([*DIRECTIONS[prediction["direction"]], 0.])
        delta = vector * prediction["distance_m"]
        prediction.update({"scene_id": row["scene_id"], "target_asset": row["target_asset"], "blocker_asset": row["blocker_asset"],
                           "ground_truth_direction": label["direction"], "ground_truth_distance_m": label["distance_m"],
                           "direction_correct": prediction["direction"] == label["direction"],
                           "distance_error_mm": abs(prediction["distance_m"] - label["distance_m"]) * 1000,
                           "displacement_error_mm": float(np.linalg.norm(delta - label["delta_shelf_m"]) * 1000),
                           "angular_error_deg": math.degrees(math.acos(float(np.clip(vector @ np.asarray(label["direction_vector_shelf"]), -1, 1)))),
                           "delta_shelf_m": delta.tolist(), "geometry": geometry_check(objects, prediction["direction"], prediction["distance_m"])})
        predictions.append(prediction)
        image = evaluation_preview(folder, objects, label, prediction, split_caption)
        image.save(output / f'{row["scene_id"]}.png')
        images.append(image)
    training_labels = [read_json(training_root / row["sample_dir"] / "label.json") for row in training_manifest["samples"]]
    evaluation_labels = [{"direction": d, "distance_m": s} for d, s in zip(target_directions, target_distances)]
    summary = {
        "sample_count": len(rows), "same_training_dataset": same_training_dataset,
        "training_scene_id_overlap": overlap,
        "identical_training_observations": identical_observations,
        "held_out_layouts_same_assets": held_out_layouts,
        "protocol": "paired held-out layout challenges, same assets, no retraining" if held_out_layouts else "resubstitution / training-set overfit check" if same_training_dataset else "external dataset (independence not automatically guaranteed)",
        "direction_accuracy": float(np.mean([p["direction_correct"] for p in predictions])),
        "distance_mae_mm": float(np.mean([p["distance_error_mm"] for p in predictions])),
        "distance_max_error_mm": max(p["distance_error_mm"] for p in predictions),
        "displacement_mae_mm": float(np.mean([p["displacement_error_mm"] for p in predictions])),
        "proxy_valid_count": sum(p["geometry"]["proxy_valid"] for p in predictions),
        "overfit_debug_pass": all(p["direction_correct"] and p["distance_error_mm"] <= 5 for p in predictions) if same_training_dataset else None,
        "direction_and_5mm_check_pass": all(p["direction_correct"] and p["distance_error_mm"] <= 5 for p in predictions),
        "constant_baseline": constant_baseline(training_labels, evaluation_labels),
        "sampling_seed": seed, "checkpoint_step": state["step"],
        "limitations": ("Paired cases are diagnostic, not a representative test set. Same object identities; only layouts held out. " if held_out_layouts else "No held-out result for training scenes; external data independence requires audit. ") + "No executor/grasp validation, only static-AABB proxy audit. Uncorrected continuous outputs.",
    }
    checkpoint_hash_after = file_hash(checkpoint)
    assert checkpoint_hash_before == checkpoint_hash_after, "Checkpoint changed during evaluation"
    summary["checkpoint_sha256"] = checkpoint_hash_after
    summary["checkpoint_unchanged"] = True
    report = {"checkpoint": str(checkpoint), "dataset": str(dataset), "summary": summary, "predictions": predictions}
    write_json(output / "metrics.json", report)
    fields = ("scene_id", "ground_truth_direction", "direction", "ground_truth_distance_m", "distance_m", "direction_correct", "distance_error_mm", "displacement_error_mm", "confidence")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    sheet = Image.new("RGB", (960, 360 * len(images)), "white")
    for index, image in enumerate(images):
        sheet.paste(image.resize((480, 360)), (0, 360 * index))
        p = predictions[index]
        draw = ImageDraw.Draw(sheet)
        lines = [p["scene_id"], f'GT: {p["ground_truth_direction"]} {p["ground_truth_distance_m"]*100:.2f} cm',
                 f'Prediction: {p["direction"]} {p["distance_m"]*100:.3f} cm', f'Distance error: {p["distance_error_mm"]:.4f} mm',
                 f'Static geometry valid: {p["geometry"]["proxy_valid"]}', split_caption]
        for line, text in enumerate(lines):
            draw.text((500, 360 * index + 55 + line * 37), text, font=font(20), fill="black")
    sheet.save(output / "predictions_contact_sheet.png")
    if same_training_dataset and (checkpoint.parent / "history.json").exists():
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history = read_json(checkpoint.parent / "history.json")
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        steps = [h["step"] for h in history]
        axes[0].semilogy(steps, [max(h["loss"], 1e-9) for h in history])
        axes[0].set_title("Training loss")
        axes[1].plot(steps, [h["direction_accuracy"] * 100 for h in history])
        axes[1].set_title("Training direction accuracy (%)")
        axes[1].set_ylim(-2, 102)
        axes[2].semilogy(steps, [max(h["distance_mae_mm"], 1e-6) for h in history])
        axes[2].set_title("Training distance MAE (mm)")
        for ax in axes:
            ax.set_xlabel("Optimizer steps")
            ax.grid(alpha=.25)
        fig.suptitle("5 training scenes only — no held-out evaluation")
        fig.tight_layout()
        fig.savefig(output / "learning_curves.png", dpi=160)
        plt.close(fig)
    print(f"[eval] direction={summary['direction_accuracy']:.0%} MAE={summary['distance_mae_mm']:.4f}mm proxy={summary['proxy_valid_count']}/{len(rows)}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sampling-seed", type=int)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.dataset, args.output, args.device, args.sampling_seed)
