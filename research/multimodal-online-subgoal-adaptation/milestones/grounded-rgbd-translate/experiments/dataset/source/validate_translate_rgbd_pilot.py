#!/usr/bin/env python3
"""Audit generated TRANSLATE samples without starting Isaac Sim."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from generate_translate_rgbd_pilot import backproject
from revise_translate_pilot_visibility import acceptable, visibility_metrics
from translate_pilot_geometry import Rect, target_corridor, translation_candidates


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(folder):
    manifest = read(folder / "manifest.json")
    samples = manifest["samples"]
    assert len(samples) == manifest["sample_count"]
    role_assets = [row[key] for row in samples for key in ("target_asset", "blocker_asset")]
    bulk = manifest.get("bulk_generation")
    if not bulk:
        assert len(role_assets) == len(set(role_assets)), "Repeated target/blocker asset"
    results = []
    paired = manifest.get("paired_validation")
    if paired:
        training_root = Path(paired["training_dataset"])
        training_manifest = read(training_root / "manifest.json")
        training_ids = {r["scene_id"] for r in training_manifest["samples"]}
        training_rgb_hashes = {hashlib.sha256((training_root / r["sample_dir"] / "rgb.png").read_bytes()).hexdigest() for r in training_manifest["samples"]}
        assert manifest["split"] == "validation" and not training_ids.intersection(r["scene_id"] for r in samples)
    for sample_index, row in enumerate(samples):
        path = folder / row["sample_dir"]
        rgb = np.asarray(Image.open(path / "rgb.png").convert("RGB"))
        depth = np.load(path / "depth_z_m.npy", allow_pickle=False)
        ids = np.asarray(Image.open(path / "instance_ids.png"))
        target_mask = np.asarray(Image.open(path / "target_mask.png")) > 0
        blocker_mask = np.asarray(Image.open(path / "blocker_mask.png")) > 0
        valid = np.asarray(Image.open(path / "depth_valid.png")) > 0
        label = read(path / "label.json")
        objects = read(path / "objects.json")
        calib = read(path / "calibration.json")
        assert rgb.shape[:2] == depth.shape == ids.shape == target_mask.shape == blocker_mask.shape
        assert depth.dtype == np.float32 and np.isfinite(depth).all() and (depth >= 0).all()
        assert np.array_equal(valid, depth > 0)
        assert np.array_equal(target_mask, ids == row["target_id"])
        assert np.array_equal(blocker_mask, ids == row["blocker_id"])
        assert not (target_mask & blocker_mask).any()
        if bulk:
            reference = np.asarray(Image.open(path / "target_reference_mask.png")) > 0
            visibility = visibility_metrics(ids, reference)
            assert .40 <= visibility["visible_target_fraction"] <= .80
            assert .20 <= visibility["blocker_occlusion_fraction"] <= .60
            assert visibility["other_occlusion_fraction"] <= .10
            assert visibility == read(path / "visibility_audit.json") == row["visibility"]
            assert row["direction"] == label["direction"] in ("LEFT", "RIGHT")
            assert row["label"] == label
            assert hashlib.sha256((path / "rgb.png").read_bytes()).hexdigest() == row["rgb_sha256"]
        if paired:
            src = training_root / row["source_sample_dir"]
            original_objects, original_label = read(src / "objects.json"), read(src / "label.json")
            assert calib == read(src / "calibration.json"), "Camera changed"
            assert [o["asset"] for o in objects] == [o["asset"] for o in original_objects], "Object identities changed"
            assert label["direction"] == {"LEFT": "RIGHT", "RIGHT": "LEFT"}[original_label["direction"]]
            assert abs(label["distance_m"] - original_label["distance_m"]) >= .01 - 1e-9
            for obj, old in zip(objects, original_objects):
                assert obj["yaw_deg"] == old["yaw_deg"]
                assert np.linalg.norm(np.asarray(obj["reference_point_shelf_m"][:2]) - old["reference_point_shelf_m"][:2]) >= .03
            assert hashlib.sha256((path / "rgb.png").read_bytes()).hexdigest() not in training_rgb_hashes, "Training RGB duplicate"
            reference = np.asarray(Image.open(path / "target_reference_mask.png")) > 0
            metrics = visibility_metrics(ids, reference)
            assert acceptable(metrics), "Validation visibility outside policy"
            assert metrics == read(path / "visibility_audit.json")["after"]
        revision = manifest.get("revision", {})
        if row["scene_id"] in revision.get("changed_scenes", []):
            reference = np.asarray(Image.open(path / "target_reference_mask.png")) > 0
            metrics = visibility_metrics(ids, reference)
            assert acceptable(metrics), "Target visibility / blocker occlusion outside policy"
            audit = read(path / "visibility_audit.json")
            assert metrics == audit["after"], "Visibility audit does not match saved masks"
            source = Path(revision["source"]) / row["sample_dir"]
            original_ids = np.asarray(Image.open(source / "instance_ids.png"))
            assert visibility_metrics(original_ids, reference) == audit["before"]
            original_objects = read(source / "objects.json")
            assert objects[:1] + objects[2:] == original_objects[:1] + original_objects[2:], "Non-blocker objects changed"
            assert calib == read(source / "calibration.json"), "Camera changed"
            for key in ("asset", "yaw_deg", "instance_id", "usd_root"):
                assert objects[1][key] == original_objects[1][key]
            assert np.allclose(objects[1]["reference_point_shelf_m"][1:], original_objects[1]["reference_point_shelf_m"][1:])
        elif revision:
            source = Path(revision["source"]) / row["sample_dir"]
            assert {f.name for f in path.iterdir()} == {f.name for f in source.iterdir()}
            assert all(f.read_bytes() == (source / f.name).read_bytes() for f in path.iterdir()), "Untouched scene changed"
        assert len(set(obj["asset"] for obj in objects)) == len(objects)
        assert (2 <= len(objects) - 2 <= 7) if bulk else (3 <= len(objects) - 2 <= 5)
        assert len(row["distractor_assets"]) == len(objects) - 2
        assert objects[0]["asset"] == row["target_asset"] and objects[1]["asset"] == row["blocker_asset"]
        rects = [Rect.from_list(obj["footprint_shelf_m"]) for obj in objects]
        assert all(rect.inside() for rect in rects)
        assert all(a.overlap(b) < 1e-10 for i, a in enumerate(rects) for b in rects[i + 1:])
        candidates = translation_candidates(rects[0], rects[1], rects[2:])
        assert candidates, "No valid goal exists under the declared proxy"
        assert any(item["direction"] == label["direction"] and abs(item["distance_m"] - label["distance_m"]) < 1e-6 for item in candidates)
        assert abs(candidates[0]["distance_m"] - label["distance_m"]) < 1e-6, "Selected goal is not the shortest sampled valid candidate"
        direction = np.asarray(label["direction_vector_shelf"])
        delta = np.asarray(label["delta_shelf_m"])
        assert np.isclose(np.linalg.norm(direction), 1)
        assert np.allclose(direction * label["distance_m"], delta)
        assert label["action"] == "TRANSLATE" and label["object_id"] == row["blocker_id"]
        assert np.allclose(np.asarray(label["current_reference_point_shelf_m"]) + delta, label["goal_reference_point_shelf_m"])
        corridor = target_corridor(rects[0])
        initial_overlap = rects[1].overlap(corridor)
        goal_overlap = rects[1].moved(*delta[:2]).overlap(corridor)
        assert initial_overlap > 0 and goal_overlap < 1e-10
        full_cloud = backproject(depth, calib)
        per_object = []
        for obj in objects:
            mask = ids == obj["instance_id"]
            assert mask.sum() >= 50
            observed = full_cloud[mask & valid]
            bounds = obj["aabb_shelf_m"]
            fraction = float(np.mean(np.all((observed >= np.asarray(bounds["min"]) - 0.008) & (observed <= np.asarray(bounds["max"]) + 0.008), axis=1)))
            assert (mask & valid).sum() / mask.sum() >= 0.98
            assert fraction >= 0.95, "Backprojected mask points do not match the world geometry"
            per_object.append({"asset": obj["asset"], "role": obj["role"], "visible_pixels": int(mask.sum()), "points_inside_aabb_fraction": fraction})
        with np.load(path / "pointcloud.npz", allow_pickle=False) as cloud:
            uv = cloud["pixel_uv"]
            assert len(uv) > 1000
            assert np.allclose(cloud["xyz_shelf_m"], full_cloud[uv[:, 1], uv[:, 0]], atol=2e-6)
            assert np.array_equal(cloud["instance_id"], ids[uv[:, 1], uv[:, 0]])
            assert np.array_equal(cloud["rgb"], rgb[uv[:, 1], uv[:, 0]])
            point_count = len(uv)
        goal_rgb = np.asarray(Image.open(path / "goal_rgb.png").convert("RGB"))
        changed_pixels = int(np.count_nonzero(np.max(np.abs(rgb.astype(np.int16) - goal_rgb.astype(np.int16)), axis=-1) > 25))
        assert changed_pixels > 100, "Goal render appears unchanged"
        assert (path / "scene.usda").stat().st_size > 100
        results.append({"scene_id": row["scene_id"], "target": row["target_asset"], "blocker": row["blocker_asset"],
                        "direction": label["direction"], "distance_cm": round(label["distance_m"] * 100, 3),
                        "point_count": point_count, "goal_render_changed_pixels": changed_pixels,
                        "initial_corridor_overlap_m2": initial_overlap, "goal_corridor_overlap_m2": goal_overlap,
                        "objects": per_object})
        if bulk and (sample_index + 1) % 25 == 0:
            print(f"[audit] {sample_index + 1}/{len(samples)} passed", flush=True)
    return {"passed": True, "scene_count": len(samples), "unique_target_and_blocker_assets": len(set(role_assets)),
            "checks": ["role assets may repeat across scenes" if bulk else "unique role assets", "distinct objects within each scene", "aligned RGB-D/masks", "metric calibration against object bounds", "pointcloud-to-pixel correspondence", "non-overlapping initial geometry", "proxy-minimal translation label", "clear target corridor at goal", "goal RGB differs from initial RGB"],
            "revision_checks": ["revised target silhouette visibility 50-75%", "revised blocker occlusion 25-50%", "only blocker X changed", "unrevised samples byte-identical"] if manifest.get("revision") else [],
            "paired_validation_checks": ["new scene IDs and RGB", "same identities/yaws/camera", "each object XY changed >=3cm", "opposite shortest-label direction", "distance label differs >=1cm", "target visible 50-75% and blocker occlusion 25-50%"] if paired else [],
            "limitation": "Static AABB proxy audit; not manipulation or grasp execution success.", "samples": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset.resolve())
    (args.dataset / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "scene_count": report["scene_count"], "unique_role_assets": report["unique_target_and_blocker_assets"], "goals": [{k: row[k] for k in ("scene_id", "target", "blocker", "direction", "distance_cm")} for row in report["samples"]]}, ensure_ascii=False, indent=2))
