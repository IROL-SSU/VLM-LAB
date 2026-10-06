#!/usr/bin/env python3
"""Generate resumable, balanced, diverse 1000-scene observed RGB-D dataset.

50 ordered role pairs x 20 independent layouts; 500 LEFT / 500 RIGHT. Same
conservative 8-direction minimum-distance label policy as the pilot. No model
checkpoint is loaded. Visibility is measured using a target-only render.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_translate_rgbd_pilot import (
    ROOT, WIDTH, HEIGHT, CAMERA_PATH, SHELF_Z_CM, annotated_preview, backproject,
    calibration, font, topdown_preview, write_json,
)
from revise_translate_pilot_visibility import read, visibility_metrics
from translate_pilot_geometry import BOUNDS, sample_layout

ASSETS = ("mug_7", "Mug_2", "Mug_4", "cup_7", "cup_8", "Cup_2", "Cup_4",
          "acafela", "cantata", "coldgrape", "top", "cocopalm", "biracsikhye", "letsbe", "minutemad")
PALETTES = {
    "light_oak": ((.80, .65, .45), (.65, .49, .31)),
    "walnut": ((.38, .24, .14), (.25, .15, .08)),
    "ivory": ((.82, .80, .73), (.66, .64, .59)),
    "slate": ((.34, .40, .44), (.22, .27, .30)),
    "sage": ((.50, .61, .48), (.34, .43, .32)),
}
DISTANCE_BINS = {"short": (.02, .05), "medium": (.05, .09), "long": (.09, .251)}


def distance_bin(distance):
    return next(name for name, (lo, hi) in DISTANCE_BINS.items() if lo - 1e-9 <= distance < hi - 1e-9)


def make_schedule(seed=20261003):
    rng = random.Random(seed)
    names = list(ASSETS)
    rng.shuffle(names)
    pairs = [("cup_7", "Cup_2"), ("letsbe", "cantata"), ("coldgrape", "Cup_4"),
             ("Mug_2", "Mug_4"), ("cocopalm", "biracsikhye")]
    # Spread roles over the full asset catalog, then add non-duplicate pairs.
    for shift in (1, 4, 7, 10):
        for i, name in enumerate(names):
            pair = (name, names[(i + shift) % len(names)])
            if pair not in pairs and len(pairs) < 50:
                pairs.append(pair)
    rng.shuffle(pairs)
    schedule = []
    for pair_index, (target, blocker) in enumerate(pairs):
        for direction in ("LEFT", "RIGHT"):
            # Each pair and direction: 8 train, 1 validation, 1 test.
            splits = ["train"] * 8 + ["validation", "test"]
            bins = ["short"] * 3 + ["medium"] * 4 + ["long"] * 3
            rng.shuffle(splits)
            rng.shuffle(bins)
            for instance, (split, desired_bin) in enumerate(zip(splits, bins)):
                schedule.append({"pair_id": f"pair_{pair_index + 1:02d}", "target_asset": target,
                                 "blocker_asset": blocker, "direction": direction, "split": split,
                                 "desired_distance_bin": desired_bin, "seed": rng.randrange(2**31)})
    rng.shuffle(schedule)
    for index, row in enumerate(schedule, 1):
        row["scene_id"] = f"translate_diverse_{index:04d}"
    return schedule


def varied_calibration(position_cm, pitch_deg):
    result = calibration()
    pitch = math.radians(pitch_deg)
    right = np.array([1., 0., 0.])
    up = np.array([0., math.cos(pitch), math.sin(pitch)])
    forward = np.array([0., math.sin(pitch), -math.cos(pitch)])
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, -up, forward], axis=1)
    matrix[:3, 3] = np.asarray(position_cm) / 100 - [0, 0, SHELF_Z_CM / 100]
    result["T_camera_optical_to_shelf"] = matrix.tolist()
    result["camera_world_position_cm"] = list(position_cm)
    result["camera_pitch_deg"] = pitch_deg
    return result


def visible_enough(metrics):
    return (.40 <= metrics["visible_target_fraction"] <= .80
            and .20 <= metrics["blocker_occlusion_fraction"] <= .60
            and metrics["other_occlusion_fraction"] <= .10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/translate_diverse_1000_20261001_v1")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-scenes", type=int, default=1000, help="Debug batch limit; full predeclared schedule is always 1000")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based inclusive schedule index")
    parser.add_argument("--end-index", type=int, default=1000, help="Exclusive schedule index")
    parser.add_argument("--worker-tag", default="", help="Isolates worker progress/manifests when rendering disjoint ranges")
    parser.add_argument("--assemble-only", action="store_true", help="Rebuild final manifest from atomic sample records, no renderer")
    args = parser.parse_args()
    if not 1 <= args.max_scenes <= 1000:
        raise ValueError("max-scenes must be 1..1000")
    if not 0 <= args.start_index < args.end_index <= 1000:
        raise ValueError("Invalid schedule range")
    suffix = f"_{args.worker_tag}" if args.worker_tag else ""
    out = args.output.resolve()
    schedule = make_schedule(args.seed)
    if out.exists() and not args.resume:
        raise FileExistsError(f"Use a fresh output or explicit --resume: {out}")
    if args.resume:
        assert read(out / "schedule.json") == schedule, "Schedule changed; refuse incompatible resume"
    else:
        out.mkdir(parents=True)
        write_json(out / "schedule.json", schedule)
        write_json(out / "generation_config.json", {
            "schema": "shelfscene.translate_diverse.v1", "requested_scenes": 1000, "seed": args.seed,
            "role_pairs": 50, "per_pair": 20, "LEFT": 500, "RIGHT": 500,
            "splits": {"train": 800, "validation": 100, "test": 100},
            "split_policy": "independent randomized layouts; each role pair has 16 train / 2 validation / 2 test; no multiview or augmentation siblings",
            "role_assets": list(ASSETS), "distractor_count_range": [2, 7], "yaw_range_deg": [-180, 180],
            "target_visible_fraction": [.40, .80], "blocker_occlusion_fraction": [.20, .60],
            "max_other_occlusion_fraction": .10, "min_visible_pixels_each_object": 50,
            "palettes": list(PALETTES), "camera_xyz_cm_ranges": [[22, 34], [-100, -90], [62, 66]], "camera_pitch_deg_range": [76, 80],
            "distance_bins_m": DISTANCE_BINS, "desired_distance_bin_counts": {"short": 300, "medium": 400, "long": 300},
            "distance_bin_policy": "best-effort per scene; after 100 visual configurations may accept another feasible bin, explicitly recorded",
            "label_policy": "shortest valid 1mm-grid displacement over original 8 directions; accept only prescribed LEFT/RIGHT label",
            "limitations": "One shelf geometry with varied materials, lighting and viewpoint; 15 existing object meshes; no robot executor/dynamics/grasp validation; static conservative AABB corridor proxy",
        })
    template = ROOT / "experiments/translate_random_pilot_20261001_v2/samples/translate_random_001/scene.usda"
    records = []
    for item in schedule:
        record_path = out / "samples" / item["scene_id"] / "record.json"
        if record_path.exists():
            records.append(read(record_path))
    if args.assemble_only:
        save_manifest(out, records, args.seed)
        print(f"[bulk] ASSEMBLED {len(records)}/1000", flush=True)
        return
    seen = {record["scene_id"] for record in records}
    start = time.monotonic()
    starting_count = len(records)
    write_json(out / f"progress{suffix}.json", {"completed": len(records), "requested": 1000, "status": "starting"})
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT,
                         "renderer": "RaytracedLighting", "multi_gpu": False})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.utils.semantics import add_labels, upgrade_prim_semantics_to_labels
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
        import real_object_lib as rol
        assert set(ASSETS) == set(rol.ALL_NAMES)
        ctx = omni.usd.get_context()
        ctx.open_stage(str(template))
        stage = ctx.get_stage()
        stage.RemovePrim("/World/Objects")
        UsdGeom.Xform.Define(stage, "/World/Objects")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
        def measure(prim):
            cache.Clear()
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            return np.array(bounds.GetMin()), np.array(bounds.GetMax())
        prims, translations, rotations = {}, {}, {}
        for name in ASSETS:
            _, _, op = rol.reference_and_place(stage, "/World/Objects", name, 0, 0, 0, SHELF_Z_CM)
            prim = stage.GetPrimAtPath(f"/World/Objects/{name}")
            prims[name], translations[name] = prim, op
            rotations[name] = next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ)
            upgrade_prim_semantics_to_labels(prim)
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Gprim):
                    add_labels(child, [f"bulk_{name}"], "class", overwrite=True)
        cam = UsdGeom.Camera.Get(stage, CAMERA_PATH)
        cam_ops = UsdGeom.Xformable(cam.GetPrim()).GetOrderedXformOps()
        cam_t = next(op for op in cam_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        cam_r = next(op for op in cam_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateX)
        rp = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
        annotators = {}
        for key, name in (("rgb", "rgb"), ("depth", "distance_to_image_plane"), ("ids", "instance_segmentation_fast")):
            ann = rep.AnnotatorRegistry.get_annotator(name, init_params={"colorize": False} if key == "ids" else {})
            ann.attach([rp])
            annotators[key] = ann
        for _ in range(12):
            app.update()

        def capture(names, frames=3):
            for _ in range(frames):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                app.update()
            payload = annotators["ids"].get_data()
            raw = np.asarray(payload["data"])
            ids = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
            for index, name in enumerate(names, 1):
                root = f"/World/Objects/{name}"
                matches = [int(k) for k, v in payload["info"]["idToLabels"].items() if str(v) == root or str(v).startswith(root + "/")]
                ids[np.isin(raw, matches)] = index
            rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy()
            depth = np.asarray(annotators["depth"].get_data()).squeeze().astype(np.float32).copy()
            depth[~np.isfinite(depth) | (depth <= 0)] = 0
            return rgb, depth, ids

        for item in schedule[args.start_index:min(args.end_index, args.max_scenes)]:
            if item["scene_id"] in seen:
                continue
            rng = random.Random(item["seed"])
            target, blocker = item["target_asset"], item["blocker_asset"]
            accepted = None
            rejections = Counter()
            for attempt in range(300):
                distractors = rng.sample([n for n in ASSETS if n not in (target, blocker)], rng.randint(2, 7))
                names = [target, blocker, *distractors]
                for name, prim in prims.items():
                    imageable = UsdGeom.Imageable(prim)
                    imageable.MakeVisible() if name in names else imageable.MakeInvisible()
                dims, zeros, yaws = {}, {}, {}
                for name in names:
                    yaw = rng.uniform(-180, 180)
                    rotations[name].Set(yaw)
                    translations[name].Set(Gf.Vec3d(0, 0, SHELF_Z_CM))
                    lo, hi = measure(prims[name])
                    center = (lo + hi) / 2
                    shift = np.array([-center[0], -center[1], SHELF_Z_CM + SHELF_Z_CM - lo[2]])
                    translations[name].Set(Gf.Vec3d(*shift))
                    zeros[name], yaws[name], dims[name] = shift, yaw, (hi - lo) / 100
                # Filter geometry before expensive rendering. Reuse identities/yaws
                # for these draws, but resample ALL positions independently.
                layout = None
                for _ in range(60):
                    candidate = sample_layout(rng, dims, target, blocker, distractors, max_tries=2000)
                    best = candidate["candidates"][0]
                    if best["direction"] != item["direction"]:
                        continue
                    if attempt < 100 and distance_bin(best["distance_m"]) != item["desired_distance_bin"]:
                        continue
                    layout = candidate
                    break
                if layout is None:
                    rejections["geometry_bin_or_direction"] += 1
                    continue
                for name in names:
                    x, y = layout["centers"][name]
                    translations[name].Set(Gf.Vec3d(*(zeros[name] + [x * 100, y * 100, 0])))
                palette = rng.choice(list(PALETTES))
                for mat, color in zip(("Wood", "DarkWood"), PALETTES[palette]):
                    UsdShade.Shader.Get(stage, f"/World/Materials/{mat}/Shader").GetInput("diffuseColor").Set(Gf.Vec3f(*color))
                wall_color = rng.choice(((.86, .87, .86), (.72, .77, .82), (.82, .75, .66), (.65, .69, .65), (.90, .89, .82)))
                UsdShade.Shader.Get(stage, "/World/Materials/Wall/Shader").GetInput("diffuseColor").Set(Gf.Vec3f(*wall_color))
                lighting = {"Key": rng.uniform(5000, 11000), "Fill": rng.uniform(1200, 4500), "Dome": rng.uniform(600, 1400)}
                for name, intensity in lighting.items():
                    stage.GetPrimAtPath(f"/World/Lights/{name}").GetAttribute("inputs:intensity").Set(intensity)
                position = [rng.uniform(22, 34), rng.uniform(-100, -90), rng.uniform(62, 66)]
                pitch = rng.uniform(76, 80)
                cam_t.Set(Gf.Vec3d(*position))
                cam_r.Set(pitch)
                calib = varied_calibration(position, pitch)
                rgb, depth, ids = capture(names)
                if any(np.count_nonzero(ids == i) < 50 for i in range(1, len(names) + 1)):
                    rejections["object_hidden"] += 1
                    continue
                for name in names[1:]:
                    UsdGeom.Imageable(prims[name]).MakeInvisible()
                _, _, reference_ids = capture(names)
                reference = reference_ids == 1
                for name in names[1:]:
                    UsdGeom.Imageable(prims[name]).MakeVisible()
                metrics = visibility_metrics(ids, reference)
                if not visible_enough(metrics):
                    rejections["occlusion_fraction"] += 1
                    continue
                objects = []
                for index, name in enumerate(names, 1):
                    lo, hi = measure(prims[name])
                    lo, hi = lo / 100 - [0, 0, .42], hi / 100 - [0, 0, .42]
                    objects.append({"instance_id": index, "asset": name, "role": "target" if index == 1 else "blocker" if index == 2 else "distractor",
                                    "yaw_deg": yaws[name], "usd_root": f"/World/Objects/{name}", "asset_source": rol.OBJECTS[name][0],
                                    "aabb_shelf_m": {"min": lo.tolist(), "max": hi.tolist()}, "reference_point_shelf_m": ((lo + hi) / 2).tolist(),
                                    "footprint_shelf_m": [lo[0], lo[1], hi[0], hi[1]]})
                points = backproject(depth, calib)
                audits = []
                for obj in objects:
                    mask = ids == obj["instance_id"]
                    cloud = points[mask & (depth > 0)]
                    box = obj["aabb_shelf_m"]
                    fraction = float(np.mean(np.all((cloud >= np.asarray(box["min"]) - .008) & (cloud <= np.asarray(box["max"]) + .008), axis=1))) if len(cloud) else 0.
                    audits.append({"instance_id": obj["instance_id"], "visible_pixels": int(mask.sum()), "valid_depth_fraction": float(len(cloud) / mask.sum()), "backprojected_inside_aabb_fraction_8mm_tolerance": fraction})
                if min(a["valid_depth_fraction"] for a in audits) < .98 or min(a["backprojected_inside_aabb_fraction_8mm_tolerance"] for a in audits) < .95:
                    rejections["calibration_audit"] += 1
                    continue
                accepted = True
                break
            if not accepted:
                write_json(out / f"failure{suffix}.json", {"scene": item, "rejections": dict(rejections)})
                raise RuntimeError(f"Failed scene {item['scene_id']}: {rejections}")
            scene_id = item["scene_id"]
            folder = out / "samples" / scene_id
            temp = out / "pending" / scene_id
            if temp.exists():
                temp.rename(out / f"interrupted_{scene_id}_{time.time_ns()}")
            temp.mkdir(parents=True)
            label = {**layout["candidates"][0], "object_id": 2, "action": "TRANSLATE", "reference_frame": "ShelfFrame",
                     "current_reference_point_shelf_m": objects[1]["reference_point_shelf_m"]}
            label["goal_reference_point_shelf_m"] = (np.asarray(label["current_reference_point_shelf_m"]) + label["delta_shelf_m"]).tolist()
            Image.fromarray(rgb).save(temp / "rgb.png", compress_level=3)
            np.save(temp / "depth_z_m.npy", depth)
            Image.fromarray(ids).save(temp / "instance_ids.png")
            for name, mask in (("target_mask", ids == 1), ("blocker_mask", ids == 2), ("depth_valid", depth > 0), ("target_reference_mask", reference)):
                Image.fromarray((mask * 255).astype(np.uint8)).save(temp / f"{name}.png")
            crop = (depth > 0) & (points[..., 0] >= BOUNDS[0] - .02) & (points[..., 0] <= BOUNDS[2] + .02) & (points[..., 1] >= BOUNDS[1] - .02) & (points[..., 1] <= BOUNDS[3] + .02) & (points[..., 2] >= -.01) & (points[..., 2] <= .37)
            vv, uu = np.where(crop)
            np.savez_compressed(temp / "pointcloud.npz", xyz_shelf_m=points[crop].astype(np.float32), rgb=rgb[crop], instance_id=ids[crop], pixel_uv=np.stack([uu, vv], axis=1))
            environment = {"shelf_palette": palette, "wall_color": wall_color, "lighting_intensities": lighting, "camera_position_cm": position, "camera_pitch_deg": pitch}
            for filename, value in (("calibration", calib), ("label", label), ("objects", objects), ("environment", environment),
                                    ("geometry_audit", {"objects": audits, "all_direction_candidates": layout["candidates"], "corridor_footprint_m": layout["corridor_footprint_m"]}),
                                    ("visibility_audit", metrics), ("sampling_audit", {"attempts": attempt + 1, "rejections": dict(rejections), "planned": item,
                                                                                     "actual_distance_bin": distance_bin(label["distance_m"]), "bin_fallback": distance_bin(label["distance_m"]) != item["desired_distance_bin"]})):
                write_json(temp / f"{filename}.json", value)
            stage.GetRootLayer().Export(str(temp / "scene.usda"))
            annotated_preview(rgb, ids, objects, label, calib).save(temp / "preview.png", compress_level=3)
            topdown_preview(objects, label, layout["corridor_footprint_m"]).save(temp / "topdown.png")
            original = np.asarray(translations[blocker].Get()).copy()
            translations[blocker].Set(Gf.Vec3d(*(original + np.asarray(label["delta_shelf_m"]) * 100)))
            goal_rgb, _, _ = capture(names)
            Image.fromarray(goal_rgb).save(temp / "goal_rgb.png", compress_level=3)
            translations[blocker].Set(Gf.Vec3d(*original))
            record = {**item, "sample_dir": str(folder.relative_to(out)), "target_id": 1, "blocker_id": 2,
                      "distractor_assets": distractors, "label": label, "distance_bin": distance_bin(label["distance_m"]),
                      "environment": environment, "visibility": metrics, "visual_attempts": attempt + 1,
                      "rgb_sha256": hashlib.sha256((temp / "rgb.png").read_bytes()).hexdigest(),
                      "layout_sha256": hashlib.sha256(str([(o["asset"], o["yaw_deg"], o["reference_point_shelf_m"]) for o in objects]).encode()).hexdigest()}
            write_json(temp / "record.json", record)
            folder.parent.mkdir(exist_ok=True)
            temp.rename(folder)
            records.append(record)
            elapsed = time.monotonic() - start
            rate = (len(records) - starting_count) / elapsed
            write_json(out / f"progress{suffix}.json", {"completed": len(records), "requested": 1000, "last_scene_id": scene_id, "status": "generating",
                                                "elapsed_this_run_s": elapsed, "scenes_per_minute": rate * 60,
                                                "estimated_remaining_minutes": (1000 - len(records)) / max(rate, 1e-9) / 60})
            print(f"[bulk] {len(records)}/1000 {scene_id} {target}/{blocker} {label['direction']} {label['distance_m']*100:.1f}cm attempts={attempt+1} rate={rate*60:.1f}/min", flush=True)
            if len(records) % 25 == 0:
                write_json(out / f"records_checkpoint{suffix}.json", records)
        ordered = sorted(records, key=lambda r: r["scene_id"])
        manifest = {"schema": "shelfscene.translate_diverse.v1", "sample_count": len(ordered), "requested_sample_count": 1000,
                    "seed": args.seed, "units": "meters", "action": "TRANSLATE", "translation_space": "shelf XY plane",
                    "status": "generation complete; independent validation pending" if len(ordered) == 1000 else "partial generation",
                    "label_source": "static conservative world-AABB target-front-corridor proxy",
                    "bulk_generation": read(out / "generation_config.json"), "samples": ordered}
        write_json(out / f"manifest{suffix}.json", manifest)
        write_json(out / f"splits{suffix}.json", {split: [r["scene_id"] for r in ordered if r["split"] == split] for split in ("train", "validation", "test")})
        source_dir = out / f"source{suffix}"
        source_dir.mkdir(exist_ok=True)
        for name in ("generate_translate_1000.py", "generate_translate_rgbd_pilot.py", "translate_pilot_geometry.py", "real_object_lib.py", "revise_translate_pilot_visibility.py"):
            shutil.copy2(Path(__file__).parent / name, source_dir / name)
        write_json(out / f"progress{suffix}.json", {"completed": len(ordered), "requested": 1000, "status": "generated" if len(ordered) == 1000 else "partial"})
        print(f"[bulk] COMPLETE CURRENT BATCH: {len(ordered)} scenes in {out}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


def save_manifest(out, records, seed):
    ordered = sorted(records, key=lambda r: r["scene_id"])
    write_json(out / "manifest.json", {"schema": "shelfscene.translate_diverse.v1", "sample_count": len(ordered), "requested_sample_count": 1000,
                                       "seed": seed, "units": "meters", "action": "TRANSLATE", "translation_space": "shelf XY plane",
                                       "status": "generated; independent validation pending", "label_source": "static conservative world-AABB target-front-corridor proxy",
                                       "bulk_generation": read(out / "generation_config.json"), "samples": ordered})
    write_json(out / "splits.json", {split: [r["scene_id"] for r in ordered if r["split"] == split] for split in ("train", "validation", "test")})
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    for name in ("generate_translate_1000.py", "audit_translate_1000.py", "validate_translate_rgbd_pilot.py", "generate_translate_rgbd_pilot.py", "translate_pilot_geometry.py", "real_object_lib.py", "revise_translate_pilot_visibility.py"):
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    write_json(out / "progress.json", {"completed": len(ordered), "requested": 1000, "status": "generated" if len(ordered) == 1000 else "partial"})


if __name__ == "__main__":
    main()
