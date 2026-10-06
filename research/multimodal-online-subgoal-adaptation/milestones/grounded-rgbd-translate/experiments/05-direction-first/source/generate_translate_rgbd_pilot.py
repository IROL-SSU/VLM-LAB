#!/usr/bin/env python3
"""Generate a randomized, grounded TRANSLATE pilot with aligned RGB-D.

Run using Isaac Sim's python.sh. Outputs are new files under --output only.
Five scenes use ten distinct target/blocker assets and random distractors.
Labels are a declared static-AABB corridor-clearing proxy, not robot trials.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from translate_pilot_geometry import (
    BOUNDS, CORRIDOR_MARGIN, DIRECTIONS, OBJECT_GAP, STEP, Rect, sample_layout,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 720
SHELF_Z_CM = 42.0
CAM_POSITION_CM = np.array([28.0, -95.0, 64.0])
CAM_PITCH_DEG = 78.0
CAMERA_PATH = "/World/Cameras/TranslateCam"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def calibration():
    pitch = math.radians(CAM_PITCH_DEG)
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, math.cos(pitch), math.sin(pitch)])
    forward = np.array([0.0, math.sin(pitch), -math.cos(pitch)])
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, -up, forward], axis=1)
    matrix[:3, 3] = CAM_POSITION_CM / 100 - np.array([0, 0, SHELF_Z_CM / 100])
    return {
        "image_width": WIDTH, "image_height": HEIGHT,
        "K": [[32 / 36 * WIDTH, 0, WIDTH / 2], [0, 32 / 27 * HEIGHT, HEIGHT / 2], [0, 0, 1]],
        "T_camera_optical_to_shelf": matrix.tolist(),
        "camera_optical_axes": "+X right, +Y down, +Z forward",
        "shelf_axes": "+X right, +Y back, +Z up; origin is world (0,0,42cm)",
        "shelf_bounds_xy_m": list(BOUNDS),
        "stage_meters_per_unit": 0.01,
        "depth_definition": "optical-axis Z in meters; distance_to_image_plane annotator",
        "invalid_depth_value": 0,
        "direction_vectors": {name: [*vector, 0.0] for name, vector in DIRECTIONS.items()},
        "pixel_convention": "backprojection at pixel centers (u+0.5, v+0.5)",
    }


def backproject(depth, calib):
    v, u = np.indices(depth.shape, dtype=np.float32)
    k = np.asarray(calib["K"])
    optical = np.stack([(u + 0.5 - k[0, 2]) * depth / k[0, 0],
                        (v + 0.5 - k[1, 2]) * depth / k[1, 1], depth], axis=-1)
    transform = np.asarray(calib["T_camera_optical_to_shelf"], dtype=np.float32)
    return optical @ transform[:3, :3].T + transform[:3, 3]


def project(point, calib):
    transform = np.asarray(calib["T_camera_optical_to_shelf"])
    xyz = transform[:3, :3].T @ (np.asarray(point) - transform[:3, 3])
    k = np.asarray(calib["K"])
    return (float(k[0, 0] * xyz[0] / xyz[2] + k[0, 2]), float(k[1, 1] * xyz[1] / xyz[2] + k[1, 2]))


def annotated_preview(rgb, id_map, objects, label, calib):
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in objects:
        rows, cols = np.where(id_map == obj["instance_id"])
        if not rows.size:
            continue
        color = {"target": "#22ef80", "blocker": "#ffae33"}.get(obj["role"], "#dddddd")
        box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
        draw.rectangle(box, outline=color, width=3)
        text = f'{obj["instance_id"]}: {obj["asset"]} ({obj["role"]})'
        x, y = box[0], max(2, box[1] - 23)
        draw.rectangle([x, y, min(WIDTH - 1, x + len(text) * 9), y + 22], fill="#202020")
        draw.text((x + 2, y + 2), text, font=font(15), fill=color)
    start = project(label["current_reference_point_shelf_m"], calib)
    end = project(label["goal_reference_point_shelf_m"], calib)
    draw.line([start, end], fill="#ff3232", width=6)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [end, (end[0] - 16 * math.cos(angle - 0.5), end[1] - 16 * math.sin(angle - 0.5)),
              (end[0] - 16 * math.cos(angle + 0.5), end[1] - 16 * math.sin(angle + 0.5))]
    draw.polygon(points, fill="#ff3232")
    draw.rectangle((0, HEIGHT - 35, WIDTH, HEIGHT), fill="#181818")
    draw.text((12, HEIGHT - 29), f'TRANSLATE {label["direction"]} {label["distance_m"] * 100:.1f} cm | geometry-proxy label', font=font(20), fill="white")
    return image


def topdown_preview(objects, label, corridor):
    image = Image.new("RGB", (960, 460), "#f1ede6")
    draw = ImageDraw.Draw(image)
    def xy(x, y):
        return (35 + (x - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]) * 890,
                365 - (y - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1]) * 310)
    def box(rect):
        return [*xy(rect[0], rect[3]), *xy(rect[2], rect[1])]
    draw.rectangle(box(BOUNDS), fill="#e2c59b", outline="black", width=3)
    draw.rectangle(box(corridor), fill="#c9f4d8", outline="#159350", width=2)
    for obj in objects:
        color = {"target": "#21aa66", "blocker": "#f1a630"}.get(obj["role"], "#899bad")
        draw.rectangle(box(obj["footprint_shelf_m"]), fill=color, outline="#333333", width=2)
        center = obj["reference_point_shelf_m"]
        draw.text(xy(center[0], center[1]), str(obj["instance_id"]), font=font(18), fill="black")
    draw.rectangle(box(label["goal_footprint_m"]), outline="#df3333", width=4)
    p0, p1 = label["current_reference_point_shelf_m"], label["goal_reference_point_shelf_m"]
    draw.line([xy(*p0[:2]), xy(*p1[:2])], fill="#df3333", width=4)
    draw.text((35, 8), "TOP-DOWN | green: target corridor | red outline: blocker goal", font=font(19), fill="black")
    draw.text((35, 389), "Shelf front (-Y) at bottom; +X right. Static AABB footprints, not contact dynamics.", font=font(17), fill="black")
    draw.text((35, 423), f'{label["direction"]}: {label["distance_m"] * 100:.1f} cm', font=font(20), fill="#aa2222")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/translate_random_pilot_20261001")
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "manifest.json").exists() or (out / "samples").exists():
        raise FileExistsError(f"Refusing to overwrite an existing/partial dataset: {out}")
    out.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT,
                         "renderer": "RaytracedLighting", "multi_gpu": False})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.utils.semantics import add_labels, upgrade_prim_semantics_to_labels
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
        import real_object_lib as rol

        if not 1 <= args.num_scenes <= len(rol.ALL_NAMES) // 2:
            raise ValueError("Unique target/blocker assets require 1 <= num-scenes <= 7")
        rng = random.Random(args.seed)
        role_assets = rng.sample(rol.ALL_NAMES, 2 * args.num_scenes)
        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        def material(name, rgb):
            mat = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/Materials/{name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        wood = material("Wood", (0.80, 0.65, 0.45))
        dark = material("DarkWood", (0.65, 0.49, 0.31))
        wall = material("Wall", (0.86, 0.87, 0.86))
        floor = material("Floor", (0.5, 0.5, 0.48))
        def cube(path, center, size, mat):
            obj = UsdGeom.Cube.Define(stage, path)
            obj.AddTranslateOp().Set(Gf.Vec3d(*center))
            obj.AddScaleOp().Set(Gf.Vec3d(*(value / 2 for value in size)))
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
        for name, x in (("Left", -76), ("Middle", -20), ("Right", 76)):
            cube(f"/World/Shelf/Post{name}", (x, 0, 60), (4, 32, 120), dark)
        for index, z in enumerate((1.5, 40.5, 79.5, 118.5)):
            cube(f"/World/Shelf/Board{index}", (0, 0, z), (156, 32, 3), wood)
        cube("/World/Environment/BackWall", (0, 25, 85), (240, 4, 180), wall)
        cube("/World/Environment/Floor", (0, 0, -2), (600, 600, 4), floor)
        for name, orientation, intensity in (("Key", (-55, 25, 0), 8000), ("Fill", (-30, -100, 0), 2400)):
            light = UsdLux.DistantLight.Define(stage, f"/World/Lights/{name}")
            UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*orientation))
            light.CreateIntensityAttr(intensity)
        UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(900)
        cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        cam.AddTranslateOp().Set(Gf.Vec3d(*CAM_POSITION_CM))
        cam.AddRotateXOp().Set(CAM_PITCH_DEG)
        cam.CreateFocalLengthAttr(32)
        cam.CreateHorizontalApertureAttr(36)
        cam.CreateVerticalApertureAttr(27)
        cam.CreateClippingRangeAttr(Gf.Vec2f(1, 100000))
        rp = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
        annotators = {}
        for key, name in (("rgb", "rgb"), ("depth", "distance_to_image_plane"), ("ids", "instance_segmentation_fast")):
            annotator = rep.AnnotatorRegistry.get_annotator(name, init_params={"colorize": False} if key == "ids" else {})
            annotator.attach([rp])
            annotators[key] = annotator
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
        calib = calibration()
        write_json(out / "calibration.json", calib)
        records = []

        def measure(prim):
            bbox_cache.Clear()
            bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            return np.array(bounds.GetMin(), dtype=float), np.array(bounds.GetMax(), dtype=float)

        def capture(names):
            latest = None
            for _ in range(12):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                app.update()
                payload = annotators["ids"].get_data()
                if not isinstance(payload, dict) or not np.asarray(payload.get("data")).size:
                    continue
                raw_ids = np.asarray(payload["data"])
                id_map = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                mapping = payload.get("info", {}).get("idToLabels", {})
                for index, name in enumerate(names, 1):
                    root = f"/World/Objects/{name}"
                    matches = [int(key) for key, value in mapping.items()
                               if str(value) == root or str(value).startswith(root + "/")]
                    id_map[np.isin(raw_ids, matches)] = index
                rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy()
                depth = np.asarray(annotators["depth"].get_data()).squeeze().astype(np.float32).copy()
                depth[~np.isfinite(depth) | (depth <= 0)] = 0
                latest = rgb, depth, id_map
                if rgb.shape == (HEIGHT, WIDTH, 3) and all(np.count_nonzero(id_map == i) >= 50 for i in range(1, len(names) + 1)):
                    # One extra synchronized frame prevents using the first material-loading frame.
                    if _ >= 3:
                        return latest
            return latest

        for scene_index in range(args.num_scenes):
            target_asset, blocker_asset = role_assets[2 * scene_index:2 * scene_index + 2]
            scene_id = f"translate_random_{scene_index + 1:03d}"
            accepted = None
            for visual_attempt in range(25):
                if stage.GetPrimAtPath("/World/Objects").IsValid():
                    stage.RemovePrim("/World/Objects")
                UsdGeom.Xform.Define(stage, "/World/Objects")
                distractors = rng.sample([name for name in rol.ALL_NAMES if name not in (target_asset, blocker_asset)], rng.randint(3, 5))
                names = [target_asset, blocker_asset, *distractors]
                ops, dims, object_specs = {}, {}, []
                for index, name in enumerate(names, 1):
                    yaw = rng.uniform(-180, 180)
                    _, _, op = rol.reference_and_place(stage, "/World/Objects", name, 0, 0, yaw, SHELF_Z_CM)
                    prim = stage.GetPrimAtPath(f"/World/Objects/{name}")
                    lo, hi = measure(prim)
                    center = (lo + hi) / 2
                    old = np.asarray(op.Get())
                    # Center each rotated asset by its actual world AABB and floor-align it.
                    op.Set(Gf.Vec3d(float(old[0] - center[0]), float(old[1] - center[1]), float(old[2] + SHELF_Z_CM - lo[2])))
                    lo, hi = measure(prim)
                    dims[name] = tuple((hi - lo) / 100)
                    ops[name] = op
                    upgrade_prim_semantics_to_labels(prim)
                    for descendant in Usd.PrimRange(prim):
                        if descendant.IsA(UsdGeom.Gprim):
                            add_labels(descendant, [f"pilot_{index}"], "class", overwrite=True)
                    object_specs.append({"instance_id": index, "asset": name, "role": "target" if index == 1 else "blocker" if index == 2 else "distractor", "yaw_deg": yaw})
                layout = sample_layout(rng, dims, target_asset, blocker_asset, distractors)
                for spec in object_specs:
                    name = spec["asset"]
                    x, y = layout["centers"][name]
                    old = np.asarray(ops[name].Get())
                    ops[name].Set(Gf.Vec3d(float(old[0] + x * 100), float(old[1] + y * 100), float(old[2])))
                    lo, hi = measure(stage.GetPrimAtPath(f"/World/Objects/{name}"))
                    lo_m, hi_m = lo / 100 - np.array([0, 0, 0.42]), hi / 100 - np.array([0, 0, 0.42])
                    spec.update({"usd_root": f"/World/Objects/{name}", "asset_source": rol.OBJECTS[name][0],
                                 "aabb_shelf_m": {"min": lo_m.tolist(), "max": hi_m.tolist()},
                                 "reference_point_shelf_m": ((lo_m + hi_m) / 2).tolist(),
                                 "footprint_shelf_m": [lo_m[0], lo_m[1], hi_m[0], hi_m[1]]})
                for _ in range(8):
                    app.update()
                captured = capture(names)
                if captured is None:
                    raise RuntimeError("No RGB-D/instance frame produced")
                rgb, depth, id_map = captured
                counts = [int(np.count_nonzero(id_map == i)) for i in range(1, len(names) + 1)]
                if min(counts) < 50:
                    print(f"[pilot] {scene_id} resampling partially/fully hidden objects: {counts}", flush=True)
                    continue
                points = backproject(depth, calib)
                audits = []
                for spec in object_specs:
                    mask = id_map == spec["instance_id"]
                    valid = mask & (depth > 0)
                    cloud = points[valid]
                    lo = np.array(spec["aabb_shelf_m"]["min"])
                    hi = np.array(spec["aabb_shelf_m"]["max"])
                    ratio = float(np.mean(np.all((cloud >= lo - 0.008) & (cloud <= hi + 0.008), axis=1))) if len(cloud) else 0.0
                    audits.append({"instance_id": spec["instance_id"], "visible_pixels": int(mask.sum()),
                                   "valid_depth_fraction": float(valid.sum() / mask.sum()),
                                   "backprojected_inside_aabb_fraction_8mm_tolerance": ratio})
                if min(item["valid_depth_fraction"] for item in audits) < 0.98 or min(item["backprojected_inside_aabb_fraction_8mm_tolerance"] for item in audits) < 0.95:
                    raise RuntimeError(f"RGB-D calibration/segmentation mismatch: {audits}")
                accepted = rgb, depth, id_map, points, object_specs, layout, audits
                break
            if accepted is None:
                raise RuntimeError(f"Could not make all objects visible: {scene_id}")
            rgb, depth, id_map, points, objects, layout, audits = accepted
            sample_dir = out / "samples" / scene_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            label = {**layout["candidates"][0], "object_id": 2, "action": "TRANSLATE", "reference_frame": "ShelfFrame",
                     "current_reference_point_shelf_m": objects[1]["reference_point_shelf_m"]}
            label["goal_reference_point_shelf_m"] = (np.array(label["current_reference_point_shelf_m"]) + np.array(label["delta_shelf_m"])).tolist()
            Image.fromarray(rgb).save(sample_dir / "rgb.png")
            np.save(sample_dir / "depth_z_m.npy", depth)
            Image.fromarray(id_map).save(sample_dir / "instance_ids.png")
            for name, index in (("target", 1), ("blocker", 2)):
                Image.fromarray(((id_map == index) * 255).astype(np.uint8)).save(sample_dir / f"{name}_mask.png")
            Image.fromarray(((depth > 0) * 255).astype(np.uint8)).save(sample_dir / "depth_valid.png")
            crop = (depth > 0) & (points[..., 0] >= BOUNDS[0] - 0.02) & (points[..., 0] <= BOUNDS[2] + 0.02) & (points[..., 1] >= BOUNDS[1] - 0.02) & (points[..., 1] <= BOUNDS[3] + 0.02) & (points[..., 2] >= -0.01) & (points[..., 2] <= 0.37)
            rows, cols = np.where(crop)
            np.savez_compressed(sample_dir / "pointcloud.npz", xyz_shelf_m=points[crop].astype(np.float32), rgb=rgb[crop], instance_id=id_map[crop], pixel_uv=np.stack([cols, rows], axis=1))
            write_json(sample_dir / "calibration.json", calib)
            write_json(sample_dir / "label.json", label)
            write_json(sample_dir / "objects.json", objects)
            write_json(sample_dir / "geometry_audit.json", {"objects": audits, "all_direction_candidates": layout["candidates"], "corridor_footprint_m": layout["corridor_footprint_m"], "layout_attempts": layout["attempts"], "visual_attempts": visual_attempt + 1})
            stage.GetRootLayer().Export(str(sample_dir / "scene.usda"))
            annotated_preview(rgb, id_map, objects, label, calib).save(sample_dir / "preview.png")
            topdown_preview(objects, label, layout["corridor_footprint_m"]).save(sample_dir / "topdown.png")
            blocker_op = ops[blocker_asset]
            original = np.asarray(blocker_op.Get()).copy()
            delta = np.asarray(label["delta_shelf_m"]) * 100
            blocker_op.Set(Gf.Vec3d(*(original + delta)))
            goal_capture = capture(names)
            if goal_capture is not None:
                Image.fromarray(goal_capture[0]).save(sample_dir / "goal_rgb.png")
            blocker_op.Set(Gf.Vec3d(*original))
            records.append({"scene_id": scene_id, "sample_dir": str(sample_dir.relative_to(out)), "target_asset": target_asset,
                            "blocker_asset": blocker_asset, "target_id": 1, "blocker_id": 2,
                            "distractor_assets": distractors, "label": label})
            write_json(out / "progress.json", records)
            print(f"[pilot] saved {scene_id}: target={target_asset}, blocker={blocker_asset}, distractors={distractors}, {label['direction']} {label['distance_m']:.3f}m", flush=True)

        manifest = {
            "schema": "shelfscene.translate_rgbd_pilot.v1", "seed": args.seed, "sample_count": len(records),
            "status": "generated; run independent validator before training",
            "grounding_source": "simulator instance IDs / masks, not a VLM prediction",
            "units": "meters", "action": "TRANSLATE", "translation_space": "shelf XY plane",
            "label_source": "static conservative world-AABB target-front-corridor proxy",
            "label_policy": {
                "corridor": "target AABB x-range expanded by margin; from shelf front to target front edge",
                "corridor_side_margin_m": CORRIDOR_MARGIN, "object_gap_m": OBJECT_GAP,
                "direction_count": len(DIRECTIONS), "distance_search_resolution_m": STEP,
                "selection": "shortest valid 1mm-grid distance among 8 directions; alphabetical direction tie-break",
                "validity": "goal clears entire corridor; continuous swept rectangle avoids other AABBs and shelf bounds",
                "limitations": "Proxy geometry labels; no executor, dynamics, or real grasp success evaluated. Visible input does not contain hidden simulator geometry.",
            },
            "randomization": {"unique_target_and_blocker_assets_across_scenes": True,
                              "distractor_count_range": [3, 5], "yaw_deg_range": [-180, 180],
                              "positions": "rejection-sampled inside shelf; non-overlap; one designated corridor blocker",
                              "visibility": "all selected objects have at least 50 visible pixels"},
            "samples": records,
        }
        write_json(out / "manifest.json", manifest)
        sheet = Image.new("RGB", (960, 400 * len(records)), "white")
        for index, row in enumerate(records):
            folder = out / row["sample_dir"]
            with Image.open(folder / "preview.png") as image:
                sheet.paste(image.resize((480, 360)), (0, index * 400 + 30))
            with Image.open(folder / "topdown.png") as image:
                sheet.paste(image.resize((480, 230)), (480, index * 400 + 95))
            ImageDraw.Draw(sheet).text((12, index * 400 + 5), f'{row["scene_id"]} | target: {row["target_asset"]} | blocker: {row["blocker_asset"]}', font=font(18), fill="black")
        sheet.save(out / "contact_sheet.png")
        print(f"[pilot] COMPLETE: {out}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
