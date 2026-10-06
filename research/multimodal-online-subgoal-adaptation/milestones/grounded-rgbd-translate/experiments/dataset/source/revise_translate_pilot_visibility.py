#!/usr/bin/env python3
"""Revise scenes 2/4 by moving only their blocker; keep original dataset intact.

Render a target-only reference (shelf retained) to measure actual silhouette
occlusion, not overlap between the mutually exclusive visible instance masks.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_translate_rgbd_pilot import (
    ROOT, WIDTH, HEIGHT, CAMERA_PATH, annotated_preview, backproject, font,
    topdown_preview, write_json,
)
from translate_pilot_geometry import BOUNDS, OBJECT_GAP, Rect, target_corridor, translation_candidates


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def visibility_metrics(ids, target_reference):
    count = int(target_reference.sum())
    if count == 0:
        raise ValueError("Empty target-only silhouette")
    return {
        "reference_target_pixels": count,
        "visible_target_fraction": float(((ids == 1) & target_reference).sum() / count),
        "blocker_occlusion_fraction": float(((ids == 2) & target_reference).sum() / count),
        "other_occlusion_fraction": float(((ids != 1) & (ids != 2) & target_reference).sum() / count),
    }


def acceptable(metrics):
    return (0.50 <= metrics["visible_target_fraction"] <= 0.75
            and 0.25 <= metrics["blocker_occlusion_fraction"] <= 0.50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/translate_random_pilot_20261001")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/translate_random_pilot_20261001_v2")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    manifest = read(source / "manifest.json")
    # Only new copies are modified; original samples and all their labels survive.
    out.mkdir(parents=True)
    shutil.copy2(source / "calibration.json", out / "calibration.json")
    revisions = []
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT,
                         "renderer": "RaytracedLighting", "multi_gpu": False})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Usd, UsdGeom
        for number, row in enumerate(manifest["samples"], 1):
            src, dst = source / row["sample_dir"], out / row["sample_dir"]
            if number not in (2, 4):
                shutil.copytree(src, dst)
                continue
            dst.mkdir(parents=True)
            objects = read(src / "objects.json")
            calib = read(src / "calibration.json")
            ctx = omni.usd.get_context()
            ctx.open_stage(str(src / "scene.usda"))
            stage = ctx.get_stage()
            for _ in range(12):
                app.update()
            rp = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
            annotators = {}
            for key, name in (("rgb", "rgb"), ("depth", "distance_to_image_plane"), ("ids", "instance_segmentation_fast")):
                ann = rep.AnnotatorRegistry.get_annotator(name, init_params={"colorize": False} if key == "ids" else {})
                ann.attach([rp])
                annotators[key] = ann

            def capture():
                for _ in range(5):
                    rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                    app.update()
                payload = annotators["ids"].get_data()
                raw = np.asarray(payload["data"])
                ids = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                for obj in objects:
                    root = obj["usd_root"]
                    matches = [int(key) for key, value in payload["info"]["idToLabels"].items()
                               if str(value) == root or str(value).startswith(root + "/")]
                    ids[np.isin(raw, matches)] = obj["instance_id"]
                rgb = np.asarray(annotators["rgb"].get_data())[..., :3].copy()
                depth = np.asarray(annotators["depth"].get_data()).squeeze().astype(np.float32).copy()
                depth[~np.isfinite(depth) | (depth <= 0)] = 0
                return rgb, depth, ids

            before = capture()
            for obj in objects[1:]:
                UsdGeom.Imageable(stage.GetPrimAtPath(obj["usd_root"])).MakeInvisible()
            reference_rgb, _, reference_ids = capture()
            reference = reference_ids == 1
            for obj in objects[1:]:
                UsdGeom.Imageable(stage.GetPrimAtPath(obj["usd_root"])).MakeVisible()
            before_metrics = visibility_metrics(before[2], reference)
            blocker = objects[1]
            prim = stage.GetPrimAtPath(blocker["usd_root"])
            op = next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            original_pos = np.asarray(op.Get()).copy()
            original_objects = copy.deepcopy(objects)
            base_rects = [Rect.from_list(obj["footprint_shelf_m"]) for obj in objects]
            trials = []
            accepted = None
            # Search nearest changes first. 2 mm is positional search resolution;
            # sub-goal distance still uses the original 1 mm label grid.
            for offset in sorted((i * 0.002 for i in range(-60, 61) if i), key=lambda x: (abs(x), x)):
                moving = base_rects[1].moved(offset, 0)
                obstacles = [base_rects[0], *base_rects[2:]]
                if not moving.inside() or any(moving.overlap(rect.expanded(OBJECT_GAP)) > 1e-10 for rect in obstacles):
                    continue
                candidates = translation_candidates(base_rects[0], moving, base_rects[2:])
                if not candidates or not 0.02 <= candidates[0]["distance_m"] <= 0.25:
                    continue
                op.Set(Gf.Vec3d(*(original_pos + [offset * 100, 0, 0])))
                rgb, depth, ids = capture()
                metrics = visibility_metrics(ids, reference)
                trials.append({"blocker_offset_x_m": offset, **metrics})
                print(f"[revision] scene {number}: dx={offset:.3f} visible={metrics['visible_target_fraction']:.3f} occluded={metrics['blocker_occlusion_fraction']:.3f}", flush=True)
                if acceptable(metrics) and all(np.count_nonzero(ids == obj["instance_id"]) >= 50 for obj in objects):
                    accepted = offset, rgb, depth, ids, metrics, candidates
                    break
            if accepted is None:
                raise RuntimeError(f"Could not satisfy visibility policy for scene {number}")
            offset, rgb, depth, ids, metrics, candidates = accepted
            # Re-measure rendered geometry instead of assuming the USD transform.
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo = np.asarray(bounds.GetMin()) / 100 - [0, 0, 0.42]
            hi = np.asarray(bounds.GetMax()) / 100 - [0, 0, 0.42]
            blocker.update({"aabb_shelf_m": {"min": lo.tolist(), "max": hi.tolist()},
                            "reference_point_shelf_m": ((lo + hi) / 2).tolist(),
                            "footprint_shelf_m": [lo[0], lo[1], hi[0], hi[1]]})
            assert np.allclose(blocker["footprint_shelf_m"], base_rects[1].moved(offset, 0).as_list())
            label = {**candidates[0], "object_id": 2, "action": "TRANSLATE", "reference_frame": "ShelfFrame",
                     "current_reference_point_shelf_m": blocker["reference_point_shelf_m"]}
            label["goal_reference_point_shelf_m"] = (np.asarray(label["current_reference_point_shelf_m"]) + label["delta_shelf_m"]).tolist()
            points = backproject(depth, calib)
            audits = []
            for obj in objects:
                mask = ids == obj["instance_id"]
                cloud = points[mask & (depth > 0)]
                box = obj["aabb_shelf_m"]
                audits.append({"instance_id": obj["instance_id"], "visible_pixels": int(mask.sum()),
                               "valid_depth_fraction": float(len(cloud) / mask.sum()),
                               "backprojected_inside_aabb_fraction_8mm_tolerance": float(np.mean(np.all((cloud >= np.asarray(box["min"]) - .008) & (cloud <= np.asarray(box["max"]) + .008), axis=1)))})
            Image.fromarray(rgb).save(dst / "rgb.png")
            np.save(dst / "depth_z_m.npy", depth)
            Image.fromarray(ids).save(dst / "instance_ids.png")
            for name, mask in (("target_mask", ids == 1), ("blocker_mask", ids == 2), ("depth_valid", depth > 0), ("target_reference_mask", reference)):
                Image.fromarray((mask * 255).astype(np.uint8)).save(dst / f"{name}.png")
            Image.fromarray(reference_rgb).save(dst / "target_reference_rgb.png")
            crop = (depth > 0) & (points[..., 0] >= BOUNDS[0] - .02) & (points[..., 0] <= BOUNDS[2] + .02) & (points[..., 1] >= BOUNDS[1] - .02) & (points[..., 1] <= BOUNDS[3] + .02) & (points[..., 2] >= -.01) & (points[..., 2] <= .37)
            vv, uu = np.where(crop)
            np.savez_compressed(dst / "pointcloud.npz", xyz_shelf_m=points[crop].astype(np.float32), rgb=rgb[crop], instance_id=ids[crop], pixel_uv=np.stack([uu, vv], axis=1))
            corridor = target_corridor(base_rects[0]).as_list()
            for name, data in (("calibration", calib), ("objects", objects), ("label", label),
                               ("geometry_audit", {"objects": audits, "all_direction_candidates": candidates, "corridor_footprint_m": corridor}),
                               ("visibility_audit", {"definition": "Fractions of target-only rendered silhouette; shelf retained, other objects hidden for reference only", "before": before_metrics, "after": metrics, "trials": trials})):
                write_json(dst / f"{name}.json", data)
            stage.GetRootLayer().Export(str(dst / "scene.usda"))
            annotated_preview(rgb, ids, objects, label, calib).save(dst / "preview.png")
            topdown_preview(objects, label, corridor).save(dst / "topdown.png")
            revised_pos = np.asarray(op.Get()).copy()
            op.Set(Gf.Vec3d(*(revised_pos + np.asarray(label["delta_shelf_m"]) * 100)))
            Image.fromarray(capture()[0]).save(dst / "goal_rgb.png")
            op.Set(Gf.Vec3d(*revised_pos))
            row["label"] = label
            revisions.append({"scene_id": row["scene_id"], "blocker_offset_x_m": offset, "before": before_metrics, "after": metrics,
                              "original_objects": original_objects})
            for ann in annotators.values():
                ann.detach([rp.path])
            rp.destroy()
        manifest["revision"] = {"source": str(source), "changed_scenes": [r["scene_id"] for r in revisions],
                                "change": "Only blocker X position in scenes 2 and 4; all other scene assets, poses and camera preserved",
                                "visibility_policy_for_changed_scenes": {"visible_target_fraction": [0.50, 0.75], "blocker_occlusion_fraction": [0.25, 0.50]}}
        write_json(out / "manifest.json", manifest)
        write_json(out / "progress.json", manifest["samples"])
        write_json(out / "revision.json", revisions)
        sheet = Image.new("RGB", (960, 400 * len(manifest["samples"])), "white")
        for index, row in enumerate(manifest["samples"]):
            folder = out / row["sample_dir"]
            with Image.open(folder / "preview.png") as im:
                sheet.paste(im.resize((480, 360)), (0, index * 400 + 30))
            with Image.open(folder / "topdown.png") as im:
                sheet.paste(im.resize((480, 230)), (480, index * 400 + 95))
            ImageDraw.Draw(sheet).text((12, index * 400 + 5), f'{row["scene_id"]} | target: {row["target_asset"]} | blocker: {row["blocker_asset"]}', font=font(18), fill="black")
        sheet.save(out / "contact_sheet.png")
        comparison = Image.new("RGB", (1000, 420 * len(revisions)), "#eeeeee")
        draw = ImageDraw.Draw(comparison)
        for index, rev in enumerate(revisions):
            relative = Path("samples") / rev["scene_id"]
            masks = [np.asarray(Image.open(base / relative / "instance_ids.png")) for base in (source, out)]
            union = np.isin(masks[0], [1, 2]) | np.isin(masks[1], [1, 2])
            vv, uu = np.where(union)
            cx, cy = (uu.min() + uu.max()) / 2, (vv.min() + vv.max()) / 2
            side = max(int(uu.max() - uu.min()), int(vv.max() - vv.min())) + 80
            crop_box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
            for column, (base, state) in enumerate(((source, "before"), (out, "after"))):
                with Image.open(base / relative / "rgb.png") as im:
                    comparison.paste(im.crop(crop_box).resize((340, 340)), (column * 500 + 80, index * 420 + 65))
                m = rev[state]
                draw.text((column * 500 + 12, index * 420 + 5), f'{rev["scene_id"]} {state.upper()}', font=font(21), fill="black")
                draw.text((column * 500 + 12, index * 420 + 34), f'Target visible {m["visible_target_fraction"]:.1%} | blocker occludes {m["blocker_occlusion_fraction"]:.1%}', font=font(17), fill="black")
        comparison.save(out / "visibility_before_after.png")
        print(f"[revision] COMPLETE: {out}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
