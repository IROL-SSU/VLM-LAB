#!/usr/bin/env python3
"""Independent all-sample audit, split/diversity report and browsable galleries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_translate_rgbd_pilot import font, write_json
from revise_translate_pilot_visibility import read
from validate_translate_rgbd_pilot import validate


def summarize(folder, allow_partial=False):
    manifest = read(folder / "manifest.json")
    rows = manifest["samples"]
    if not allow_partial:
        assert len(rows) == 1000, f"Expected 1000, got {len(rows)}"
    report = validate(folder)
    write_json(folder / "validation.json", report)
    splits = read(folder / "splits.json")
    assert len(set(r["scene_id"] for r in rows)) == len(rows)
    assert len(set(r["rgb_sha256"] for r in rows)) == len(rows), "Duplicate RGB"
    assert len(set(r["layout_sha256"] for r in rows)) == len(rows), "Duplicate geometric layout"
    flat_splits = [scene_id for values in splits.values() for scene_id in values]
    assert len(flat_splits) == len(set(flat_splits)) == len(rows)
    assert set(flat_splits) == {r["scene_id"] for r in rows}
    near_groups = defaultdict(list)
    pair_counts = defaultdict(Counter)
    for row in rows:
        assert row["scene_id"] in splits[row["split"]]
        objects = read(folder / row["sample_dir"] / "objects.json")
        # Ignore camera/materials. Quantize all object XY to 2cm and yaw to 15deg,
        # preserving role identities, to catch augmentation-like siblings.
        coarse = tuple(sorted((o["asset"], o["role"], *(int(round(x / .02)) for x in o["reference_point_shelf_m"][:2]), int(round(o["yaw_deg"] / 15))) for o in objects))
        near_groups[coarse].append(row)
        pair_counts[row["pair_id"]][(row["split"], row["direction"])] += 1
    cross_split_near = [[r["scene_id"] for r in group] for group in near_groups.values() if len({r["split"] for r in group}) > 1]
    assert not cross_split_near, f"Coarse-layout duplicates across splits: {cross_split_near}"
    directions = Counter(r["direction"] for r in rows)
    if not allow_partial:
        assert directions == {"LEFT": 500, "RIGHT": 500}
        assert {k: len(v) for k, v in splits.items()} == {"train": 800, "validation": 100, "test": 100}
        assert len(pair_counts) == 50
        expected_pair = {(s, d): n for s, n in (("train", 8), ("validation", 1), ("test", 1)) for d in ("LEFT", "RIGHT")}
        assert all(counts == expected_pair for counts in pair_counts.values())
    distances = np.array([r["label"]["distance_m"] * 100 for r in rows])
    stats = {
        "passed": True, "sample_count": len(rows), "independent_per_sample_audit_passed": report["scene_count"],
        "directions": dict(directions), "split_counts": {k: len(v) for k, v in splits.items()},
        "split_directions": {s: dict(Counter(r["direction"] for r in rows if r["split"] == s)) for s in splits},
        "ordered_target_blocker_pairs": len(pair_counts),
        "unique_target_assets": len(set(r["target_asset"] for r in rows)), "unique_blocker_assets": len(set(r["blocker_asset"] for r in rows)),
        "target_asset_counts": dict(Counter(r["target_asset"] for r in rows)), "blocker_asset_counts": dict(Counter(r["blocker_asset"] for r in rows)),
        "distance_bins": dict(Counter(r["distance_bin"] for r in rows)),
        "distance_cm_percentiles_0_25_50_75_100": np.percentile(distances, [0, 25, 50, 75, 100]).tolist(),
        "distance_bin_fallback_count": sum(r["distance_bin"] != r["desired_distance_bin"] for r in rows),
        "distractor_counts": dict(sorted(Counter(len(r["distractor_assets"]) for r in rows).items())),
        "palette_counts": dict(Counter(r["environment"]["shelf_palette"] for r in rows)),
        "target_visible_fraction_range": [min(r["visibility"]["visible_target_fraction"] for r in rows), max(r["visibility"]["visible_target_fraction"] for r in rows)],
        "blocker_occlusion_fraction_range": [min(r["visibility"]["blocker_occlusion_fraction"] for r in rows), max(r["visibility"]["blocker_occlusion_fraction"] for r in rows)],
        "duplicate_rgb_count": 0, "duplicate_exact_layout_count": 0, "cross_split_coarse_layout_duplicates": cross_split_near,
        "coarse_duplicate_definition": "same complete role-labeled object set, XY rounded to 2cm and yaw rounded to 15deg; camera/material ignored",
        "labels": "minimum static AABB target-front corridor clearing displacement; LEFT/RIGHT-only accepted labels; not grasp/executor success",
        "scene_scope": "same shelf mesh, 15 object meshes; randomized identities/yaw/positions/clutter/materials/lighting/calibrated viewpoint",
        "split_scope": "layout generalization, not unseen-object or unseen-pair generalization",
    }
    write_json(folder / "dataset_summary.json", stats)
    manifest["status"] = "validated" if len(rows) == 1000 else "partial generation; produced samples validated"
    manifest["validated_sample_count"] = len(rows)
    write_json(folder / "manifest.json", manifest)
    for split, ids in splits.items():
        subset = {**manifest, "sample_count": len(ids), "validated_sample_count": len(ids), "split": split, "samples": [r for r in rows if r["split"] == split]}
        write_json(folder / f"manifest_{split}.json", subset)
    gallery = folder / "contact_sheets"
    gallery.mkdir(exist_ok=True)
    links = []
    for start in range(0, len(rows), 50):
        subset = rows[start:start+50]
        sheet = Image.new("RGB", (1600, 265 * ((len(subset) + 4) // 5)), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            x, y = index % 5 * 320, index // 5 * 265
            with Image.open(folder / row["sample_dir"] / "preview.png") as im:
                sheet.paste(im.resize((320, 240)), (x, y + 25))
            draw.text((x+4, y+4), f'{row["scene_id"][-4:]} {row["split"]} {row["direction"]} {row["label"]["distance_m"]*100:.1f}cm', font=font(13), fill="black")
        name = f"scenes_{start+1:04d}_{start+len(subset):04d}.jpg"
        sheet.save(gallery / name, quality=88)
        links.append(f'<li><a href="contact_sheets/{name}">Scenes {start+1}–{start+len(subset)}</a></li>')
    # Overview covers palette, split, and distance variation deterministically.
    chosen = []
    for palette in ("light_oak", "walnut", "ivory", "slate", "sage"):
        options = [r for r in rows if r["environment"]["shelf_palette"] == palette]
        chosen.extend(options[:4])
    overview = Image.new("RGB", (1600, 270 * ((len(chosen)+4)//5)), "white")
    draw = ImageDraw.Draw(overview)
    for index, row in enumerate(chosen):
        x, y = index % 5 * 320, index // 5 * 270
        with Image.open(folder / row["sample_dir"] / "rgb.png") as im:
            overview.paste(im.resize((320, 240)), (x, y+30))
        draw.text((x+4, y+4), f'{row["scene_id"][-4:]} {row["environment"]["shelf_palette"]} {row["direction"]} {row["label"]["distance_m"]*100:.1f}cm', font=font(13), fill="black")
    overview.save(folder / "overview.jpg", quality=92)
    content = ('<!doctype html><meta charset="utf-8"><title>TRANSLATE RGB-D dataset</title>'
               '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
               f'<h1>TRANSLATE: {len(rows)} scenes</h1><p>LEFT/RIGHT only. Static geometry labels; no executor validation.</p>'
               '<img src="overview.jpg"><h2>All scenes</h2><ul>' + ''.join(links) + '</ul>'
               '<h2>Audit summary</h2><pre>' + html.escape(json.dumps(stats, indent=2)) + '</pre>')
    (folder / "gallery.html").write_text(content, encoding="utf-8")
    write_json(folder / "completion.json", {"complete": len(rows) == 1000, "audited_samples": len(rows), "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(), "trained_model": False})
    write_json(folder / "progress.json", {"completed": len(rows), "requested": 1000, "status": manifest["status"]})
    print(json.dumps(stats, indent=2), flush=True)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summarize(args.dataset.resolve(), args.allow_partial)
