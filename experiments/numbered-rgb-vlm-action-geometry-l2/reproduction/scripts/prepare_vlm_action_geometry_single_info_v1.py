#!/usr/bin/env python3
"""Prepare Numbered RGB, geometry payloads, GT, and the frozen v1 run table.

Run with Isaac Sim's Python:

  /home/ssu/isaacsim/python.sh \
    scripts/prepare_vlm_action_geometry_single_info_v1.py

The script is resumable at scene granularity. It never edits the source
manifest, source RGB images, or source USD files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--experiment",
    type=Path,
    default=Path("experiments/vlm_action_geometry_single_info_v1"),
)
parser.add_argument("--max-scenes", type=int)
parser.add_argument(
    "--scene-id",
    action="append",
    help="Prepare only the named scene; repeat the option to select multiple scenes.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Regenerate selected scene records even when resumable outputs exist.",
)
parser.add_argument("--settle-ticks", type=int, default=35)
parser.add_argument(
    "--l2-clearance-mode",
    choices=(
        "surface_gap_5cm",
        "finger_placement_zone",
        "visibility_only",
        "occlusion_only",
        "target_access_only",
        "potential_interference",
        "instance_interference",
        "instance_occlusion_ltr",
        "primary_grasp_blocker",
    ),
    default="surface_gap_5cm",
    help=(
        "L2 clearance oracle. The default preserves v1/v2; "
        "finger_placement_zone uses the parallel-jaw side zones."
    ),
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 960})

import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels, upgrade_prim_semantics_to_labels
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Usd, UsdGeom

import real_object_lib as rol
import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/action_scene_variants_20260915"
MANIFEST_PATH = SOURCE / "scene_manifest.json"
EXPERIMENT = args.experiment if args.experiment.is_absolute() else ROOT / args.experiment
RECORDS_DIR = EXPERIMENT / "audit/scene_records"
ATTEMPTS_PATH = EXPERIMENT / "audit/preparation_attempts.jsonl"
CAMERA_DEFAULT = "/World/Cameras/ActionSceneCam"
DEFAULT_RADIUS = 16
MIN_RADIUS = 11
BADGE_GAP_PX = 2
GEOMETRY_METHOD = "visual_convex_hull_shelf_plane_v1"
VISIBILITY_METHOD = "isaac_instance_prim_path_target_only_pixel_exact_v2"
L2_CLEARANCE_MODE = args.l2_clearance_mode
FINGER_SIDE_CLEARANCE_CM = 2.2
L2_PROMPT_VERSION = (
    "finger_placement_zone_v3"
    if L2_CLEARANCE_MODE == "finger_placement_zone"
    else "all_visible_instances_occlusion_left_to_right_v9"
    if L2_CLEARANCE_MODE == "instance_occlusion_ltr"
    else "all_visible_instances_interference_v8"
    if L2_CLEARANCE_MODE == "instance_interference"
    else "single_primary_grasp_blocker_v10"
    if L2_CLEARANCE_MODE == "primary_grasp_blocker"
    else "potential_grasp_interference_v7"
    if L2_CLEARANCE_MODE == "potential_interference"
    else "target_access_boolean_v6"
    if L2_CLEARANCE_MODE == "target_access_only"
    else "occlusion_boolean_v5"
    if L2_CLEARANCE_MODE == "occlusion_only"
    else "visibility_only_v4"
    if L2_CLEARANCE_MODE == "visibility_only"
    else "formal_predicate_v2"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def instance_masks(payload: dict, root_paths: dict[str, str]) -> dict[str, np.ndarray]:
    """Return visible masks keyed by scene-object ID using USD prim paths.

    Isaac Sim normalizes semantic class values to lowercase. Matching masks via
    class labels therefore dropped objects whose prim names contained uppercase
    characters (for example ``Cup_4`` and ``Mug_2``). Instance segmentation's
    ``idToLabels`` maps IDs to stable USD prim paths and avoids that ambiguity.
    """
    segmentation = np.asarray(payload.get("data"))
    if segmentation.ndim > 2:
        segmentation = np.squeeze(segmentation)
    id_to_paths = payload.get("info", {}).get("idToLabels", {})
    result = {}
    for instance_id, root_path in root_paths.items():
        matching = [
            int(semantic_id)
            for semantic_id, value in id_to_paths.items()
            if str(value) == root_path or str(value).startswith(root_path + "/")
        ]
        result[instance_id] = np.isin(segmentation, matching)
    return result


def capture_semantic_payload(annotator) -> dict:
    frames = []
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        for _ in range(2):
            simulation_app.update()
        payload = annotator.get_data()
        data = np.asarray(payload.get("data") if isinstance(payload, dict) else payload)
        if data.size:
            frames.append(payload)
        if len(frames) >= 3:
            sizes = [np.asarray(frame["data"]).size for frame in frames]
            return frames[int(np.argmax(sizes))]
    raise RuntimeError("Semantic annotator produced no non-empty frame")


def label_object(root, label: str) -> None:
    upgrade_prim_semantics_to_labels(root)
    add_labels(root, [label], "class", overwrite=True)
    count = 0
    for descendant in Usd.PrimRange(root):
        if descendant.IsA(UsdGeom.Gprim):
            add_labels(descendant, [label], "class", overwrite=True)
            count += 1
    if count == 0:
        raise RuntimeError(f"No renderable geometry under {root.GetPath()}")


def world_points_for_root(stage, root) -> list[tuple[float, float, float]]:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    output: list[tuple[float, float, float]] = []
    for prim in Usd.PrimRange(root):
        local_points = []
        if prim.IsA(UsdGeom.Mesh):
            values = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            local_points = [Gf.Vec3d(value) for value in values]
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
            local_points = [
                Gf.Vec3d(x, y, z)
                for x in (-half, half)
                for y in (-half, half)
                for z in (-half, half)
            ]
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            radius = float(cylinder.GetRadiusAttr().Get() or 1.0)
            half_height = float(cylinder.GetHeightAttr().Get() or 2.0) / 2.0
            local_points = [
                Gf.Vec3d(
                    radius * math.cos(math.radians(angle)),
                    radius * math.sin(math.radians(angle)),
                    z,
                )
                for angle in range(0, 360, 10)
                for z in (-half_height, half_height)
            ]
        if not local_points:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        for point in local_points:
            value = matrix.Transform(point)
            output.append((float(value[0]), float(value[1]), float(value[2])))
    if output:
        return output
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    bounds = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
    lower, upper = bounds.GetMin(), bounds.GetMax()
    return [
        (float(x), float(y), float(z))
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]


def object_geometry(stage, root) -> dict:
    points = world_points_for_root(stage, root)
    polygon = common.convex_hull((x, y) for x, y, _ in points)
    if len(polygon) < 3:
        raise RuntimeError(f"Degenerate footprint for {root.GetPath()}: {polygon}")
    center = common.polygon_centroid(polygon)
    return {
        "instance_id": root.GetName(),
        "prim_path": str(root.GetPath()),
        "footprint_xy_cm": [[round(x, 6), round(y, 6)] for x, y in polygon],
        "center_xy_cm": [round(center[0], 6), round(center[1], 6)],
        "z_range_cm": [round(min(point[2] for point in points), 6), round(max(point[2] for point in points), 6)],
        "source_point_count": len(points),
    }


def analyze_mask(mask: np.ndarray) -> dict:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("Cannot place a badge on an empty mask")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == largest
    bbox = [
        int(stats[largest, cv2.CC_STAT_LEFT]),
        int(stats[largest, cv2.CC_STAT_TOP]),
        int(stats[largest, cv2.CC_STAT_WIDTH]),
        int(stats[largest, cv2.CC_STAT_HEIGHT]),
    ]
    centroid_x, centroid_y = map(float, centroids[largest])
    padded = np.pad(component.astype(np.uint8), ((1, 1), (1, 1)))
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    threshold = max(1.0, float(distance.max()) * 0.5)
    safe_y, safe_x = np.where(component & (distance >= threshold))
    if not safe_x.size:
        safe_y, safe_x = np.where(component)
    offsets = (
        (0.0, 0.0), (0.0, -0.12), (0.0, 0.12), (-0.1, 0.0),
        (0.1, 0.0), (0.0, -0.24), (0.0, 0.24), (-0.18, -0.12),
        (0.18, -0.12), (-0.18, 0.12), (0.18, 0.12),
    )
    candidates = []
    for offset_x, offset_y in offsets:
        target_x = centroid_x + bbox[2] * offset_x
        target_y = centroid_y + bbox[3] * offset_y
        distances = (safe_x - target_x) ** 2 + (safe_y - target_y) ** 2
        index = int(np.argmin(distances))
        candidate = (int(safe_x[index]), int(safe_y[index]))
        if candidate not in candidates:
            candidates.append(candidate)
    maximum_y, maximum_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    maximum_point = (int(maximum_x), int(maximum_y))
    if maximum_point not in candidates:
        candidates.append(maximum_point)
    return {
        "mask": mask,
        "component": component,
        "visible_area_px": int(mask.sum()),
        "component_count": count - 1,
        "largest_component_area_px": int(stats[largest, cv2.CC_STAT_AREA]),
        "bbox_xywh": bbox,
        "centroid_xy": [round(centroid_x, 3), round(centroid_y, 3)],
        "max_interior_clearance_px": round(float(distance.max()), 3),
        "candidates": candidates,
    }


def badges_overlap(first_xy, first_radius, second_xy, second_radius) -> bool:
    required = first_radius + second_radius + BADGE_GAP_PX
    return (
        (first_xy[0] - second_xy[0]) ** 2 + (first_xy[1] - second_xy[1]) ** 2
        < required**2
    )


def assign_badges(items: list[dict]) -> None:
    order = sorted(
        range(len(items)), key=lambda index: items[index]["mask_analysis"]["largest_component_area_px"]
    )
    placed = {}

    def search(position: int) -> bool:
        if position == len(order):
            return True
        item_index = order[position]
        candidates = items[item_index]["mask_analysis"]["candidates"]
        options = [
            (candidate, radius, rank)
            for rank, candidate in enumerate(candidates)
            for radius in range(DEFAULT_RADIUS, MIN_RADIUS - 1, -1)
        ]
        options.sort(key=lambda value: (value[2] * 10 + DEFAULT_RADIUS - value[1], value[2], -value[1]))
        for candidate, radius, rank in options:
            if any(badges_overlap(candidate, radius, old[0], old[1]) for old in placed.values()):
                continue
            placed[item_index] = (candidate, radius, rank)
            if search(position + 1):
                return True
            placed.pop(item_index)
        return False

    if not search(0):
        raise RuntimeError("Could not place non-overlapping badges")
    for index, (center, radius, rank) in placed.items():
        items[index]["badge_center_xy"] = list(center)
        items[index]["badge_radius_px"] = radius
        items[index]["badge_candidate_rank"] = rank


def draw_badges(source: Path, items: list[dict], destination: Path) -> None:
    image = Image.open(source).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for item in sorted(items, key=lambda value: value["display_id"]):
        x, y = item["badge_center_xy"]
        radius = item["badge_radius_px"]
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(220, 38, 38, 238), outline=(255, 255, 255, 255), width=2,
        )
        font = load_font(max(15, int(round(radius * 1.05))))
        text = str(item["display_id"])
        box = draw.textbbox((0, 0), text, font=font)
        width, height = box[2] - box[0], box[3] - box[1]
        if width > radius * 2 - 4 or height > radius * 2 - 4:
            raise RuntimeError(f"Display ID {text} does not fit badge radius {radius}")
        draw.text(
            (x - width / 2 - box[0], y - height / 2 - box[1]),
            text, font=font, fill=(255, 255, 255, 255),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(destination)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def category_for(instance_id: str) -> str:
    if instance_id == "target_teal":
        return "cylinder"
    if instance_id == "blocker_cracker_box":
        return "box"
    if instance_id in rol.OBJECTS:
        return rol.OBJECTS[instance_id][2]
    return "object"


def rounded_polygon(record: dict):
    return [(float(x), float(y)) for x, y in record["footprint_xy_cm"]]


def finger_placement_analysis(
    objects: list[dict], target_id: str, required_side_clearance_cm: float
) -> dict:
    """Build and test the left/right parallel-jaw finger-placement zones.

    The closing axis is shelf LEFT_RIGHT (the world X axis). Each zone is the
    strip immediately outside one lateral side of the target's shelf-plane
    footprint, with width equal to the gripper's required side clearance. A
    physical object blocks a side only when its footprint intersects that
    strip. Proximity in FRONT/BACK therefore does not fail this predicate.
    """
    target = next(item for item in objects if item["instance_id"] == target_id)
    target_polygon = rounded_polygon(target)
    xs = [point[0] for point in target_polygon]
    ys = [point[1] for point in target_polygon]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = float(required_side_clearance_cm)
    zones = {
        "LEFT": [
            (xmin - width, ymin), (xmin, ymin), (xmin, ymax), (xmin - width, ymax)
        ],
        "RIGHT": [
            (xmax, ymin), (xmax + width, ymin), (xmax + width, ymax), (xmax, ymax)
        ],
    }
    relations = {}
    blockers_by_side = {"LEFT": [], "RIGHT": []}
    for item in objects:
        if item["instance_id"] == target_id:
            continue
        polygon = rounded_polygon(item)
        distances = {
            side: common.polygon_distance(polygon, zone)
            for side, zone in zones.items()
        }
        occupied = [side for side, gap in distances.items() if gap <= 1e-7]
        for side in occupied:
            blockers_by_side[side].append(item["instance_id"])
        relations[item["instance_id"]] = {
            "distance_to_zone_cm": distances,
            "occupied_zones": occupied,
        }
    return {
        "required_side_clearance_cm": width,
        "zones": zones,
        "relations": relations,
        "blockers_by_side": blockers_by_side,
        "blocker_instance_ids": sorted(set(blockers_by_side["LEFT"] + blockers_by_side["RIGHT"])),
    }


def build_geometry_payloads(
    scene_id: str,
    objects: list[dict],
    target_id: str,
    finger_analysis: dict | None = None,
) -> dict[str, dict]:
    # Geometry and GT describe the same physical scene-object universe. A fully
    # occluded object may have no honest badge location in RGB, but it still
    # affects clearance and motion feasibility and must not disappear here.
    geometry_objects = objects
    target = next(item for item in objects if item["instance_id"] == target_id)
    target_polygon = rounded_polygon(target)
    target_center = tuple(target["center_xy_cm"])
    payloads = {}
    if L2_CLEARANCE_MODE == "finger_placement_zone" and finger_analysis is None:
        raise ValueError(f"{scene_id}: finger placement analysis is required")

    relations_numeric = []
    relation_cache = {}
    for item in geometry_objects:
        if item["instance_id"] == target_id:
            continue
        polygon = rounded_polygon(item)
        center = tuple(item["center_xy_cm"])
        dx, dy = center[0] - target_center[0], center[1] - target_center[1]
        bearing = math.degrees(math.atan2(dy, dx)) % 360.0
        gap = common.polygon_distance(target_polygon, polygon)
        arcs = common.occupied_angular_arc(target_center, polygon)
        relation_cache[item["instance_id"]] = {"bearing": bearing, "gap": gap, "arcs": arcs}
        numeric_relation = {
            "object_id": item["display_id"],
            "surface_gap_cm": round(gap, 1),
            "center_bearing_deg": round(bearing, 1),
            "occupied_angular_sectors_deg": arcs,
        }
        if finger_analysis is not None:
            zone_relation = finger_analysis["relations"][item["instance_id"]]
            numeric_relation["distance_to_finger_placement_zone_cm"] = {
                side: round(value, 3)
                for side, value in zone_relation["distance_to_zone_cm"].items()
            }
        relations_numeric.append(numeric_relation)
    relations_numeric.sort(key=lambda value: value["object_id"])
    payloads["R_NUMERIC"] = {
        "condition": "R_NUMERIC",
        "target_id": target["display_id"],
        "relations": relations_numeric,
    }
    if finger_analysis is not None:
        payloads["R_NUMERIC"]["gripper"] = {
            "type": "PARALLEL_JAW",
            "closing_axis": "SHELF_LEFT_RIGHT",
            "required_side_clearance_cm": finger_analysis["required_side_clearance_cm"],
        }
    for resolution in common.DIRECTIONS:
        relations = []
        for item in geometry_objects:
            if item["instance_id"] == target_id:
                continue
            cached = relation_cache[item["instance_id"]]
            qualitative_relation = {
                "object_id": item["display_id"],
                "center_direction": common.nearest_direction(cached["bearing"], resolution),
                "occupied_target_sides": common.occupied_directions(cached["arcs"], resolution),
            }
            if L2_CLEARANCE_MODE == "surface_gap_5cm":
                qualitative_relation["clearance_zone"] = (
                    "CLEARANCE_ZONE_OVERLAP" if cached["gap"] <= 5.0 else "CLEARANCE_ZONE_CLEAR"
                )
            elif L2_CLEARANCE_MODE == "finger_placement_zone":
                occupied = finger_analysis["relations"][item["instance_id"]]["occupied_zones"]
                qualitative_relation["finger_placement_zones_occupied"] = occupied
            relations.append(qualitative_relation)
        relations.sort(key=lambda value: value["object_id"])
        payloads[f"R_QUALITATIVE_{resolution}"] = {
            "condition": "R_QUALITATIVE",
            "direction_resolution": resolution,
            "target_id": target["display_id"],
            "relations": relations,
        }
        if finger_analysis is not None:
            payloads[f"R_QUALITATIVE_{resolution}"]["finger_placement_zone_status"] = {
                side: "OCCUPIED" if finger_analysis["blockers_by_side"][side] else "CLEAR"
                for side in ("LEFT", "RIGHT")
            }

    for resolution, direction_names in common.DIRECTIONS.items():
        free_numeric = []
        free_qualitative = []
        motion_numeric = []
        motion_qualitative = []
        for item in geometry_objects:
            polygon = rounded_polygon(item)
            other_polygons = [
                rounded_polygon(other) for other in geometry_objects if other is not item
            ]
            free_values = {
                direction: common.directional_free_space(polygon, other_polygons, direction)
                for direction in direction_names
            }
            translation_values = {
                direction: common.motion_capacity_translation(polygon, other_polygons, direction)
                for direction in direction_names
            }
            rotations = {
                "CW": common.motion_capacity_rotation(polygon, other_polygons, -1.0),
                "CCW": common.motion_capacity_rotation(polygon, other_polygons, 1.0),
            }
            free_numeric.append({
                "object_id": item["display_id"],
                "free_space_cm": {name: round(value, 1) for name, value in free_values.items()},
            })
            free_qualitative.append({
                "object_id": item["display_id"],
                "free_space_level": {name: common.free_space_level(value) for name, value in free_values.items()},
            })
            motion_numeric.append({
                "object_id": item["display_id"],
                "max_safe_translation_cm": {name: int(math.floor(value + 1e-9)) for name, value in translation_values.items()},
                "max_safe_rotation_deg": rotations,
            })
            motion_qualitative.append({
                "object_id": item["display_id"],
                "max_safe_translation_level": {name: common.translation_level(value) for name, value in translation_values.items()},
                "max_safe_rotation_level": {name: common.rotation_level(value) for name, value in rotations.items()},
            })
        for rows in (free_numeric, free_qualitative, motion_numeric, motion_qualitative):
            rows.sort(key=lambda value: value["object_id"])
        payloads[f"F_NUMERIC_{resolution}"] = {
            "condition": "F_NUMERIC", "direction_resolution": resolution, "objects": free_numeric,
        }
        payloads[f"F_QUALITATIVE_{resolution}"] = {
            "condition": "F_QUALITATIVE", "direction_resolution": resolution, "objects": free_qualitative,
        }
        payloads[f"M_NUMERIC_{resolution}"] = {
            "condition": "M_NUMERIC", "direction_resolution": resolution, "objects": motion_numeric,
        }
        payloads[f"M_QUALITATIVE_{resolution}"] = {
            "condition": "M_QUALITATIVE", "direction_resolution": resolution, "objects": motion_qualitative,
        }
    return payloads


def process_scene(manifest_scene: dict) -> dict:
    scene_id = manifest_scene["scene_id"]
    source_image = SOURCE / manifest_scene["image"]
    source_usd = SOURCE / manifest_scene["usd"]
    started = time.perf_counter()
    attempt = {"scene_id": scene_id, "started_at": utc_now(), "source_usd": str(source_usd)}
    try:
        context = omni.usd.get_context()
        context.open_stage(str(source_usd))
        for _ in range(args.settle_ticks):
            simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open {source_usd}")
        objects_root = stage.GetPrimAtPath("/World/SceneObjects")
        if not objects_root.IsValid():
            raise RuntimeError("Missing /World/SceneObjects")
        roots = [child for child in objects_root.GetChildren() if child.IsActive()]
        if not roots:
            raise RuntimeError("Scene has no physical object roots")
        geometries = [object_geometry(stage, root) for root in roots]
        geometry_by_id = {item["instance_id"]: item for item in geometries}
        ordered_ids = [
            item["instance_id"]
            for item in sorted(geometries, key=lambda value: (value["center_xy_cm"][0], value["instance_id"]))
        ]
        sparse_ids, assignment_seed, correlation = common.stable_sparse_ids(scene_id, ordered_ids)
        display_by_id = dict(zip(ordered_ids, sparse_ids))
        root_path_by_id = {}
        for index, root in enumerate(roots, 1):
            label = f"v1_instance_{index:03d}_{root.GetName().casefold()}"
            label_object(root, label)
            root_path_by_id[root.GetName()] = str(root.GetPath())
        for _ in range(12):
            simulation_app.update()

        camera_path = manifest_scene.get("camera", CAMERA_DEFAULT)
        render_product = rep.create.render_product(
            camera_path, (1280, 960), force_new=True, name=f"V1Prep_{scene_id}"
        )
        annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation_fast", init_params={"colorize": False}
        )
        annotator.attach([render_product])
        try:
            current_payload = capture_semantic_payload(annotator)
            current_masks = instance_masks(current_payload, root_path_by_id)
            target_instance = manifest_scene["target"]
            original_visibility = {}
            for root in roots:
                if root.GetName() == target_instance:
                    continue
                imageable = UsdGeom.Imageable(root)
                attribute = imageable.GetVisibilityAttr()
                original_visibility[root.GetName()] = attribute.Get() or UsdGeom.Tokens.inherited
                attribute.Set(UsdGeom.Tokens.invisible)
            for _ in range(8):
                simulation_app.update()
            target_only_payload = capture_semantic_payload(annotator)
            target_only_masks = instance_masks(
                target_only_payload,
                {target_instance: root_path_by_id[target_instance]},
            )
            for root in roots:
                if root.GetName() != target_instance:
                    UsdGeom.Imageable(root).GetVisibilityAttr().Set(original_visibility[root.GetName()])
        finally:
            annotator.detach([render_product.path])
            render_product.destroy()

        target_mask = current_masks[target_instance]
        target_only_mask = target_only_masks[target_instance]
        target_area = int(target_mask.sum())
        target_only_area = int(target_only_mask.sum())
        if target_only_area <= 0:
            raise RuntimeError("Target-only render produced an empty silhouette")
        if target_area == 0:
            visibility = "NOT_VISIBLE"
        elif np.array_equal(target_mask, target_only_mask):
            visibility = "FULL"
        else:
            visibility = "PARTIAL"

        object_rows = []
        visible_badges = []
        object_mask_dir = EXPERIMENT / f"audit/object_visible_masks/{scene_id}"
        for geometry in geometries:
            instance_id = geometry["instance_id"]
            mask = current_masks[instance_id]
            row = {
                **geometry,
                "category": category_for(instance_id),
                "display_id": display_by_id[instance_id],
                "visible": bool(mask.any()),
                "visible_area_px": int(mask.sum()),
                "visible_centroid_xy": (
                    [
                        round(float(np.where(mask)[1].mean()), 6),
                        round(float(np.where(mask)[0].mean()), 6),
                    ]
                    if bool(mask.any())
                    else None
                ),
            }
            if row["visible"]:
                analysis = analyze_mask(mask)
                badge = {"instance_id": instance_id, "display_id": row["display_id"], "mask_analysis": analysis}
                visible_badges.append(badge)
            save_mask(object_mask_dir / f"{row['display_id']:03d}_{instance_id}.png", mask)
            object_rows.append(row)
        assign_badges(visible_badges)

        numbered_path = EXPERIMENT / f"images/numbered_rgb/{scene_id}.png"
        draw_badges(source_image, visible_badges, numbered_path)
        instance_map = np.zeros(target_mask.shape, dtype=np.uint16)
        badge_by_id = {item["instance_id"]: item for item in visible_badges}
        mapping_rows = []
        for row in sorted(object_rows, key=lambda value: value["display_id"]):
            instance_id = row["instance_id"]
            if row["visible"]:
                instance_map[current_masks[instance_id]] = row["display_id"]
            badge = badge_by_id.get(instance_id)
            mapping_rows.append({
                "display_id": row["display_id"],
                "simulator_instance_id": instance_id,
                "prim_path": row["prim_path"],
                "category": row["category"],
                "visible": row["visible"],
                "visible_area_px": row["visible_area_px"],
                "visible_centroid_xy": row["visible_centroid_xy"],
                "badge_center_xy": badge.get("badge_center_xy") if badge else None,
                "badge_radius_px": badge.get("badge_radius_px") if badge else None,
                "badge_candidate_rank": badge.get("badge_candidate_rank") if badge else None,
            })
        map_image_path = EXPERIMENT / f"images/id_mappings/{scene_id}_visible_ids.png"
        map_image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(instance_map).save(map_image_path)
        target_current_path = EXPERIMENT / f"audit/masks/{scene_id}_target_current.png"
        target_only_path = EXPERIMENT / f"audit/masks/{scene_id}_target_only.png"
        save_mask(target_current_path, target_mask)
        save_mask(target_only_path, target_only_mask)

        mapping_path = EXPERIMENT / f"images/id_mappings/{scene_id}.json"
        mapping_document = {
            "scene_id": scene_id,
            "assignment_seed": assignment_seed,
            "left_to_right_instance_ids": ordered_ids,
            "left_to_right_display_ids": sparse_ids,
            "spearman_id_vs_left_to_right": round(correlation, 6),
            "target_object_id": display_by_id[target_instance],
            "instances": mapping_rows,
        }
        common.write_json(mapping_path, mapping_document)

        target_polygon = rounded_polygon(next(item for item in object_rows if item["instance_id"] == target_instance))
        gaps = {}
        for row in object_rows:
            if row["instance_id"] != target_instance:
                gaps[row["instance_id"]] = common.polygon_distance(target_polygon, rounded_polygon(row))
        finger_analysis = None
        if L2_CLEARANCE_MODE == "finger_placement_zone":
            required_side_clearance_cm = float(
                manifest_scene.get("gripper_assumption", {}).get(
                    "required_side_clearance_cm", FINGER_SIDE_CLEARANCE_CM
                )
            )
            finger_analysis = finger_placement_analysis(
                object_rows, target_instance, required_side_clearance_cm
            )
            clearance_blockers = sorted(
                display_by_id[instance_id]
                for instance_id in finger_analysis["blocker_instance_ids"]
            )
        elif L2_CLEARANCE_MODE in {
            "visibility_only", "occlusion_only", "target_access_only",
            "potential_interference", "instance_interference", "instance_occlusion_ltr",
        }:
            clearance_blockers = []
        else:
            clearance_blockers = sorted(
                display_by_id[instance_id] for instance_id, gap in gaps.items() if gap <= 5.0
            )
        missing_target = target_only_mask & ~target_mask
        occlusion_blockers = sorted(
            display_by_id[instance_id]
            for instance_id, mask in current_masks.items()
            if instance_id != target_instance and bool((mask & missing_target).any())
        )
        has_occlusion = visibility != "FULL"
        has_clearance = bool(clearance_blockers)
        if has_occlusion and has_clearance:
            blocking_reason = "BOTH"
        elif has_occlusion:
            blocking_reason = "OCCLUSION"
        elif has_clearance:
            blocking_reason = "CLEARANCE_OVERLAP"
        else:
            blocking_reason = "NONE"
        direct_graspable = visibility == "FULL" and not has_clearance
        valid_blockers = sorted(set(occlusion_blockers) | set(clearance_blockers))
        gt_path = EXPERIMENT / f"geometry/ground_truth/{scene_id}.json"
        gt = {
            "scene_id": scene_id,
            "target_object_id": display_by_id[target_instance],
            "L1": {
                "target_found": visibility != "NOT_VISIBLE",
                "visibility": visibility,
                "visible_area_px": target_area,
                "target_only_area_px": target_only_area,
                "visibility_ratio": round(target_area / target_only_area, 9),
            },
            "L2": {
                "direct_graspable": direct_graspable,
                "minimum_surface_gap_cm": round(min(gaps.values()) if gaps else float("inf"), 6),
                "surface_gaps_cm": {
                    str(display_by_id[key]): round(value, 6) for key, value in sorted(gaps.items())
                },
            },
            "L3": {
                "blocking_reason": blocking_reason,
                "valid_blocker_ids": valid_blockers,
                "occlusion_blocker_ids": occlusion_blockers,
                "clearance_blocker_ids": clearance_blockers,
            },
            "L4": {
                "evaluation": "direct_isaac_sim_replay",
                "manifest_focus_action_advisory_only": manifest_scene.get("focus_action"),
                "manifest_action_parameters_advisory_only": manifest_scene.get("action_parameters"),
            },
            "methods": {
                "visibility": VISIBILITY_METHOD,
                "surface_gap": GEOMETRY_METHOD,
                "full_pixel_rule": "numpy.array_equal(current_target_mask,target_only_mask)",
                "clearance_mode": L2_CLEARANCE_MODE,
            },
        }
        if L2_CLEARANCE_MODE == "occlusion_only":
            gt["L2"]["occluded_by_other_object"] = has_occlusion
        elif L2_CLEARANCE_MODE == "target_access_only":
            gt["L2"]["target_access_blocked"] = has_occlusion
        elif L2_CLEARANCE_MODE == "potential_interference":
            occlusion_ratio = 1.0 - (target_area / target_only_area)
            gt["L2"].update({
                "potential_grasp_interference": occlusion_ratio >= 0.05,
                "occlusion_ratio": round(occlusion_ratio, 9),
                "meaningful_occlusion_threshold": 0.05,
            })
            gt["methods"]["potential_grasp_interference_rule"] = (
                "occlusion_ratio_greater_than_or_equal_to_0.05"
            )
        elif L2_CLEARANCE_MODE == "instance_interference":
            assessment_rows = []
            for row in sorted(object_rows, key=lambda value: value["display_id"]):
                if not row["visible"]:
                    continue
                object_id = int(row["display_id"])
                if row["instance_id"] == target_instance:
                    relation = "TARGET"
                    overlap_pixels = 0
                    overlap_ratio = 0.0
                else:
                    overlap_pixels = int(
                        (current_masks[row["instance_id"]] & missing_target).sum()
                    )
                    overlap_ratio = overlap_pixels / target_only_area
                    relation = (
                        "INTERFERES"
                        if overlap_ratio >= 0.05
                        else "DOES_NOT_INTERFERE"
                    )
                assessment_rows.append({
                    "object_id": object_id,
                    "relation_to_target": relation,
                    "candidate_occlusion_pixels": overlap_pixels,
                    "candidate_occlusion_ratio": round(overlap_ratio, 9),
                })
            gt["L2"].update({
                "target_id": int(display_by_id[target_instance]),
                "instance_assessments": assessment_rows,
                "meaningful_occlusion_threshold": 0.05,
            })
            gt["methods"]["instance_interference_rule"] = (
                "per_visible_instance_candidate_occlusion_ratio_"
                "greater_than_or_equal_to_0.05"
            )
        elif L2_CLEARANCE_MODE == "instance_occlusion_ltr":
            visible_rows_ltr = sorted(
                (row for row in object_rows if row["visible"]),
                key=lambda row: (row["visible_centroid_xy"][0], row["display_id"]),
            )
            visible_ids_ltr = [int(row["display_id"]) for row in visible_rows_ltr]
            assessment_rows = []
            for row in visible_rows_ltr:
                if row["instance_id"] == target_instance:
                    continue
                overlap_pixels = int(
                    (current_masks[row["instance_id"]] & missing_target).sum()
                )
                overlap_ratio = overlap_pixels / target_only_area
                assessment_rows.append({
                    "object_id": int(row["display_id"]),
                    "occludes_target": overlap_ratio >= 0.05,
                    "candidate_occlusion_pixels": overlap_pixels,
                    "candidate_occlusion_ratio": round(overlap_ratio, 9),
                })
            gt["L2"].update({
                "target_id": int(display_by_id[target_instance]),
                "visible_object_ids_left_to_right": visible_ids_ltr,
                "occlusion_assessments_left_to_right": assessment_rows,
                "meaningful_occlusion_threshold": 0.05,
            })
            gt["methods"]["instance_occlusion_rule"] = (
                "per_visible_non_target_instance_candidate_occlusion_ratio_"
                "greater_than_or_equal_to_0.05"
            )
            gt["methods"]["left_to_right_order"] = (
                "visible_instance_mask_pixel_centroid_x_then_display_id"
            )
        elif L2_CLEARANCE_MODE == "primary_grasp_blocker":
            blocker_candidates = []
            for row in object_rows:
                if not row["visible"] or row["instance_id"] == target_instance:
                    continue
                overlap_pixels = int(
                    (current_masks[row["instance_id"]] & missing_target).sum()
                )
                overlap_ratio = overlap_pixels / target_only_area
                if overlap_ratio >= 0.05:
                    blocker_candidates.append({
                        "object_id": int(row["display_id"]),
                        "candidate_occlusion_pixels": overlap_pixels,
                        "candidate_occlusion_ratio": round(overlap_ratio, 9),
                    })
            blocker_candidates.sort(
                key=lambda value: (
                    -value["candidate_occlusion_ratio"], value["object_id"]
                )
            )
            gt["L2"].update({
                "object_to_remove_id": (
                    blocker_candidates[0]["object_id"]
                    if blocker_candidates else None
                ),
                "primary_blocker_candidates": blocker_candidates,
                "meaningful_occlusion_threshold": 0.05,
            })
            gt["methods"]["primary_grasp_blocker_rule"] = (
                "largest_visible_non_target_instance_candidate_occlusion_ratio_"
                "greater_than_or_equal_to_0.05_then_smallest_display_id"
            )
        if finger_analysis is None:
            if L2_CLEARANCE_MODE == "surface_gap_5cm":
                gt["methods"]["clearance_threshold_cm"] = 5.0
            else:
                gt["methods"]["direct_graspable_rule"] = "visibility_is_FULL_only"
                gt["methods"]["clearance_ignored"] = True
        else:
            gt["L2"].update({
                "gripper": {
                    "type": "PARALLEL_JAW",
                    "closing_axis": "SHELF_LEFT_RIGHT",
                    "required_side_clearance_cm": finger_analysis["required_side_clearance_cm"],
                },
                "finger_placement_zones_xy_cm": {
                    side: [[round(x, 6), round(y, 6)] for x, y in polygon]
                    for side, polygon in finger_analysis["zones"].items()
                },
                "finger_placement_zone_status": {
                    side: "OCCUPIED" if finger_analysis["blockers_by_side"][side] else "CLEAR"
                    for side in ("LEFT", "RIGHT")
                },
                "finger_zone_blocker_ids": clearance_blockers,
            })
            gt["methods"].update({
                "clearance_oracle": "target_side_finger_zone_footprint_intersection",
                "required_side_clearance_cm": finger_analysis["required_side_clearance_cm"],
            })
        common.write_json(gt_path, gt)

        l2_instance_schema = None
        if L2_CLEARANCE_MODE in {"instance_interference", "instance_occlusion_ltr"}:
            l2_instance_schema_path = EXPERIMENT / f"schemas/l2_instances/{scene_id}.json"
            if L2_CLEARANCE_MODE == "instance_occlusion_ltr":
                visible_ids = [
                    int(row["display_id"])
                    for row in sorted(
                        (row for row in object_rows if row["visible"]),
                        key=lambda row: (
                            row["visible_centroid_xy"][0], row["display_id"]
                        ),
                    )
                ]
                scene_schema = instance_occlusion_ltr_schema(
                    visible_ids, int(display_by_id[target_instance])
                )
            else:
                visible_ids = sorted(
                    int(row["display_id"]) for row in object_rows if row["visible"]
                )
                scene_schema = instance_interference_schema(
                    visible_ids, int(display_by_id[target_instance])
                )
            common.write_json(
                l2_instance_schema_path,
                scene_schema,
            )
            l2_instance_schema = {
                "path": str(l2_instance_schema_path.relative_to(EXPERIMENT)),
                "sha256": common.sha256_path(l2_instance_schema_path),
            }

        geometry_payloads = build_geometry_payloads(
            scene_id, object_rows, target_instance, finger_analysis
        )
        geometry_files = {}
        folder_by_prefix = {
            "R_NUMERIC": "r_numeric", "R_QUALITATIVE": "r_qualitative",
            "F_NUMERIC": "f_numeric", "F_QUALITATIVE": "f_qualitative",
            "M_NUMERIC": "m_numeric", "M_QUALITATIVE": "m_qualitative",
        }
        for key, payload in geometry_payloads.items():
            prefix = next(prefix for prefix in folder_by_prefix if key.startswith(prefix))
            suffix = key[len(prefix):].strip("_").lower()
            folder = folder_by_prefix[prefix] + (f"_{suffix}" if suffix else "")
            path = EXPERIMENT / f"geometry/{folder}/{scene_id}.json"
            common.write_json(path, payload)
            geometry_files[key] = {
                "path": str(path.relative_to(EXPERIMENT)),
                "sha256": common.sha256_path(path),
            }

        geometry_record_path = EXPERIMENT / f"audit/scene_geometry/{scene_id}.json"
        common.write_json(geometry_record_path, {
            "scene_id": scene_id,
            "shelf_bounds_cm": list(common.SHELF_BOUNDS_CM),
            "geometry_method": GEOMETRY_METHOD,
            "objects": object_rows,
        })
        record = {
            "scene_id": scene_id,
            "scene_family": common.scene_family(scene_id),
            "source_rgb": str(source_image.relative_to(ROOT)),
            "source_rgb_sha256": common.sha256_path(source_image),
            "source_usd": str(source_usd.relative_to(ROOT)),
            "source_usd_sha256": common.sha256_path(source_usd),
            "numbered_rgb": str(numbered_path.relative_to(EXPERIMENT)),
            "numbered_rgb_sha256": common.sha256_path(numbered_path),
            "id_mapping": str(mapping_path.relative_to(EXPERIMENT)),
            "id_mapping_sha256": common.sha256_path(mapping_path),
            "target_object_id": display_by_id[target_instance],
            "visible_ids": sorted(row["display_id"] for row in object_rows if row["visible"]),
            "ground_truth": str(gt_path.relative_to(EXPERIMENT)),
            "ground_truth_sha256": common.sha256_path(gt_path),
            "scene_geometry": str(geometry_record_path.relative_to(EXPERIMENT)),
            "scene_geometry_sha256": common.sha256_path(geometry_record_path),
            "geometry_files": geometry_files,
            "l2_instance_schema": l2_instance_schema,
            "manifest_focus_action_advisory_only": manifest_scene.get("focus_action"),
            "manifest_action_parameters_advisory_only": manifest_scene.get("action_parameters"),
        }
        common.write_json(RECORDS_DIR / f"{scene_id}.json", record)
        attempt.update({"status": "success", "finished_at": utc_now(), "wall_seconds": round(time.perf_counter() - started, 3)})
        append_jsonl(ATTEMPTS_PATH, attempt)
        return record
    except Exception as exc:
        attempt.update({
            "status": "error", "finished_at": utc_now(),
            "wall_seconds": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        append_jsonl(ATTEMPTS_PATH, attempt)
        raise


SYSTEM_PROMPT = """You are a robot manipulation decision module for a cluttered shelf.

Each red numbered badge identifies the physical object underneath it.
The badge position, size, and number are identity markers only. Do not use
them as evidence of object geometry, distance, depth, or action feasibility.

Judge only the current observation. Do not generate a multi-step sequence.
Return exactly one JSON object matching the supplied schema.
Do not output explanations, Markdown, confidence scores, or hidden reasoning."""

LEVEL_PROMPTS = {
    "L1": """Determine whether the target object can be located in the numbered RGB image
and classify its visibility.

FULL means 100% of the target-only projected silhouette is visible, ignoring
the badge overlay. PARTIAL means some but not all of that silhouette is
visible. NOT_VISIBLE means no part of the target can be located.""",
    "L2": """For this benchmark, direct_graspable is a formal Boolean predicate.
It does not mean general real-world grasp feasibility.

Evaluate the following two tests independently.

1. Visibility test
Determine visibility from the numbered RGB image.
The test passes only when the target is FULL.

FULL means 100% of the target-only projected silhouette is visible,
ignoring the numbered badge overlay.
PARTIAL and NOT_VISIBLE fail this test.

Geometry describing distance, direction, or occupied angular sectors
must not be used as evidence that the target is visually occluded.

2. Clearance test
The test passes only when every other physical object has a
surface-to-surface gap strictly greater than 5.0 cm from the target's
outer boundary.

Use the supplied information as follows:
- For R_NUMERIC, surface_gap_cm is authoritative.
  Every surface_gap_cm must be greater than 5.0.
- For R_QUALITATIVE, clearance_zone is authoritative.
  CLEARANCE_ZONE_CLEAR passes.
  CLEARANCE_ZONE_OVERLAP fails.
- If neither clearance representation is supplied, make the best
  binary estimate from the numbered RGB and the available information.
  Do not return false only because an exact distance is unavailable.
- Other geometry may help evaluate these two tests, but it must not
  introduce an additional decision criterion.

Final decision:
- Return direct_graspable=true if and only if both tests pass.
- Otherwise return direct_graspable=false.

Exact boundary:
- 5.0 cm fails.
- Any value greater than 5.0 cm passes.

Do not add approach-path, gripper-pose, reachability, stability,
collision-path, confidence, uncertainty, or safety-margin requirements.

Examples:
- FULL and every gap > 5.0 cm -> true
- FULL and any gap <= 5.0 cm -> false
- PARTIAL and every gap > 5.0 cm -> false""",
    "L3": """Identify why the target is not directly graspable and select one causal
blocker for the next action.

OCCLUSION means the target is not FULL.
CLEARANCE_OVERLAP means another object is within 5 cm of the target's outer
boundary. BOTH means both conditions hold. NONE means the target is directly
graspable.

If several blockers are valid, select one. Use null when the reason is NONE.""",
    "L4": """Choose exactly one next action from:
RETRIEVE, TRANSLATE, ROTATE, LIFT_AND_RELOCATE.

RETRIEVE: directly grasp and remove the target. Use it only when the target
is directly graspable.

TRANSLATE: move one selected object along the shelf plane. Output a shelf
direction, an integer distance in 1 cm increments, and its qualitative level.

ROTATE: rotate one selected object around the vertical axis through the
center of its footprint while keeping the center approximately fixed.
Output CW or CCW as viewed from above, an angle in 5-degree increments,
and its qualitative level.

LIFT_AND_RELOCATE: lift one selected object and relocate it. Do not output
a destination, lift height, or path.

A proposed action is valid only if it can be executed without contacting any
unselected object and the target becomes directly graspable after the action.
Output only one next action, not a sequence.

Allowed translation directions:
{direction_list}""",
}

if L2_CLEARANCE_MODE == "finger_placement_zone":
    LEVEL_PROMPTS["L2"] = """For this benchmark, direct_graspable is a formal Boolean predicate.
It does not mean general real-world grasp feasibility.

Evaluate the following two tests independently.

1. Visibility test
Determine visibility from the numbered RGB image.
The test passes only when the target is FULL.

FULL means 100% of the target-only projected silhouette is visible,
ignoring the numbered badge overlay.
PARTIAL and NOT_VISIBLE fail this test.

Geometry describing distance, direction, occupied angular sectors, or
finger-placement clearance must not be used as evidence that the target is
visually occluded.

2. Finger-placement test
Assume a parallel-jaw gripper whose fingers close along the shelf LEFT_RIGHT
axis. The test passes only when the immediate LEFT and RIGHT finger-placement
zones beside the target are both clear.

An object blocks the grasp only when its shelf-plane footprint occupies or
overlaps one of those two finger-placement zones. Mere proximity outside the
finger-placement zones, including proximity in FRONT or BACK, does not fail
this test. Do not impose a universal target-to-object distance threshold.

Use the supplied information as follows:
- For R_NUMERIC, distance_to_finger_placement_zone_cm is authoritative.
  A value of 0 means that object occupies that side's zone; a positive value
  means it is outside that zone. Both LEFT and RIGHT must remain unoccupied.
  surface_gap_cm is supplemental and is not a decision threshold.
- For R_QUALITATIVE, finger_placement_zone_status is authoritative.
  Both LEFT and RIGHT must be CLEAR. OCCUPIED fails the test.
- If neither representation is supplied, make the best binary visual estimate
  of whether there is room immediately beside the target for both fingers.
  Do not return false only because an exact distance is unavailable.
- Other geometry may help evaluate these two tests, but it must not introduce
  an additional decision criterion.

Final decision:
- Return direct_graspable=true if and only if both tests pass.
- Otherwise return direct_graspable=false.

Do not add approach-path, gripper-pose, reachability, stability,
collision-path, confidence, uncertainty, or extra safety-margin requirements.

Examples:
- FULL and LEFT=CLEAR and RIGHT=CLEAR -> true
- FULL and either side OCCUPIED -> false
- PARTIAL and both sides CLEAR -> false"""
    LEVEL_PROMPTS["L3"] = """Identify why the target is not directly graspable and select one causal
blocker for the next action.

OCCLUSION means the target is not FULL.
CLEARANCE_OVERLAP means another object's shelf-plane footprint occupies a
LEFT or RIGHT finger-placement zone beside the target. Mere proximity outside
those zones does not count. BOTH means both conditions hold. NONE means the
target is directly graspable.

If several blockers are valid, select one. Use null when the reason is NONE."""
elif L2_CLEARANCE_MODE == "visibility_only":
    LEVEL_PROMPTS["L2"] = """For this experiment, direct_graspable is a visibility-only Boolean proxy.

Evaluate only whether another physical object visually blocks any part of the
target in the numbered RGB image.

- Return direct_graspable=true when the target is FULLY visible: no other
  physical object covers any part of its projected silhouette.
- Return direct_graspable=false when the target is PARTIALLY visible or
  NOT_VISIBLE because another physical object covers it.

An object merely being beside, near, or touching the target does not make the
target ungraspable in this experiment. Even if there would not be enough room
for gripper fingers, return true whenever the target is FULLY visible.

Ignore target-to-object distance, finger-placement clearance, approach path,
reachability, stability, and all other grasp constraints. Geometry information
must not override the visual occlusion judgment.

Examples:
- Fully visible target with a neighboring object immediately beside it -> true
- Partially hidden target with ample side clearance -> false
- Fully visible isolated target -> true"""
    LEVEL_PROMPTS["L3"] = """Identify whether another physical object visually occludes the target.

OCCLUSION means the target is PARTIAL or NOT_VISIBLE. NONE means the target is
FULLY visible. Clearance and neighboring-object proximity are ignored.

Select one visible occluding object when the reason is OCCLUSION. Use null when
the reason is NONE."""
elif L2_CLEARANCE_MODE == "occlusion_only":
    LEVEL_PROMPTS["L2"] = """Determine whether the target object is occluded by another physical object
in the numbered RGB image.

Return occluded_by_other_object=true if another physical object visually
covers any part of the target object's projected silhouette.

Return occluded_by_other_object=false only if no other physical object
visually covers any part of the target object's projected silhouette.

Consider only visual occlusion. Ignore object proximity, contact, available
finger space, grasp clearance, approach path, reachability, stability, and
all other manipulation constraints.

The supplied geometry information must not override the visual occlusion
judgment."""
    LEVEL_PROMPTS["L3"] = """Identify whether another physical object visually occludes the target.

OCCLUSION means the target is PARTIAL or NOT_VISIBLE. NONE means the target is
FULLY visible. Clearance and neighboring-object proximity are ignored.

Select one visible occluding object when the reason is OCCLUSION. Use null when
the reason is NONE."""
elif L2_CLEARANCE_MODE == "target_access_only":
    LEVEL_PROMPTS["L2"] = """The robot approaches the target from the open front of the shelf.

Determine whether another physical object blocks direct access to the target.

Return target_access_blocked=true only when another physical object lies
between the shelf opening and the target, such that the target cannot be
directly reached without first moving that object.

Otherwise, return target_access_blocked=false."""
    LEVEL_PROMPTS["L3"] = """Identify whether another physical object blocks direct access to the target
from the open front of the shelf.

OCCLUSION means another physical object lies between the shelf opening and the
target. NONE means direct access is not blocked.

Select one visible blocking object when the reason is OCCLUSION. Use null when
the reason is NONE."""
elif L2_CLEARANCE_MODE == "potential_interference":
    LEVEL_PROMPTS["L2"] = """The robot will directly grasp the target from the open front of the shelf.

Determine whether another physical object is likely to interfere with the
gripper reaching or closing on the target.

Return potential_grasp_interference=true when another physical object visibly
overlaps a meaningful part of the target or is directly in front of the target
such that the gripper may contact it during the grasp.

Otherwise, return potential_grasp_interference=false."""
    LEVEL_PROMPTS["L3"] = """Identify whether another physical object is likely to interfere with the
gripper reaching or closing on the target from the open front of the shelf.

Select one visible interfering object when the reason is OCCLUSION. Use null
when the reason is NONE."""
elif L2_CLEARANCE_MODE == "instance_interference":
    LEVEL_PROMPTS["L2"] = """Assess every numbered physical object in the image relative to the target.

The robot will directly grasp the target from the open front of the shelf.

Return exactly one assessment for every numbered object, including the target.
Use TARGET for the target object.
Use INTERFERES when another object is likely to interfere with the gripper
reaching or closing on the target because it visibly overlaps a meaningful
part of the target or is directly in front of the target such that the gripper
may contact it during the grasp.
Use DOES_NOT_INTERFERE otherwise.

List the assessments in ascending object_id order."""
    LEVEL_PROMPTS["L3"] = """Identify whether another physical object is likely to interfere with the
gripper reaching or closing on the target from the open front of the shelf.

Select one visible interfering object when the reason is OCCLUSION. Use null
when the reason is NONE."""
elif L2_CLEARANCE_MODE == "instance_occlusion_ltr":
    LEVEL_PROMPTS["L2"] = """Identify all numbered physical objects visible in the image.

For every numbered object except the target, determine whether that specific
object visually occludes the target.

Set occludes_target=true only when the object is visibly in front of the target
and covers part of the target in the image.

Set occludes_target=false when the object does not directly cause any part of
the target to be visually hidden.

Evaluate visual occlusion only. Do not evaluate grasp interference, proximity,
finger clearance, contact risk, or general action feasibility.

List the numbered objects from left to right according to the horizontal center
of each object's visible region in the image. Do not use the badge position to
determine the ordering.

Return exactly one occlusion assessment for every visible object except the
target, in the same left-to-right order."""
    LEVEL_PROMPTS["L3"] = """Identify whether another physical object visually occludes the target.

OCCLUSION means another physical object visibly covers part of the target.
NONE means no other physical object visibly covers the target.

Select one visible occluding object when the reason is OCCLUSION. Use null when
the reason is NONE."""
elif L2_CLEARANCE_MODE == "primary_grasp_blocker":
    SYSTEM_PROMPT = """You are a robot manipulation planner.
Return only valid JSON."""
    LEVEL_PROMPTS["L2"] = """The robot must grasp the target object.

Select the single numbered object that most obstructs grasping the target.
If no object obstructs the grasp, return null."""


def instance_interference_schema(visible_ids: list[int], target_id: int) -> dict:
    """Build an exact, ordered output schema for one scene's numbered objects."""
    item_schemas = []
    for object_id in sorted(visible_ids):
        relation_schema = (
            {"const": "TARGET"}
            if object_id == target_id
            else {
                "type": "string",
                "enum": ["INTERFERES", "DOES_NOT_INTERFERE"],
            }
        )
        item_schemas.append({
            "type": "object",
            "properties": {
                "object_id": {"const": object_id},
                "relation_to_target": relation_schema,
            },
            "required": ["object_id", "relation_to_target"],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {
            "target_id": {"const": target_id},
            "instance_assessments": {
                "type": "array",
                "prefixItems": item_schemas,
                "items": False,
                "minItems": len(item_schemas),
                "maxItems": len(item_schemas),
            },
        },
        "required": ["target_id", "instance_assessments"],
        "additionalProperties": False,
    }


def instance_occlusion_ltr_schema(visible_ids_ltr: list[int], target_id: int) -> dict:
    """Build an exact left-to-right schema for visible IDs and non-target occlusion."""
    assessment_ids = [object_id for object_id in visible_ids_ltr if object_id != target_id]
    assessment_schemas = [
        {
            "type": "object",
            "properties": {
                "object_id": {"const": object_id},
                "occludes_target": {"type": "boolean"},
            },
            "required": ["object_id", "occludes_target"],
            "additionalProperties": False,
        }
        for object_id in assessment_ids
    ]
    return {
        "type": "object",
        "properties": {
            "visible_object_ids_left_to_right": {
                "type": "array",
                "prefixItems": [{"const": object_id} for object_id in visible_ids_ltr],
                "items": False,
                "minItems": len(visible_ids_ltr),
                "maxItems": len(visible_ids_ltr),
            },
            "target_id": {"const": target_id},
            "occlusion_assessments_left_to_right": {
                "type": "array",
                "prefixItems": assessment_schemas,
                "items": False,
                "minItems": len(assessment_schemas),
                "maxItems": len(assessment_schemas),
            },
        },
        "required": [
            "visible_object_ids_left_to_right",
            "target_id",
            "occlusion_assessments_left_to_right",
        ],
        "additionalProperties": False,
    }


def schemas() -> dict[str, dict]:
    base_l4 = {
        "RETRIEVE": {
            "type": "object", "properties": {
                "action": {"const": "RETRIEVE"}, "object_id": {"type": "integer"},
            }, "required": ["action", "object_id"], "additionalProperties": False,
        },
        "LIFT_AND_RELOCATE": {
            "type": "object", "properties": {
                "action": {"const": "LIFT_AND_RELOCATE"}, "object_id": {"type": "integer"},
            }, "required": ["action", "object_id"], "additionalProperties": False,
        },
    }
    output = {
        "L1": {
            "type": "object", "properties": {
                "target_found": {"type": "boolean"},
                "visibility": {"type": "string", "enum": ["FULL", "PARTIAL", "NOT_VISIBLE"]},
            }, "required": ["target_found", "visibility"], "additionalProperties": False,
        },
        "L2": (
            {
                "type": "object",
                "properties": {
                    "visible_object_ids_left_to_right": {
                        "type": "array", "items": {"type": "integer"}
                    },
                    "target_id": {"type": "integer"},
                    "occlusion_assessments_left_to_right": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "object_id": {"type": "integer"},
                                "occludes_target": {"type": "boolean"},
                            },
                            "required": ["object_id", "occludes_target"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "visible_object_ids_left_to_right", "target_id",
                    "occlusion_assessments_left_to_right",
                ],
                "additionalProperties": False,
            }
            if L2_CLEARANCE_MODE == "instance_occlusion_ltr"
            else {
                "type": "object",
                "properties": {
                    "object_to_remove_id": {"type": ["integer", "null"]}
                },
                "required": ["object_to_remove_id"],
                "additionalProperties": False,
            }
            if L2_CLEARANCE_MODE == "primary_grasp_blocker"
            else {
                "type": "object",
                "properties": {
                    "target_id": {"type": "integer"},
                    "instance_assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "object_id": {"type": "integer"},
                                "relation_to_target": {
                                    "type": "string",
                                    "enum": [
                                        "TARGET", "INTERFERES",
                                        "DOES_NOT_INTERFERE",
                                    ],
                                },
                            },
                            "required": ["object_id", "relation_to_target"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["target_id", "instance_assessments"],
                "additionalProperties": False,
            }
            if L2_CLEARANCE_MODE == "instance_interference"
            else {
                "type": "object",
                "properties": {"potential_grasp_interference": {"type": "boolean"}},
                "required": ["potential_grasp_interference"],
                "additionalProperties": False,
            }
            if L2_CLEARANCE_MODE == "potential_interference"
            else {
                "type": "object",
                "properties": {"target_access_blocked": {"type": "boolean"}},
                "required": ["target_access_blocked"],
                "additionalProperties": False,
            }
            if L2_CLEARANCE_MODE == "target_access_only"
            else {
                "type": "object",
                "properties": {"occluded_by_other_object": {"type": "boolean"}},
                "required": ["occluded_by_other_object"],
                "additionalProperties": False,
            }
            if L2_CLEARANCE_MODE == "occlusion_only"
            else {
                "type": "object",
                "properties": {"direct_graspable": {"type": "boolean"}},
                "required": ["direct_graspable"],
                "additionalProperties": False,
            }
        ),
        "L3": {
            "type": "object", "properties": {
                "blocking_reason": {"type": "string", "enum": ["OCCLUSION", "CLEARANCE_OVERLAP", "BOTH", "NONE"]},
                "blocker_id": {"type": ["integer", "null"]},
            }, "required": ["blocking_reason", "blocker_id"], "additionalProperties": False,
        },
    }
    for resolution, directions in common.DIRECTIONS.items():
        translate = {
            "type": "object", "properties": {
                "action": {"const": "TRANSLATE"}, "object_id": {"type": "integer"},
                "direction": {"type": "string", "enum": list(directions)},
                "distance_cm": {"type": "integer", "minimum": 1},
                "distance_level": {"type": "string", "enum": ["SHORT", "MEDIUM", "LONG"]},
            },
            "required": ["action", "object_id", "direction", "distance_cm", "distance_level"],
            "additionalProperties": False,
        }
        rotate = {
            "type": "object", "properties": {
                "action": {"const": "ROTATE"}, "object_id": {"type": "integer"},
                "rotation_direction": {"type": "string", "enum": ["CW", "CCW"]},
                "angle_deg": {"type": "integer", "minimum": 5, "multipleOf": 5},
                "angle_level": {"type": "string", "enum": ["SMALL", "MEDIUM", "LARGE"]},
            },
            "required": ["action", "object_id", "rotation_direction", "angle_deg", "angle_level"],
            "additionalProperties": False,
        }
        output[f"L4_{resolution}"] = {"oneOf": [base_l4["RETRIEVE"], translate, rotate, base_l4["LIFT_AND_RELOCATE"]]}
    return output


def write_static_files(manifest: dict) -> dict[str, str]:
    prompt_paths = {}
    for name, content in {"system": SYSTEM_PROMPT, **{key.lower(): value for key, value in LEVEL_PROMPTS.items()}}.items():
        path = EXPERIMENT / f"prompts/{name}_en.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")
        prompt_paths[name] = str(path.relative_to(EXPERIMENT))
    schema_paths = {}
    for name, schema in schemas().items():
        path = EXPERIMENT / f"schemas/{name.lower()}.json"
        common.write_json(path, schema)
        schema_paths[name] = str(path.relative_to(EXPERIMENT))
    config = {
        "experiment_version": common.EXPERIMENT_VERSION,
        "l2_prompt_version": L2_PROMPT_VERSION,
        "source_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "source_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "scene_count": len(manifest["scenes"]),
        "scheduled_calls": 6250,
        "main_seeds": list(common.SEEDS),
        "run_order_seed": common.RUN_ORDER_SEED,
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_path": str(ROOT / "models/Qwen3-VL-30B-A3B-Instruct"),
        "generation": {
            "temperature": 0.3, "top_p": 0.9, "top_k": 0,
            "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "max_tokens": 1024,
        },
        "runtime": {
            "backend": "vllm_offline", "batch_size": 1, "max_num_seqs": 1,
            "max_model_len": 8192, "gpu_memory_utilization": 0.9,
            "moe_backend": "triton", "structured_output": "json_schema_oneOf",
        },
        "isaac_sim": {
            "geometry_method": GEOMETRY_METHOD,
            "visibility_method": VISIBILITY_METHOD,
            "l2_clearance_mode": L2_CLEARANCE_MODE,
            "parallel_jaw_closing_axis": "SHELF_LEFT_RIGHT",
            "required_side_clearance_cm": (
                FINGER_SIDE_CLEARANCE_CM
                if L2_CLEARANCE_MODE == "finger_placement_zone"
                else None
            ),
            "shelf_bounds_cm": list(common.SHELF_BOUNDS_CM),
            "motion_translation_internal_step_cm": 0.25,
            "motion_rotation_step_deg": 5,
        },
        "prompts": prompt_paths,
        "schemas": schema_paths,
    }
    config_path = EXPERIMENT / "config/experiment_config.json"
    common.write_json(config_path, config)
    return config


def geometry_key(condition_id: str, resolution: str | None) -> str | None:
    prefix = {
        "C0": None, "C1": "R_NUMERIC", "C2": "R_QUALITATIVE",
        "C3": "F_NUMERIC", "C4": "F_QUALITATIVE",
        "C5": "M_NUMERIC", "C6": "M_QUALITATIVE",
    }[condition_id]
    if prefix is None:
        return None
    if prefix == "R_NUMERIC":
        return prefix
    return f"{prefix}_{resolution}"


def condition_rows(level: str) -> list[tuple[str, str | None]]:
    if level == "L4":
        return [(condition, resolution) for condition in ("C0", "C1", "C2", "C3", "C4", "C5", "C6") for resolution in ("D4", "D8")]
    return [
        ("C0", None), ("C1", None),
        ("C2", "D4"), ("C2", "D8"),
        ("C3", "D4"), ("C3", "D8"),
        ("C4", "D4"), ("C4", "D8"),
        ("C5", "D4"), ("C5", "D8"),
        ("C6", "D4"), ("C6", "D8"),
    ]


def build_run_table(scenes: list[dict], config: dict) -> list[dict]:
    rows = []
    for scene in scenes:
        for level in ("L1", "L2", "L3", "L4"):
            for condition_id, resolution in condition_rows(level):
                key = geometry_key(condition_id, resolution)
                geometry = scene["geometry_files"].get(key) if key else None
                schema_key = f"L4_{resolution}" if level == "L4" else level
                if level == "L2" and L2_CLEARANCE_MODE in {
                    "instance_interference", "instance_occlusion_ltr"
                }:
                    schema_relative = scene["l2_instance_schema"]["path"]
                    schema_sha256 = scene["l2_instance_schema"]["sha256"]
                else:
                    schema_relative = config["schemas"][schema_key]
                    schema_sha256 = common.sha256_path(EXPERIMENT / schema_relative)
                for repeat_index, seed in enumerate(common.SEEDS, 1):
                    run_id = (
                        f"v1__{scene['scene_id']}__{level}__{condition_id}_"
                        f"{resolution or 'NA'}__r{repeat_index:02d}_s{seed}"
                    )
                    rows.append({
                        "run_id": run_id,
                        "experiment_version": common.EXPERIMENT_VERSION,
                        "scene_id": scene["scene_id"],
                        "scene_family": scene["scene_family"],
                        "level": level,
                        "geometry_condition": condition_id,
                        "geometry_key": key,
                        "direction_resolution": resolution,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "target_object_id": scene["target_object_id"],
                        "numbered_rgb": scene["numbered_rgb"],
                        "numbered_rgb_sha256": scene["numbered_rgb_sha256"],
                        "source_usd": scene["source_usd"],
                        "source_usd_sha256": scene["source_usd_sha256"],
                        "id_mapping": scene["id_mapping"],
                        "id_mapping_sha256": scene["id_mapping_sha256"],
                        "ground_truth": scene["ground_truth"],
                        "ground_truth_sha256": scene["ground_truth_sha256"],
                        "geometry_json": geometry["path"] if geometry else None,
                        "geometry_json_sha256": geometry["sha256"] if geometry else common.sha256_text("null"),
                        "schema": schema_relative,
                        "schema_sha256": schema_sha256,
                    })
    random.Random(common.RUN_ORDER_SEED).shuffle(rows)
    return rows


def audit_object_coverage(records: list[dict]) -> dict:
    """Verify that RGB masks, badges, and geometry cover their intended universes."""
    failures = []
    design_label_mismatches = []
    object_count = 0
    visible_object_count = 0
    fully_occluded_objects = []
    for record in records:
        scene_id = record["scene_id"]
        mapping = json.loads((EXPERIMENT / record["id_mapping"]).read_text(encoding="utf-8"))
        rows = mapping["instances"]
        all_ids = {int(row["display_id"]) for row in rows}
        target_id = int(mapping["target_object_id"])
        visible_ids = {int(row["display_id"]) for row in rows if row["visible"]}
        badge_ids = {
            int(row["display_id"])
            for row in rows
            if row["badge_center_xy"] is not None and row["badge_radius_px"] is not None
        }
        object_count += len(rows)
        visible_object_count += len(visible_ids)
        fully_occluded_objects.extend(
            {
                "scene_id": scene_id,
                "display_id": int(row["display_id"]),
                "instance_id": row["simulator_instance_id"],
            }
            for row in rows
            if not row["visible"]
        )
        if badge_ids != visible_ids:
            failures.append(
                f"{scene_id}: badge_ids={sorted(badge_ids)} visible_ids={sorted(visible_ids)}"
            )
        map_path = EXPERIMENT / f"images/id_mappings/{scene_id}_visible_ids.png"
        map_ids = set(int(value) for value in np.unique(np.asarray(Image.open(map_path)))) - {0}
        if map_ids != visible_ids:
            failures.append(
                f"{scene_id}: mask_map_ids={sorted(map_ids)} visible_ids={sorted(visible_ids)}"
            )
        object_mask_dir = EXPERIMENT / f"audit/object_visible_masks/{scene_id}"
        mask_names = {path.name for path in object_mask_dir.glob("*.png")}
        expected_mask_names = {
            f"{int(row['display_id']):03d}_{row['simulator_instance_id']}.png" for row in rows
        }
        if mask_names != expected_mask_names:
            failures.append(
                f"{scene_id}: object-mask files do not match all physical objects"
            )

        for key, geometry_file in record["geometry_files"].items():
            payload = json.loads((EXPERIMENT / geometry_file["path"]).read_text(encoding="utf-8"))
            if key.startswith("R_"):
                actual_ids = {int(item["object_id"]) for item in payload["relations"]}
                expected_ids = all_ids - {target_id}
            else:
                actual_ids = {int(item["object_id"]) for item in payload["objects"]}
                expected_ids = all_ids
            if actual_ids != expected_ids:
                failures.append(
                    f"{scene_id}/{key}: geometry_ids={sorted(actual_ids)} "
                    f"expected_ids={sorted(expected_ids)}"
                )

        if L2_CLEARANCE_MODE == "finger_placement_zone":
            gt = json.loads((EXPERIMENT / record["ground_truth"]).read_text(encoding="utf-8"))
            zone_status = gt["L2"].get("finger_placement_zone_status", {})
            expected_direct = (
                gt["L1"]["visibility"] == "FULL"
                and zone_status == {"LEFT": "CLEAR", "RIGHT": "CLEAR"}
            )
            if bool(gt["L2"]["direct_graspable"]) != expected_direct:
                failures.append(f"{scene_id}: L2 finger-zone predicate is inconsistent")
            family = record["scene_family"]
            intended = True if family == "FC_CLEAR" else False if family == "FC_BLOCKED" else None
            if intended is not None and bool(gt["L2"]["direct_graspable"]) != intended:
                design_label_mismatches.append({
                    "scene_id": scene_id,
                    "scene_family": family,
                    "manifest_intended_direct_graspable": intended,
                    "isaac_geometry_direct_graspable": bool(gt["L2"]["direct_graspable"]),
                    "finger_placement_zone_status": zone_status,
                    "minimum_surface_gap_cm": gt["L2"]["minimum_surface_gap_cm"],
                    "required_side_clearance_cm": gt["methods"]["required_side_clearance_cm"],
                })

    report = {
        "status": "pass" if not failures else "fail",
        "generated_at": utc_now(),
        "scene_count": len(records),
        "physical_object_count": object_count,
        "rgb_visible_object_count": visible_object_count,
        "fully_occluded_object_count": len(fully_occluded_objects),
        "fully_occluded_objects": fully_occluded_objects,
        "design_label_mismatches": design_label_mismatches,
        "invariants": {
            "every_rgb_visible_object_has_badge": True,
            "every_physical_object_has_saved_visible_mask": True,
            "every_physical_object_is_in_geometry": True,
            "fully_occluded_objects_are_not_badged": True,
        },
        "failures": failures,
    }
    common.write_json(EXPERIMENT / "audit/object_coverage_report.json", report)
    return report


def finalize(manifest: dict) -> None:
    records = []
    missing = []
    for scene in manifest["scenes"]:
        path = RECORDS_DIR / f"{scene['scene_id']}.json"
        if not path.exists():
            missing.append(scene["scene_id"])
        else:
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        print(json.dumps({"status": "partial", "completed": len(records), "missing": missing}, indent=2), flush=True)
        return
    order = {scene["scene_id"]: index for index, scene in enumerate(manifest["scenes"])}
    records.sort(key=lambda row: order[row["scene_id"]])
    frozen_path = EXPERIMENT / "config/frozen_scenes.json"
    common.write_json(frozen_path, records)
    config = write_static_files(manifest)
    rows = build_run_table(records, config)
    run_table_path = EXPERIMENT / "config/run_table.jsonl"
    run_table_path.parent.mkdir(parents=True, exist_ok=True)
    with run_table_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ids = [row["run_id"] for row in rows]
    counts = {
        level: sum(row["level"] == level for row in rows)
        for level in ("L1", "L2", "L3", "L4")
    }
    family_counts = {
        family: sum(scene["scene_family"] == family for scene in records)
        for family in sorted({scene["scene_family"] for scene in records})
    }
    failures = []
    if len(rows) != 6250:
        failures.append(f"run_table_rows={len(rows)} expected=6250")
    if len(set(ids)) != len(ids):
        failures.append("duplicate run_id")
    if counts != {"L1": 1500, "L2": 1500, "L3": 1500, "L4": 1750}:
        failures.append(f"level_counts={counts}")
    coverage_report = audit_object_coverage(records)
    failures.extend(coverage_report["failures"])
    report = {
        "status": "pass" if not failures else "fail",
        "generated_at": utc_now(),
        "experiment_version": common.EXPERIMENT_VERSION,
        "source_manifest_sha256": config["source_manifest_sha256"],
        "frozen_scene_count": len(records),
        "scene_family_counts": family_counts,
        "run_table_rows": len(rows),
        "unique_run_ids": len(set(ids)),
        "level_counts": counts,
        "per_scene_seed_rows": 50,
        "object_coverage_report": "audit/object_coverage_report.json",
        "object_coverage_status": coverage_report["status"],
        "failures": failures,
        "run_table_sha256": common.sha256_path(run_table_path),
        "frozen_scenes_sha256": common.sha256_path(frozen_path),
    }
    common.write_json(EXPERIMENT / "audit/preflight_report.json", report)
    hash_rows = {}
    for path in sorted(EXPERIMENT.rglob("*")):
        if path.is_file() and path.name != "file_hashes.json":
            hash_rows[str(path.relative_to(EXPERIMENT))] = common.sha256_path(path)
    common.write_json(EXPERIMENT / "audit/file_hashes.json", hash_rows)
    if failures:
        raise RuntimeError("Preflight failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(manifest["scenes"]) != 25:
        raise RuntimeError(f"Expected 25 scenes, found {len(manifest['scenes'])}")
    selected_ids = set(args.scene_id or [])
    known_ids = {scene["scene_id"] for scene in manifest["scenes"]}
    unknown_ids = sorted(selected_ids - known_ids)
    if unknown_ids:
        raise RuntimeError(f"Unknown scene IDs: {unknown_ids}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    processed = 0
    for scene in manifest["scenes"]:
        if selected_ids and scene["scene_id"] not in selected_ids:
            continue
        record_path = RECORDS_DIR / f"{scene['scene_id']}.json"
        if record_path.exists() and not args.force:
            print(f"[resume] {scene['scene_id']}", flush=True)
            continue
        if args.max_scenes is not None and processed >= args.max_scenes:
            break
        record = process_scene(scene)
        processed += 1
        print(
            f"[{processed}] {record['scene_id']} family={record['scene_family']} "
            f"visible_ids={record['visible_ids']} target={record['target_object_id']}",
            flush=True,
        )
    finalize(manifest)
    simulation_app.close()


if __name__ == "__main__":
    main()
