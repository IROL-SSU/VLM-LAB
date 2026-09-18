#!/usr/bin/env python3
"""Pure helpers shared by the VLM action/geometry v1 experiment tools."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_VERSION = "vlm_action_geometry_single_info_v1"
SEEDS = (28101, 28102, 28103, 28104, 28105)
RUN_ORDER_SEED = 20260915
GLOBAL_ID_SEED = 20260915

DIRECTIONS = {
    "D4": ("RIGHT", "BACK", "LEFT", "FRONT"),
    "D8": (
        "RIGHT",
        "BACK_RIGHT",
        "BACK",
        "BACK_LEFT",
        "LEFT",
        "FRONT_LEFT",
        "FRONT",
        "FRONT_RIGHT",
    ),
}

DIRECTION_ANGLES = {
    "RIGHT": 0.0,
    "BACK_RIGHT": 45.0,
    "BACK": 90.0,
    "BACK_LEFT": 135.0,
    "LEFT": 180.0,
    "FRONT_LEFT": 225.0,
    "FRONT": 270.0,
    "FRONT_RIGHT": 315.0,
}

SHELF_BOUNDS_CM = (-18.0, 74.0, -16.0, 16.0)  # xmin, xmax, ymin, ymax


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_sparse_ids(scene_id: str, ordered_instance_ids: Sequence[str]) -> tuple[list[int], int, float]:
    material = (
        f"{GLOBAL_ID_SEED}|sparse-v1|{EXPERIMENT_VERSION}|"
        f"{scene_id}|{len(ordered_instance_ids)}"
    )
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    pool = list(range(10, 100))
    for _ in range(10000):
        rng.shuffle(pool)
        candidate = pool[: len(ordered_instance_ids)]
        if any(
            abs(first - second) <= 1
            for index, first in enumerate(candidate)
            for second in candidate[index + 1 :]
        ):
            continue
        correlation = spearman_identity(candidate)
        if len(candidate) <= 2 or abs(correlation) <= 0.45:
            return list(candidate), seed, correlation
    raise RuntimeError(f"Could not generate sparse display IDs for {scene_id}")


def spearman_identity(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    ranks = {
        value: rank
        for rank, value in enumerate(sorted(values))
    }
    xs = [float(index) for index in range(len(values))]
    ys = [float(ranks[value]) for value in values]
    xmean = sum(xs) / len(xs)
    ymean = sum(ys) / len(ys)
    numerator = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys))
    xscale = math.sqrt(sum((x - xmean) ** 2 for x in xs))
    yscale = math.sqrt(sum((y - ymean) ** 2 for y in ys))
    return numerator / (xscale * yscale) if xscale and yscale else 0.0


def direction_vector(name: str) -> tuple[float, float]:
    angle = math.radians(DIRECTION_ANGLES[name])
    return math.cos(angle), math.sin(angle)


def nearest_direction(angle_deg: float, resolution: str) -> str:
    return min(
        DIRECTIONS[resolution],
        key=lambda name: angular_distance(angle_deg, DIRECTION_ANGLES[name]),
    )


def angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def translation_level(distance_cm: float) -> str:
    if distance_cm <= 0:
        return "NONE"
    if distance_cm <= 5:
        return "SHORT"
    if distance_cm <= 10:
        return "MEDIUM"
    return "LONG"


def free_space_level(distance_cm: float) -> str:
    if distance_cm <= 0:
        return "OCCUPIED"
    if distance_cm <= 5:
        return "TIGHT"
    if distance_cm <= 10:
        return "LIMITED"
    return "OPEN"


def rotation_level(angle_deg: float) -> str:
    if angle_deg <= 0:
        return "NONE"
    if angle_deg <= 15:
        return "SMALL"
    if angle_deg <= 45:
        return "MEDIUM"
    return "LARGE"


Point = tuple[float, float]
Polygon = list[Point]


def cross(origin: Point, first: Point, second: Point) -> float:
    return (
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def convex_hull(points: Iterable[Point]) -> Polygon:
    unique = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(unique) <= 2:
        return list(unique)
    lower: Polygon = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-10:
            lower.pop()
        lower.append(point)
    upper: Polygon = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-10:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_centroid(polygon: Sequence[Point]) -> Point:
    if not polygon:
        raise ValueError("empty polygon")
    if len(polygon) < 3:
        return (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
    signed_area = 0.0
    xsum = 0.0
    ysum = 0.0
    for first, second in polygon_edges(polygon):
        term = first[0] * second[1] - second[0] * first[1]
        signed_area += term
        xsum += (first[0] + second[0]) * term
        ysum += (first[1] + second[1]) * term
    if abs(signed_area) < 1e-12:
        return (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
    return xsum / (3.0 * signed_area), ysum / (3.0 * signed_area)


def polygon_edges(polygon: Sequence[Point]):
    for index, point in enumerate(polygon):
        yield point, polygon[(index + 1) % len(polygon)]


def rotate_polygon(polygon: Sequence[Point], angle_deg: float, center: Point | None = None) -> Polygon:
    center = center or polygon_centroid(polygon)
    cosine = math.cos(math.radians(angle_deg))
    sine = math.sin(math.radians(angle_deg))
    output = []
    for x, y in polygon:
        dx, dy = x - center[0], y - center[1]
        output.append(
            (
                center[0] + cosine * dx - sine * dy,
                center[1] + sine * dx + cosine * dy,
            )
        )
    return output


def translate_polygon(polygon: Sequence[Point], dx: float, dy: float) -> Polygon:
    return [(x + dx, y + dy) for x, y in polygon]


def project(polygon: Sequence[Point], axis: Point) -> tuple[float, float]:
    values = [x * axis[0] + y * axis[1] for x, y in polygon]
    return min(values), max(values)


def polygons_intersect(first: Sequence[Point], second: Sequence[Point], tolerance: float = 1e-7) -> bool:
    if len(first) < 3 or len(second) < 3:
        return polygon_distance(first, second) <= tolerance
    for polygon in (first, second):
        for start, end in polygon_edges(polygon):
            axis = (-(end[1] - start[1]), end[0] - start[0])
            amin, amax = project(first, axis)
            bmin, bmax = project(second, axis)
            if amax < bmin - tolerance or bmax < amin - tolerance:
                return False
    return True


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    scale = dx * dx + dy * dy
    if scale <= 1e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / scale),
    )
    nearest = (start[0] + ratio * dx, start[1] + ratio * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def segment_distance(first_start: Point, first_end: Point, second_start: Point, second_end: Point) -> float:
    if segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        point_segment_distance(first_start, second_start, second_end),
        point_segment_distance(first_end, second_start, second_end),
        point_segment_distance(second_start, first_start, first_end),
        point_segment_distance(second_end, first_start, first_end),
    )


def orientation(first: Point, second: Point, third: Point) -> float:
    return cross(first, second, third)


def on_segment(first: Point, point: Point, second: Point, tolerance: float = 1e-9) -> bool:
    return (
        min(first[0], second[0]) - tolerance <= point[0] <= max(first[0], second[0]) + tolerance
        and min(first[1], second[1]) - tolerance <= point[1] <= max(first[1], second[1]) + tolerance
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    if (first > 0 > second or second > 0 > first) and (third > 0 > fourth or fourth > 0 > third):
        return True
    tolerance = 1e-9
    return (
        (abs(first) <= tolerance and on_segment(a, c, b))
        or (abs(second) <= tolerance and on_segment(a, d, b))
        or (abs(third) <= tolerance and on_segment(c, a, d))
        or (abs(fourth) <= tolerance and on_segment(c, b, d))
    )


def polygon_distance(first: Sequence[Point], second: Sequence[Point]) -> float:
    if not first or not second:
        raise ValueError("polygon distance requires two non-empty polygons")
    if len(first) >= 3 and len(second) >= 3 and polygons_intersect_sat(first, second):
        return 0.0
    return min(
        segment_distance(a, b, c, d)
        for a, b in polygon_edges(first)
        for c, d in polygon_edges(second)
    )


def polygons_intersect_sat(first: Sequence[Point], second: Sequence[Point]) -> bool:
    for polygon in (first, second):
        for start, end in polygon_edges(polygon):
            axis = (-(end[1] - start[1]), end[0] - start[0])
            amin, amax = project(first, axis)
            bmin, bmax = project(second, axis)
            if amax < bmin - 1e-9 or bmax < amin - 1e-9:
                return False
    return True


def polygon_inside_bounds(polygon: Sequence[Point], bounds: tuple[float, float, float, float] = SHELF_BOUNDS_CM) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in polygon)


def support_distance(polygon: Sequence[Point], direction: Point, center: Point | None = None) -> float:
    center = center or polygon_centroid(polygon)
    return max((x - center[0]) * direction[0] + (y - center[1]) * direction[1] for x, y in polygon)


def ray_segment_parameter(origin: Point, direction: Point, start: Point, end: Point) -> float | None:
    segment = (end[0] - start[0], end[1] - start[1])
    denominator = direction[0] * segment[1] - direction[1] * segment[0]
    if abs(denominator) < 1e-12:
        return None
    offset = (start[0] - origin[0], start[1] - origin[1])
    ray_t = (offset[0] * segment[1] - offset[1] * segment[0]) / denominator
    segment_t = (offset[0] * direction[1] - offset[1] * direction[0]) / denominator
    if ray_t >= 0 and -1e-9 <= segment_t <= 1 + 1e-9:
        return ray_t
    return None


def ray_polygon_first_hit(origin: Point, direction: Point, polygon: Sequence[Point]) -> float | None:
    values = [
        value
        for start, end in polygon_edges(polygon)
        if (value := ray_segment_parameter(origin, direction, start, end)) is not None
    ]
    return min(values) if values else None


def boundary_ray_distance(origin: Point, direction: Point, bounds=SHELF_BOUNDS_CM) -> float:
    xmin, xmax, ymin, ymax = bounds
    rectangle = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    hits = [
        value
        for start, end in polygon_edges(rectangle)
        if (value := ray_segment_parameter(origin, direction, start, end)) is not None
    ]
    if not hits:
        return 0.0
    return min(value for value in hits if value >= 0)


def directional_free_space(
    selected: Sequence[Point],
    others: Sequence[Sequence[Point]],
    direction_name: str,
) -> float:
    direction = direction_vector(direction_name)
    center = polygon_centroid(selected)
    own_extent = support_distance(selected, direction, center)
    candidates = [boundary_ray_distance(center, direction) - own_extent]
    for other in others:
        hit = ray_polygon_first_hit(center, direction, other)
        if hit is not None:
            candidates.append(hit - own_extent)
    return max(0.0, min(candidates))


def motion_capacity_translation(
    selected: Sequence[Point],
    others: Sequence[Sequence[Point]],
    direction_name: str,
    step_cm: float = 0.25,
    maximum_cm: float = 100.0,
) -> float:
    dx, dy = direction_vector(direction_name)
    safe = 0.0
    step_count = int(round(maximum_cm / step_cm))
    for index in range(1, step_count + 1):
        distance = index * step_cm
        moved = translate_polygon(selected, dx * distance, dy * distance)
        if not polygon_inside_bounds(moved) or any(polygons_intersect(moved, other) for other in others):
            break
        safe = distance
    return safe


def motion_capacity_rotation(
    selected: Sequence[Point],
    others: Sequence[Sequence[Point]],
    sign: float,
    step_deg: int = 5,
    maximum_deg: int = 180,
) -> int:
    safe = 0
    for angle in range(step_deg, maximum_deg + step_deg, step_deg):
        moved = rotate_polygon(selected, sign * angle)
        if not polygon_inside_bounds(moved) or any(polygons_intersect(moved, other) for other in others):
            break
        safe = angle
    return safe


def occupied_angular_arc(target_center: Point, polygon: Sequence[Point]) -> list[list[float]]:
    angles = sorted(
        math.degrees(math.atan2(y - target_center[1], x - target_center[0])) % 360.0
        for x, y in polygon
    )
    if not angles:
        return []
    if len(angles) == 1:
        value = round(angles[0], 1)
        return [[value, value]]
    gaps = [
        ((angles[(index + 1) % len(angles)] - angles[index]) % 360.0, index)
        for index in range(len(angles))
    ]
    _, gap_index = max(gaps)
    start = angles[(gap_index + 1) % len(angles)]
    end = angles[gap_index]
    if start <= end:
        return [[round(start, 1), round(end, 1)]]
    return [[round(start, 1), 360.0], [0.0, round(end, 1)]]


def angle_in_arcs(angle: float, arcs: Sequence[Sequence[float]], padding: float) -> bool:
    for start, end in arcs:
        if start - padding <= angle <= end + padding:
            return True
        if start - padding < 0 and angle >= 360 + start - padding:
            return True
        if end + padding > 360 and angle <= end + padding - 360:
            return True
    return False


def occupied_directions(arcs: Sequence[Sequence[float]], resolution: str) -> list[str]:
    half_width = 45.0 if resolution == "D4" else 22.5
    return [
        name
        for name in DIRECTIONS[resolution]
        if angle_in_arcs(DIRECTION_ANGLES[name], arcs, half_width)
    ]


def scene_family(scene_id: str) -> str:
    if scene_id.startswith("scene_translate_"):
        return "TRANSLATE"
    if scene_id.startswith("scene_rotate_"):
        return "ROTATE"
    if scene_id.startswith("scene_lift_and_relocate_"):
        return "LIFT_AND_RELOCATE"
    if scene_id.startswith("scene_fc_clear_"):
        return "FC_CLEAR"
    if scene_id.startswith("scene_fc_blocked_"):
        return "FC_BLOCKED"
    raise ValueError(f"Unknown scene family: {scene_id}")

