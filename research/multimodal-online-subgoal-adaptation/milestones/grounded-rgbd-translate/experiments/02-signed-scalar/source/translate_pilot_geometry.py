"""Static shelf-plane geometry for the TRANSLATE RGB-D pilot (all units m).

Labels clear a target-front corridor using conservative world-AABB footprints.
They are not expert labels, robot feasibility, or collision-mesh simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


DIRECTIONS = {
    "BACK": (0.0, 1.0),
    "BACK_RIGHT": (math.sqrt(0.5), math.sqrt(0.5)),
    "RIGHT": (1.0, 0.0),
    "FRONT_RIGHT": (math.sqrt(0.5), -math.sqrt(0.5)),
    "FRONT": (0.0, -1.0),
    "FRONT_LEFT": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "LEFT": (-1.0, 0.0),
    "BACK_LEFT": (-math.sqrt(0.5), math.sqrt(0.5)),
}
BOUNDS = (-0.18, -0.16, 0.74, 0.16)
CORRIDOR_MARGIN = 0.005
OBJECT_GAP = 0.002
STEP = 0.001


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def centered(cls, x, y, width, depth):
        return cls(x - width / 2, y - depth / 2, x + width / 2, y + depth / 2)

    @classmethod
    def from_list(cls, values):
        return cls(*values)

    def as_list(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def moved(self, dx, dy):
        return Rect(self.xmin + dx, self.ymin + dy, self.xmax + dx, self.ymax + dy)

    def expanded(self, margin):
        return Rect(self.xmin - margin, self.ymin - margin, self.xmax + margin, self.ymax + margin)

    def overlap(self, other):
        return max(0.0, min(self.xmax, other.xmax) - max(self.xmin, other.xmin)) * max(
            0.0, min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        )

    def inside(self, bounds=BOUNDS, gap=OBJECT_GAP):
        return (
            self.xmin >= bounds[0] + gap and self.ymin >= bounds[1] + gap
            and self.xmax <= bounds[2] - gap and self.ymax <= bounds[3] - gap
        )


def target_corridor(target, margin=CORRIDOR_MARGIN):
    return Rect(target.xmin - margin, BOUNDS[1], target.xmax + margin, target.ymin)


def clearance_limit(moving, direction, obstacles, max_distance=0.45, gap=OBJECT_GAP):
    """First contact of a translating rectangle; continuous swept-AABB test."""
    vx, vy = direction
    limit = max_distance
    for lo, hi, vlo, vhi, speed in (
        (moving.xmin, moving.xmax, BOUNDS[0], BOUNDS[2], vx),
        (moving.ymin, moving.ymax, BOUNDS[1], BOUNDS[3], vy),
    ):
        if speed > 0:
            limit = min(limit, (vhi - gap - hi) / speed)
        elif speed < 0:
            limit = min(limit, (vlo + gap - lo) / speed)
    for obstacle in obstacles:
        expanded = obstacle.expanded(gap)
        entry, leave = -math.inf, math.inf
        misses = False
        for alo, ahi, blo, bhi, speed in (
            (moving.xmin, moving.xmax, expanded.xmin, expanded.xmax, vx),
            (moving.ymin, moving.ymax, expanded.ymin, expanded.ymax, vy),
        ):
            if abs(speed) < 1e-12:
                if ahi <= blo or alo >= bhi:
                    misses = True
                    break
            else:
                t1, t2 = (blo - ahi) / speed, (bhi - alo) / speed
                entry, leave = max(entry, min(t1, t2)), min(leave, max(t1, t2))
        if not misses and leave > max(entry, 0.0):
            limit = min(limit, max(entry, 0.0))
    return max(0.0, limit)


def translation_candidates(target, blocker, distractors):
    corridor = target_corridor(target)
    if blocker.overlap(corridor) <= 1e-8:
        return []
    if any(other.overlap(corridor) > 1e-8 for other in distractors):
        return []
    results = []
    for name, vector in DIRECTIONS.items():
        limit = clearance_limit(blocker, vector, [target, *distractors])
        for index in range(1, int(math.floor((limit - 1e-8) / STEP)) + 1):
            distance = index * STEP
            dx, dy = vector[0] * distance, vector[1] * distance
            goal = blocker.moved(dx, dy)
            if goal.overlap(corridor) <= 1e-10:
                results.append({
                    "direction": name,
                    "direction_vector_shelf": [vector[0], vector[1], 0.0],
                    "distance_m": round(distance, 6),
                    "delta_shelf_m": [dx, dy, 0.0],
                    "maximum_clear_translation_m": limit,
                    "goal_footprint_m": goal.as_list(),
                })
                break
    return sorted(results, key=lambda row: (row["distance_m"], row["direction"]))


def sample_layout(rng, dimensions, target_asset, blocker_asset, distractor_assets, max_tries=10000):
    """Constrained random layouts: one corridor blocker and collision-free objects.

    dimensions maps asset -> (world-AABB width, depth, height) at sampled yaw.
    Object identities/yaws are sampled by the caller, positions here.
    """
    tw, td, _ = dimensions[target_asset]
    bw, bd, _ = dimensions[blocker_asset]
    for attempt in range(max_tries):
        tx = rng.uniform(BOUNDS[0] + tw / 2 + 0.08, BOUNDS[2] - tw / 2 - 0.08)
        ty = rng.uniform(0.03, BOUNDS[3] - td / 2 - 0.006)
        target = Rect.centered(tx, ty, tw, td)
        by = ty - td / 2 - bd / 2 - rng.uniform(0.005, 0.025)
        sign = rng.choice((-1, 1))
        bx = tx + sign * rng.uniform(0.12, 0.82) * (tw + bw) / 2
        blocker = Rect.centered(bx, by, bw, bd)
        if not target.inside() or not blocker.inside():
            continue
        corridor = target_corridor(target)
        placed = [target, blocker]
        centers = {target_asset: (tx, ty), blocker_asset: (bx, by)}
        for name in distractor_assets:
            width, depth, _ = dimensions[name]
            for _ in range(250):
                x = rng.uniform(BOUNDS[0] + width / 2 + 0.006, BOUNDS[2] - width / 2 - 0.006)
                y = rng.uniform(BOUNDS[1] + depth / 2 + 0.006, BOUNDS[3] - depth / 2 - 0.006)
                rect = Rect.centered(x, y, width, depth)
                if rect.overlap(corridor) > 1e-10:
                    continue
                if any(rect.overlap(other.expanded(OBJECT_GAP + 0.001)) > 1e-10 for other in placed):
                    continue
                placed.append(rect)
                centers[name] = (x, y)
                break
            else:
                break
        if len(placed) != 2 + len(distractor_assets):
            continue
        candidates = translation_candidates(target, blocker, placed[2:])
        if candidates and 0.02 <= candidates[0]["distance_m"] <= 0.25:
            return {
                "centers": centers,
                "footprints": {name: rect.as_list() for name, rect in zip(
                    [target_asset, blocker_asset, *distractor_assets], placed
                )},
                "corridor_footprint_m": corridor.as_list(),
                "candidates": candidates,
                "attempts": attempt + 1,
            }
    raise RuntimeError("Could not sample a valid translation layout for the selected assets")
