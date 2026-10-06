#!/usr/bin/env python3
"""
Shared helpers for building shelf scenes out of the real (non-procedural)
USD assets in Final_USD/ratio_0.8/scale_down_mug_cup(2)/scale_down_mug_cup,
used by all 5 generate_levelN_*.py benchmark scripts.

Replaces the old uniform-cylinder mugs (MUG_BODY_R=2.6, MUG_BODY_H=9.0 for
every object in every scene) that prompted the "all objects are the same
height" feedback -- these are real objects with genuinely varied height
(~5.6cm to ~18.5cm) and footprint.

IMPORTANT: must be imported AFTER `SimulationApp(...)` has been
constructed in the calling script (same requirement as `pxr`/`omni.usd` --
those modules aren't importable until the Kit app is up).

Scale note: every asset here except cup_7 is authored at metersPerUnit=1.0
(meters) while our stage uses metersPerUnit=0.01 (cm); referencing does not
auto-convert, so each needs a 100x scale except cup_7 (already cm-native).
Confirmed via a metersPerUnit probe before this was written.
"""

import random

from pxr import Gf, Usd, UsdGeom

BASE = "/home/ssu/ShelfScene/Final_USD/ratio_0.8/scale_down_mug_cup(2)/scale_down_mug_cup"

# name -> (usd path, scale, category)
OBJECTS = {
    "mug_7":       (f"{BASE}/mug_7/Collected_Mug_7/Mug_7.usd",               100.0, "mugs"),
    "Mug_2":       (f"{BASE}/Collected_Mug_2/Mug_2.usd",                     100.0, "mugs"),
    "Mug_4":       (f"{BASE}/Collected_Mug_4/Mug_4.usd",                     100.0, "mugs"),
    "cup_7":       (f"{BASE}/cup_7/Collected_cup_7/cup_7.usd",                 1.0, "cups"),
    "cup_8":       (f"{BASE}/cup_8/Collected_cup_8/cup_8.usd",               100.0, "cups"),
    "Cup_2":       (f"{BASE}/Collected_Cup_2/Cup_2.usd",                     100.0, "cups"),
    "Cup_4":       (f"{BASE}/Collected_Cup_4/Cup_4.usd",                     100.0, "cups"),
    "acafela":     (f"{BASE}/acafela/Collected_acafela/acafela.usd",         100.0, "drinks"),
    "cantata":     (f"{BASE}/cantata/Collected_cantata/cantata.usd",         100.0, "drinks"),
    "coldgrape":   (f"{BASE}/coldgrape/Collected_grapejuice/grapejuice.usd", 100.0, "drinks"),
    "top":         (f"{BASE}/top/Collected_top/top.usd",                    100.0, "drinks"),
    "cocopalm":    (f"{BASE}/cocopalm/Collected_cocopalm/cocopalm.usd",     100.0, "drinks"),
    "biracsikhye": (f"{BASE}/biracsikhye/Collected_sikhye/sikhye.usd",      100.0, "drinks"),
    "letsbe":      (f"{BASE}/letsbe/Collected_letsbe/letsbe.usd",           100.0, "drinks"),
    "minutemad":   (f"{BASE}/minutemad/Collected_yuzujuice/yuzujuice.usd",  100.0, "drinks"),
}

ALL_NAMES = list(OBJECTS.keys())

_bbox_cache = None


def _get_bbox_cache():
    global _bbox_cache
    if _bbox_cache is None:
        _bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
    return _bbox_cache


def pick_role_objects(n, rng=None, pool=None):
    """Randomly pick n distinct object names (no repeats within one scene)."""
    rng = rng or random
    pool = pool or ALL_NAMES
    return rng.sample(pool, n)


def reference_and_place(stage, parent_path, name, x, y, yaw_deg, top_surface_z):
    """Reference OBJECTS[name] under {parent_path}/{name}, scale-correct it,
    and place it so its bbox base sits exactly on the shelf floor at (x, y)
    with the given yaw. Returns (radius_cm, size_xyz_cm) measured from the
    ACTUAL scaled bbox -- callers use `radius` for spacing/gap math instead
    of a fixed constant, since every object here has a different footprint.

    Mirrors the reference-wrapping pattern from
    generate_clustered_real_objects_multiview.py: the referenced layer
    already authors its own translate/rotateXYZ/scale ops on its
    defaultPrim, so our own placement ops must live on a separate parent
    wrapper, not directly on the prim holding the reference.
    """
    path, scale, _category = OBJECTS[name]
    wrapper_path = f"{parent_path}/{name}"
    xf = UsdGeom.Xform.Define(stage, wrapper_path)
    child = UsdGeom.Xform.Define(stage, f"{wrapper_path}/asset")
    child.GetPrim().GetReferences().AddReference(path)

    translate_op = xf.AddTranslateOp()
    rotate_op = xf.AddRotateZOp()
    scale_op = xf.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale, scale, scale))

    bbox_cache = _get_bbox_cache()
    bbox_cache.Clear()
    box = bbox_cache.ComputeWorldBound(xf.GetPrim())
    rng_box = box.ComputeAlignedRange()
    lo, hi = rng_box.GetMin(), rng_box.GetMax()
    size_x, size_y, size_z = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    radius = max(size_x, size_y) / 2.0

    translate_z = top_surface_z - lo[2]
    translate_op.Set(Gf.Vec3d(x, y, translate_z))
    rotate_op.Set(yaw_deg)

    return radius, (size_x, size_y, size_z), translate_op


def reposition_xy(translate_op, x, y):
    """Update an already-placed object's (x, y) only, keeping its
    already-computed floor-aligned Z -- for when the correct offset
    depends on a radius only known AFTER the first placement (e.g.
    blocker offset computed from its own measured radius)."""
    current = translate_op.Get()
    translate_op.Set(Gf.Vec3d(x, y, current[2]))
