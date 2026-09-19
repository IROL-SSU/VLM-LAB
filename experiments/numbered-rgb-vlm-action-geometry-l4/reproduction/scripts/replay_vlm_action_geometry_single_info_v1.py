#!/usr/bin/env python3
"""Replay every valid L4 prediction directly in Isaac Sim and log outcomes.

The source USD is reopened once per scene. Each action is applied to the USD
stage in deterministic substeps, then the target is re-rendered with Isaac
Sim's semantic annotator. Exact target-only pixels and the 5 cm surface-gap
rule determine post-action direct graspability.

Run with:
  /home/ssu/isaacsim/python.sh scripts/replay_vlm_action_geometry_single_info_v1.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--experiment", type=Path,
    default=Path("experiments/vlm_action_geometry_single_info_v1"),
)
parser.add_argument("--allow-partial", action="store_true")
parser.add_argument("--max-unique-commands", type=int)
parser.add_argument("--max-attempts", type=int, default=3)
parser.add_argument("--settle-ticks", type=int, default=16)
parser.add_argument(
    "--refresh-valid-cache", action="store_true",
    help="Re-execute all semantically valid commands and append corrected latest records.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 960})
print("Replay Python runtime initialized", flush=True)

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels, upgrade_prim_semantics_to_labels
from PIL import Image
from pxr import Gf, Usd, UsdGeom

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = args.experiment if args.experiment.is_absolute() else ROOT / args.experiment
RUNS_PATH = EXPERIMENT / "logs/runs.jsonl"
REPLAY_PATH = EXPERIMENT / "logs/replay.jsonl"
ATTEMPTS_PATH = EXPERIMENT / "logs/replay_attempts.jsonl"
CACHE_DIR = EXPERIMENT / "simulator_replay/cache"
MASK_DIR = EXPERIMENT / "simulator_replay/post_masks"
PROGRESS_PATH = EXPERIMENT / "logs/replay_progress.json"
REPLAY_ENGINE_VERSION = "isaac_stage_semantic_replay_v1"
ISAAC_SIM_VERSION = (
    Path("/home/ssu/isaacsim/VERSION").read_text(encoding="utf-8").strip()
    if Path("/home/ssu/isaacsim/VERSION").exists() else "unknown"
)
CAMERA_DEFAULT = "/World/Cameras/ActionSceneCam"
TRANSLATION_SUBSTEP_CM = 0.25
ROTATION_SUBSTEP_DEG = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def latest_by_run_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["run_id"]): row for row in rows if row.get("run_id")}


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


def semantic_label_matches(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(semantic_label_matches(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(semantic_label_matches(item, expected) for item in value)
    return expected in str(value).split(",")


def capture_target_mask(annotator, target_label: str) -> np.ndarray:
    payload = None
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        for _ in range(2):
            simulation_app.update()
        candidate = annotator.get_data()
        data = np.asarray(candidate.get("data") if isinstance(candidate, dict) else candidate)
        if data.size:
            payload = candidate
            break
    if payload is None:
        raise RuntimeError("Semantic annotator produced no non-empty frame")
    segmentation = np.asarray(payload["data"])
    if segmentation.ndim > 2:
        segmentation = np.squeeze(segmentation)
    matching = [
        int(semantic_id)
        for semantic_id, values in payload.get("info", {}).get("idToLabels", {}).items()
        if semantic_label_matches(values, target_label)
    ]
    return np.isin(segmentation, matching)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def polygon(row: dict) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in row["footprint_xy_cm"]]


def translate_value(original, dx: float, dy: float):
    values = (float(original[0]) + dx, float(original[1]) + dy, float(original[2]))
    if isinstance(original, Gf.Vec3f):
        return Gf.Vec3f(*values)
    return Gf.Vec3d(*values)


def find_xform_op(prim, op_type):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == op_type and not op.IsInverseOp():
            return op
    return None


def reset_scene(state: dict) -> None:
    for item in state.values():
        if item["translate_op"] is not None:
            item["translate_op"].Set(item["translate_original"])
        if item["rotate_op"] is not None:
            item["rotate_op"].Set(item["rotate_original"])
        item["visibility_attr"].Set(item["visibility_original"])
    for _ in range(3):
        simulation_app.update()


def cache_key(record: dict) -> str:
    material = {
        "engine": REPLAY_ENGINE_VERSION,
        "scene_id": record["scene_id"],
        "source_usd_sha256": record["source_usd_sha256"],
        "command": record["parsed_response"],
    }
    return common.sha256_text(common.canonical_json(material))


def replay_stub(record: dict, *, reason: str) -> dict:
    return {
        "run_id": record["run_id"],
        "scene_id": record["scene_id"],
        "level": "L4",
        "geometry_condition": record["geometry_condition"],
        "direction_resolution": record["direction_resolution"],
        "repeat_index": record["repeat_index"],
        "seed": record["seed"],
        "model_response": record.get("parsed_response"),
        "inference_post_validation_pass": bool(record.get("post_validation_pass")),
        "simulator_executed": False,
        "cache_key": None,
        "cache_artifact": None,
        "collision": None,
        "post_visibility": None,
        "post_minimum_surface_gap_cm": None,
        "post_direct_graspable": None,
        "action_valid": False,
        "failure_reason": reason,
        "finished_at": utc_now(),
    }


def command_polygons(scene_geometry: dict, selected_id: int, command: dict, fraction: float):
    by_display = {int(item["display_id"]): item for item in scene_geometry["objects"]}
    selected = polygon(by_display[selected_id])
    action = command["action"]
    if action == "TRANSLATE":
        vx, vy = common.direction_vector(command["direction"])
        distance = float(command["distance_cm"]) * fraction
        selected = common.translate_polygon(selected, vx * distance, vy * distance)
    elif action == "ROTATE":
        sign = -1.0 if command["rotation_direction"] == "CW" else 1.0
        selected = common.rotate_polygon(selected, sign * float(command["angle_deg"]) * fraction)
    return selected, by_display


def apply_stage_fraction(selected: dict, command: dict, fraction: float) -> None:
    action = command["action"]
    if action == "TRANSLATE":
        op = selected["translate_op"]
        if op is None:
            raise RuntimeError("Selected prim has no translation op")
        distance = float(command["distance_cm"]) * fraction
        vx, vy = common.direction_vector(command["direction"])
        op.Set(translate_value(selected["translate_original"], vx * distance, vy * distance))
    elif action == "ROTATE":
        op = selected["rotate_op"]
        if op is None:
            raise RuntimeError("Selected prim has no rotate-Z op")
        angle = float(command["angle_deg"]) * fraction
        sign = -1.0 if command["rotation_direction"] == "CW" else 1.0
        op.Set(float(selected["rotate_original"]) + sign * angle)


def path_check(scene_geometry: dict, state: dict, selected_id: int, command: dict) -> dict:
    """Author and check every Isaac-stage kinematic substep."""
    action = command["action"]
    if action not in {"TRANSLATE", "ROTATE"}:
        return {
            "collision": False,
            "boundary_violation": False,
            "first_failure_step": None,
            "steps_requested": 1,
            "steps_executed": 1,
            "collision_object_ids": [],
            "stage_substeps_authored": 0,
        }
    amount = float(command["distance_cm"] if action == "TRANSLATE" else command["angle_deg"])
    substep = TRANSLATION_SUBSTEP_CM if action == "TRANSLATE" else ROTATION_SUBSTEP_DEG
    # A rotation path repeats after 360 degrees; one full cycle is sufficient.
    checked_amount = min(amount, 360.0) if action == "ROTATE" else amount
    requested_steps = max(1, int(math.ceil(checked_amount / substep)))
    by_display = {int(item["display_id"]): item for item in scene_geometry["objects"]}
    collision = False
    boundary = False
    collision_ids: list[int] = []
    executed = 0
    failure_step = None
    terminal_polygon = polygon(by_display[selected_id])
    for step in range(1, requested_steps + 1):
        current_amount = min(step * substep, checked_amount)
        fraction = current_amount / amount if amount else 1.0
        apply_stage_fraction(state[selected_id], command, fraction)
        simulation_app.update()
        terminal_polygon, _ = command_polygons(scene_geometry, selected_id, command, fraction)
        executed = step
        boundary = not common.polygon_inside_bounds(terminal_polygon)
        collision_ids = sorted(
            other_id for other_id, other in by_display.items()
            if other_id != selected_id and common.polygons_intersect(terminal_polygon, polygon(other))
        )
        collision = bool(collision_ids)
        if collision or boundary:
            failure_step = step
            break
    terminal_fraction = min(executed * substep, checked_amount) / amount if amount else 1.0
    return {
        "collision": collision,
        "boundary_violation": boundary,
        "first_failure_step": failure_step,
        "steps_requested": requested_steps,
        "steps_executed": executed,
        "checked_amount": checked_amount,
        "requested_amount": amount,
        "terminal_fraction": terminal_fraction,
        "collision_object_ids": collision_ids,
        "stage_substeps_authored": executed,
        "terminal_footprint_xy_cm": [[round(x, 6), round(y, 6)] for x, y in terminal_polygon],
    }


def set_terminal_stage_pose(state: dict, selected_id: int, command: dict, path: dict) -> None:
    selected = state[selected_id]
    action = command["action"]
    if action in {"LIFT_AND_RELOCATE", "RETRIEVE"}:
        selected["visibility_attr"].Set(UsdGeom.Tokens.invisible)
    for _ in range(4):
        simulation_app.update()


def post_geometry(scene_geometry: dict, selected_id: int, command: dict, path: dict) -> dict:
    by_display = {int(item["display_id"]): item for item in scene_geometry["objects"]}
    target_id = next(
        object_id for object_id, item in by_display.items()
        if item["instance_id"] == "target_teal"
    )
    target_polygon = polygon(by_display[target_id])
    present = {object_id: polygon(item) for object_id, item in by_display.items()}
    if command["action"] in {"LIFT_AND_RELOCATE", "RETRIEVE"}:
        present.pop(selected_id, None)
    elif command["action"] in {"TRANSLATE", "ROTATE"}:
        present[selected_id] = [tuple(value) for value in path["terminal_footprint_xy_cm"]]
    gaps = {
        str(object_id): common.polygon_distance(target_polygon, other_polygon)
        for object_id, other_polygon in present.items() if object_id != target_id
    }
    minimum = min(gaps.values()) if gaps else None
    return {
        "surface_gaps_cm": {key: round(value, 6) for key, value in gaps.items()},
        "minimum_surface_gap_cm": round(minimum, 6) if minimum is not None else None,
        "clearance_pass": minimum is None or minimum > 5.0,
    }


def execute_command(
    record: dict, scene_geometry: dict, mapping: dict, state: dict,
    annotator, target_label: str, target_only_mask: np.ndarray, baseline_matches: bool,
) -> dict:
    command = record["parsed_response"]
    selected_id = int(command["object_id"])
    if selected_id not in state:
        raise RuntimeError(f"Unknown object display ID {selected_id}")
    path = path_check(scene_geometry, state, selected_id, command)
    set_terminal_stage_pose(state, selected_id, command, path)
    post_mask = capture_target_mask(annotator, target_label)
    post_pixels = int(post_mask.sum())
    action = command["action"]
    if action == "RETRIEVE":
        post_visibility = "REMOVED" if post_pixels == 0 else "STILL_VISIBLE"
        post_full = False
    elif post_pixels == 0:
        post_visibility = "NOT_VISIBLE"
        post_full = False
    elif np.array_equal(post_mask, target_only_mask):
        post_visibility = "FULL"
        post_full = True
    else:
        post_visibility = "PARTIAL"
        post_full = False
    geometry = post_geometry(scene_geometry, selected_id, command, path)
    gt = read_json(EXPERIMENT / record["ground_truth"])
    if action == "RETRIEVE":
        retrieval_precondition = bool(gt["L2"]["direct_graspable"])
        action_valid = (
            retrieval_precondition and post_visibility == "REMOVED"
            and not path["collision"] and not path["boundary_violation"]
        )
        post_direct = None
    else:
        retrieval_precondition = None
        post_direct = post_full and geometry["clearance_pass"]
        action_valid = (
            not path["collision"] and not path["boundary_violation"] and post_direct
        )
    key = cache_key(record)
    mask_path = MASK_DIR / f"{key}.png"
    save_mask(mask_path, post_mask)
    artifact = {
        "replay_engine_version": REPLAY_ENGINE_VERSION,
        "cache_key": key,
        "scene_id": record["scene_id"],
        "source_usd": record["source_usd"],
        "source_usd_sha256": record["source_usd_sha256"],
        "command": command,
        "selected_object_id": selected_id,
        "selected_prim_path": state[selected_id]["prim_path"],
        "direct_stage_execution": True,
        "execution_mode": "USD stage transform/visibility with deterministic kinematic substeps",
        "contact_oracle": "visual convex-hull shelf-plane intersection at every substep",
        "visibility_oracle": "Isaac Sim semantic segmentation exact target-only mask",
        "clearance_oracle": "visual convex-hull surface gap > 5.0 cm",
        "baseline_target_mask_matches_preparation": baseline_matches,
        "path": path,
        "collision": bool(path["collision"]),
        "boundary_violation": bool(path["boundary_violation"]),
        "post_target_visible_area_px": post_pixels,
        "post_visibility": post_visibility,
        "post_target_mask": str(mask_path.relative_to(EXPERIMENT)),
        "post_target_mask_sha256": common.sha256_path(mask_path),
        "post_geometry": geometry,
        "retrieval_precondition_direct_graspable": retrieval_precondition,
        "post_direct_graspable": post_direct,
        "action_valid": bool(action_valid),
        "created_at": utc_now(),
    }
    return artifact


def replay_record(source: dict, artifact: dict, artifact_path: Path) -> dict:
    return {
        "run_id": source["run_id"],
        "scene_id": source["scene_id"],
        "level": "L4",
        "geometry_condition": source["geometry_condition"],
        "direction_resolution": source["direction_resolution"],
        "repeat_index": source["repeat_index"],
        "seed": source["seed"],
        "model_response": source["parsed_response"],
        "inference_post_validation_pass": bool(source["post_validation_pass"]),
        "simulator_executed": True,
        "cache_key": artifact["cache_key"],
        "cache_artifact": str(artifact_path.relative_to(EXPERIMENT)),
        "cache_artifact_sha256": common.sha256_path(artifact_path),
        "collision": artifact["collision"],
        "boundary_violation": artifact["boundary_violation"],
        "post_visibility": artifact["post_visibility"],
        "post_minimum_surface_gap_cm": artifact["post_geometry"]["minimum_surface_gap_cm"],
        "post_direct_graspable": artifact["post_direct_graspable"],
        "action_valid": artifact["action_valid"],
        "failure_reason": None if artifact["action_valid"] else "simulator_replay_failed",
        "finished_at": utc_now(),
    }


def write_progress(expected: int) -> None:
    rows = latest_by_run_id(read_jsonl(REPLAY_PATH))
    common.write_json(PROGRESS_PATH, {
        "experiment_version": common.EXPERIMENT_VERSION,
        "expected_l4_runs": expected,
        "completed_l4_runs": len(rows),
        "simulator_executed_runs": sum(bool(row.get("simulator_executed")) for row in rows.values()),
        "action_valid_runs": sum(bool(row.get("action_valid")) for row in rows.values()),
        "updated_at": utc_now(),
    })


def main() -> None:
    config = read_json(EXPERIMENT / "config/experiment_config.json")
    expected_all = int(config["scheduled_calls"])
    schedule_rows = read_jsonl(EXPERIMENT / "config/run_table.jsonl")
    expected_l4 = sum(row.get("level") == "L4" for row in schedule_rows)
    inference = latest_by_run_id(read_jsonl(RUNS_PATH))
    if len(inference) != expected_all and not args.allow_partial:
        raise RuntimeError(
            f"Inference is incomplete: {len(inference)}/{expected_all}; use --allow-partial only for smoke tests"
        )
    l4 = [row for row in inference.values() if row.get("level") == "L4"]
    if len(l4) != expected_l4 and not args.allow_partial:
        raise RuntimeError(f"L4 inference mismatch: {len(l4)}/{expected_l4}")
    completed = latest_by_run_id(read_jsonl(REPLAY_PATH))
    pending = [
        row for row in l4
        if row["run_id"] not in completed
        or (args.refresh_valid_cache and bool(row.get("post_validation_pass")))
    ]
    print(
        f"L4 replay pending={len(pending)} completed={len(completed)} expected={expected_l4}",
        flush=True,
    )

    valid_groups: dict[str, list[dict]] = defaultdict(list)
    for record in pending:
        if record.get("parse_error"):
            append_jsonl(REPLAY_PATH, replay_stub(record, reason="inference_parse_error"))
        elif not record.get("post_validation_pass"):
            append_jsonl(REPLAY_PATH, replay_stub(record, reason="inference_post_validation_error"))
        else:
            valid_groups[cache_key(record)].append(record)

    missing_keys = []
    for key, records in valid_groups.items():
        artifact_path = CACHE_DIR / f"{key}.json"
        if artifact_path.exists() and not args.refresh_valid_cache:
            artifact = read_json(artifact_path)
            for record in records:
                append_jsonl(REPLAY_PATH, replay_record(record, artifact, artifact_path))
        else:
            missing_keys.append(key)
    if args.max_unique_commands is not None:
        missing_keys = missing_keys[: args.max_unique_commands]
    by_scene: dict[str, list[str]] = defaultdict(list)
    for key in missing_keys:
        by_scene[valid_groups[key][0]["scene_id"]].append(key)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    executed_unique = 0
    runtime_started = time.perf_counter()
    for scene_index, (scene_id, keys) in enumerate(sorted(by_scene.items()), 1):
        representative = valid_groups[keys[0]][0]
        source_usd = ROOT / representative["source_usd"]
        scene_geometry = read_json(EXPERIMENT / f"audit/scene_geometry/{scene_id}.json")
        mapping = read_json(EXPERIMENT / representative["id_mapping"])
        context = omni.usd.get_context()
        context.open_stage(str(source_usd))
        for _ in range(args.settle_ticks):
            simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open {source_usd}")
        roots = {
            int(item["display_id"]): stage.GetPrimAtPath(item["prim_path"])
            for item in mapping["instances"]
        }
        state = {}
        label_by_id = {}
        for index, (display_id, prim) in enumerate(sorted(roots.items()), 1):
            if not prim.IsValid():
                raise RuntimeError(f"Missing mapped prim {mapping['instances']}")
            label = f"v1_replay_{index:03d}_{prim.GetName()}"
            label_object(prim, label)
            label_by_id[display_id] = label
            translate_op = find_xform_op(prim, UsdGeom.XformOp.TypeTranslate)
            rotate_op = find_xform_op(prim, UsdGeom.XformOp.TypeRotateZ)
            visibility_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
            state[display_id] = {
                "prim_path": str(prim.GetPath()),
                "translate_op": translate_op,
                "translate_original": translate_op.Get() if translate_op is not None else None,
                "rotate_op": rotate_op,
                "rotate_original": rotate_op.Get() if rotate_op is not None else None,
                "visibility_attr": visibility_attr,
                "visibility_original": visibility_attr.Get() or UsdGeom.Tokens.inherited,
            }
        target_id = int(mapping["target_object_id"])
        camera_path = CAMERA_DEFAULT
        render_product = rep.create.render_product(
            camera_path, (1280, 960), force_new=True, name=f"V1Replay_{scene_id}"
        )
        annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
        annotator.attach([render_product])
        try:
            for _ in range(8):
                simulation_app.update()
            baseline = capture_target_mask(annotator, label_by_id[target_id])
            expected_baseline = load_mask(EXPERIMENT / f"audit/masks/{scene_id}_target_current.png")
            baseline_matches = bool(np.array_equal(baseline, expected_baseline))
            print(
                f"[{scene_index}/{len(by_scene)}] {scene_id}: unique={len(keys)} "
                f"baseline_match={baseline_matches}", flush=True,
            )
            target_only_mask = load_mask(EXPERIMENT / f"audit/masks/{scene_id}_target_only.png")
            for key in keys:
                record = valid_groups[key][0]
                artifact_path = CACHE_DIR / f"{key}.json"
                succeeded = False
                for attempt_index in range(1, args.max_attempts + 1):
                    reset_scene(state)
                    attempt_started_at = utc_now()
                    started = time.perf_counter()
                    try:
                        artifact = execute_command(
                            record, scene_geometry, mapping, state, annotator,
                            label_by_id[target_id], target_only_mask, baseline_matches,
                        )
                        artifact["runtime"] = {
                            "isaac_sim": ISAAC_SIM_VERSION,
                            "python": platform.python_version(),
                            "translation_substep_cm": TRANSLATION_SUBSTEP_CM,
                            "rotation_substep_deg": ROTATION_SUBSTEP_DEG,
                            "wall_seconds": round(time.perf_counter() - started, 6),
                        }
                        common.write_json(artifact_path, artifact)
                        append_jsonl(ATTEMPTS_PATH, {
                            "cache_key": key, "scene_id": scene_id,
                            "attempt_index": attempt_index, "status": "success",
                            "started_at": attempt_started_at, "finished_at": utc_now(),
                            "wall_seconds": artifact["runtime"]["wall_seconds"],
                        })
                        for duplicate in valid_groups[key]:
                            append_jsonl(REPLAY_PATH, replay_record(duplicate, artifact, artifact_path))
                        succeeded = True
                        executed_unique += 1
                        break
                    except Exception as exc:
                        append_jsonl(ATTEMPTS_PATH, {
                            "cache_key": key, "scene_id": scene_id,
                            "attempt_index": attempt_index, "status": "error",
                            "started_at": attempt_started_at, "finished_at": utc_now(),
                            "wall_seconds": round(time.perf_counter() - started, 6),
                            "error_type": type(exc).__name__, "error": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                if not succeeded:
                    print(f"ERROR: replay failed after retries: {key}", flush=True)
                if executed_unique % 10 == 0:
                    write_progress(len(l4))
        finally:
            annotator.detach([render_product.path])
            render_product.destroy()
    write_progress(len(l4))
    summary = {
        "experiment_version": common.EXPERIMENT_VERSION,
        "replay_engine_version": REPLAY_ENGINE_VERSION,
        "expected_l4_runs": expected_l4,
        "available_l4_inference_runs": len(l4),
        "unique_commands_executed_this_invocation": executed_unique,
        "elapsed_seconds": round(time.perf_counter() - runtime_started, 3),
        "finished_at": utc_now(),
    }
    common.write_json(EXPERIMENT / "logs/replay_last_invocation.json", summary)
    print(json.dumps({**summary, "progress": read_json(PROGRESS_PATH)}, indent=2), flush=True)


failure = None
try:
    print("Entering L4 replay main", flush=True)
    main()
except BaseException as exc:
    failure = exc
    traceback.print_exc()
finally:
    simulation_app.close()
if failure is not None:
    raise failure
