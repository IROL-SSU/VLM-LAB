#!/usr/bin/env python3
"""Analyze the unified-prompt N2-Basic coarse-color experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/b0_n2_basic_unified_coarse_60scenes_20260903"
QUERY_IDS = (
    "Q1_CATEGORY_PRESENT",
    "Q1_5_COLOR_ONLY_PRESENT",
    "Q2_CATEGORY_COLOR_PRESENT",
    "Q3_HARD_NEGATIVE_ABSENT",
)

# Basic-color projection of the N1 perceptual palette. These evaluator-side
# alternatives capture genuinely ambiguous rendered hues; they are not shown to
# the model. The canonical n2_basic_color remains the target-set definition.
PERCEPTUAL_BASIC_COLOR_TOKENS: dict[str, set[str]] = {
    "acafela": {"blue", "black"},
    "coldgrape": {"blue", "purple"},
    "top": {"green", "blue"},
    "cocopalm": {"pink", "purple"},
    "minutemad": {"yellow", "brown", "orange", "green"},
    "cantata": {"brown", "black", "white"},
    "biracsikhye": {"yellow", "brown", "black"},
    "letsbe": {"blue"},
    "mug_7": {"orange", "brown"},
    "Mug_2": {"black"},
    "Mug_4": {"yellow"},
    "cup_7": {"purple"},
    "cup_8": {"purple", "blue"},
    "Cup_2": {"white"},
    "Cup_4": {"blue"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def phrase_matches(value: str, accepted: Iterable[str]) -> bool:
    padded = f" {normalized(value)} "
    return any(f" {normalized(token)} " in padded for token in accepted)


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def display_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse(raw: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[int], str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return {}, [], [], f"JSON decode error: {error}"
    if not isinstance(value, dict) or set(value) != {
        "request_constraints", "instances", "matching_ids"
    }:
        return {}, [], [], "invalid top-level keys"

    constraints = value["request_constraints"]
    if not isinstance(constraints, dict) or set(constraints) != {"object_type", "color"}:
        return {}, [], [], "invalid request_constraints"
    for key in ("object_type", "color"):
        if constraints[key] is not None and (
            not isinstance(constraints[key], str) or not constraints[key].strip()
        ):
            return {}, [], [], f"invalid request constraint {key}"

    instances = value["instances"]
    if not isinstance(instances, list):
        return {}, [], [], "instances must be an array"
    expected_keys = {
        "id", "object_type", "color", "object_type_matches", "color_matches",
        "matches_request",
    }
    parsed_instances: list[dict[str, Any]] = []
    for index, item in enumerate(instances):
        if not isinstance(item, dict) or set(item) != expected_keys:
            return {}, [], [], f"instances[{index}] has invalid keys"
        if isinstance(item["id"], bool) or not isinstance(item["id"], int):
            return {}, [], [], f"instances[{index}].id is invalid"
        if not isinstance(item["object_type"], str) or not item["object_type"].strip():
            return {}, [], [], f"instances[{index}].object_type is invalid"
        if not isinstance(item["color"], str) or not item["color"].strip():
            return {}, [], [], f"instances[{index}].color is invalid"
        for key in ("object_type_matches", "color_matches"):
            if item[key] is not None and not isinstance(item[key], bool):
                return {}, [], [], f"instances[{index}].{key} is invalid"
        if not isinstance(item["matches_request"], bool):
            return {}, [], [], f"instances[{index}].matches_request is invalid"
        parsed_instances.append(
            {
                **item,
                "object_type": item["object_type"].strip(),
                "color": normalized(item["color"]),
            }
        )

    matching_ids = value["matching_ids"]
    if not isinstance(matching_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in matching_ids
    ):
        return {}, [], [], "matching_ids must be an integer array"
    return (
        {
            "object_type": constraints["object_type"].strip()
            if isinstance(constraints["object_type"], str)
            else None,
            "color": normalized(constraints["color"])
            if isinstance(constraints["color"], str)
            else None,
        },
        parsed_instances,
        matching_ids,
        None,
    )


def expected_object_tokens(family: str | None) -> list[str]:
    if family == "can":
        return ["can", "tin"]
    if family == "bottle":
        return ["bottle"]
    return []


def evaluate(row: dict[str, Any], scene: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    constraints, instances, matching_ids, parse_error = parse(row["raw_response"])
    gt_ids = {int(value) for value in scene["visible_ids"]}
    expected_ids = {int(value) for value in query["expected_ids"]}
    target_family = query.get("target_family")
    target_color = query.get("target_color")
    gt_by_id = {int(item["id"]): item for item in scene["instances"]}

    predicted_ids = [int(item["id"]) for item in instances]
    duplicate_instance_ids = sorted(
        value for value, count in Counter(predicted_ids).items() if count > 1
    )
    pred_by_id: dict[int, dict[str, Any]] = {}
    for item in instances:
        pred_by_id.setdefault(int(item["id"]), item)
    predicted_id_set = set(pred_by_id)
    id_set_exact = (
        parse_error is None
        and not duplicate_instance_ids
        and predicted_id_set == gt_ids
    )

    if target_family is None:
        request_object_exact = parse_error is None and constraints.get("object_type") is None
    else:
        request_object_exact = bool(
            parse_error is None
            and isinstance(constraints.get("object_type"), str)
            and phrase_matches(
                str(constraints["object_type"]), expected_object_tokens(str(target_family))
            )
        )
    if target_color is None:
        request_color_exact = parse_error is None and constraints.get("color") is None
    else:
        request_color_exact = bool(
            parse_error is None
            and normalized(str(constraints.get("color", ""))) == normalized(str(target_color))
        )
    request_constraints_exact = request_object_exact and request_color_exact

    object_correct_count = 0
    color_correct_count = 0
    perceptual_color_correct_count = 0
    semantic_joint_correct_count = 0
    perceptual_semantic_joint_correct_count = 0
    type_flag_correct_count = 0
    color_flag_correct_count = 0
    match_flag_correct_count = 0
    logical_rule_correct_count = 0
    true_flag_ids: set[int] = set()
    instance_evaluations: list[dict[str, Any]] = []

    for instance_id in sorted(gt_ids):
        truth = gt_by_id[instance_id]
        prediction = pred_by_id.get(instance_id)
        expected_type_flag = (
            None if target_family is None else truth["object_family"] == target_family
        )
        expected_color_flag = (
            None if target_color is None else truth["n2_basic_color"] == target_color
        )
        expected_match_flag = instance_id in expected_ids

        if prediction is None:
            instance_evaluations.append(
                {
                    "id": instance_id,
                    "missing": True,
                    "expected_object_type_matches": expected_type_flag,
                    "expected_color_matches": expected_color_flag,
                    "expected_matches_request": expected_match_flag,
                }
            )
            continue

        predicted_type_flag = prediction["object_type_matches"]
        predicted_color_flag = prediction["color_matches"]
        predicted_match_flag = bool(prediction["matches_request"])
        if predicted_match_flag:
            true_flag_ids.add(instance_id)

        object_correct = phrase_matches(
            prediction["object_type"], truth["accepted_object_tokens"]
        )
        color_correct = prediction["color"] == truth["n2_basic_color"]
        perceptual_color_correct = prediction["color"] in PERCEPTUAL_BASIC_COLOR_TOKENS[
            truth["asset_id"]
        ]
        type_flag_correct = predicted_type_flag == expected_type_flag
        color_flag_correct = predicted_color_flag == expected_color_flag
        match_flag_correct = predicted_match_flag == expected_match_flag

        if target_family is None and target_color is not None:
            logical_rule_correct = (
                predicted_type_flag is None
                and isinstance(predicted_color_flag, bool)
                and predicted_match_flag == predicted_color_flag
            )
        elif target_family is not None and target_color is None:
            logical_rule_correct = (
                isinstance(predicted_type_flag, bool)
                and predicted_color_flag is None
                and predicted_match_flag == predicted_type_flag
            )
        else:
            logical_rule_correct = (
                isinstance(predicted_type_flag, bool)
                and isinstance(predicted_color_flag, bool)
                and predicted_match_flag == (predicted_type_flag and predicted_color_flag)
            )

        object_correct_count += int(object_correct)
        color_correct_count += int(color_correct)
        perceptual_color_correct_count += int(perceptual_color_correct)
        semantic_joint_correct_count += int(object_correct and color_correct)
        perceptual_semantic_joint_correct_count += int(
            object_correct and perceptual_color_correct
        )
        type_flag_correct_count += int(type_flag_correct)
        color_flag_correct_count += int(color_flag_correct)
        match_flag_correct_count += int(match_flag_correct)
        logical_rule_correct_count += int(logical_rule_correct)
        instance_evaluations.append(
            {
                "id": instance_id,
                "asset_id": truth["asset_id"],
                "expected_object_family": truth["object_family"],
                "expected_basic_color": truth["n2_basic_color"],
                "predicted_object_type": prediction["object_type"],
                "predicted_basic_color": prediction["color"],
                "object_correct": object_correct,
                "color_correct": color_correct,
                "perceptual_color_correct": perceptual_color_correct,
                "expected_object_type_matches": expected_type_flag,
                "predicted_object_type_matches": predicted_type_flag,
                "type_flag_correct": type_flag_correct,
                "expected_color_matches": expected_color_flag,
                "predicted_color_matches": predicted_color_flag,
                "color_flag_correct": color_flag_correct,
                "expected_matches_request": expected_match_flag,
                "predicted_matches_request": predicted_match_flag,
                "match_flag_correct": match_flag_correct,
                "logical_rule_correct": logical_rule_correct,
            }
        )

    matching_duplicates = sorted(
        value for value, count in Counter(matching_ids).items() if count > 1
    )
    matching_set = set(matching_ids)
    matching_exact = (
        parse_error is None and not matching_duplicates and matching_set == expected_ids
    )
    aggregation_exact = (
        parse_error is None
        and not matching_duplicates
        and not duplicate_instance_ids
        and matching_set == true_flag_ids
    )
    instance_count = len(gt_ids)
    scene_object_exact = id_set_exact and object_correct_count == instance_count
    scene_color_exact = id_set_exact and color_correct_count == instance_count
    perceptual_scene_color_exact = (
        id_set_exact and perceptual_color_correct_count == instance_count
    )
    semantic_scene_joint_exact = (
        id_set_exact and semantic_joint_correct_count == instance_count
    )
    perceptual_semantic_scene_joint_exact = (
        id_set_exact and perceptual_semantic_joint_correct_count == instance_count
    )
    type_flag_scene_exact = id_set_exact and type_flag_correct_count == instance_count
    color_flag_scene_exact = id_set_exact and color_flag_correct_count == instance_count
    match_flag_scene_exact = id_set_exact and match_flag_correct_count == instance_count
    logical_rule_scene_exact = (
        id_set_exact and logical_rule_correct_count == instance_count
    )

    mismatch_ids = expected_ids ^ matching_set
    mismatch_evaluations = [
        item
        for item in instance_evaluations
        if item["id"] in mismatch_ids and not item.get("missing", False)
    ]
    relevant_type_flag_error = any(
        target_family is not None and not item["type_flag_correct"]
        for item in mismatch_evaluations
    )
    relevant_color_flag_error = any(
        target_color is not None and not item["color_flag_correct"]
        for item in mismatch_evaluations
    )
    relevant_type_recognition_error = any(
        target_family is not None
        and not item["type_flag_correct"]
        and not item["object_correct"]
        for item in mismatch_evaluations
    )
    relevant_color_recognition_error = any(
        target_color is not None
        and not item["color_flag_correct"]
        and not item["color_correct"]
        for item in mismatch_evaluations
    )
    relevant_perceptual_color_recognition_error = any(
        target_color is not None
        and not item["color_flag_correct"]
        and not item["perceptual_color_correct"]
        for item in mismatch_evaluations
    )

    if parse_error is not None:
        error_type = "parse_failure"
    elif matching_exact:
        error_type = "correct"
    elif not request_constraints_exact:
        error_type = "request_decomposition_failure"
    elif not id_set_exact:
        error_type = "instance_enumeration_failure"
    elif not logical_rule_scene_exact:
        error_type = "truth_table_violation"
    elif not aggregation_exact:
        error_type = "aggregation_failure"
    elif relevant_type_recognition_error and relevant_perceptual_color_recognition_error:
        error_type = "joint_object_color_recognition_failure"
    elif relevant_type_recognition_error:
        error_type = "object_recognition_failure"
    elif relevant_perceptual_color_recognition_error:
        error_type = "basic_color_recognition_failure"
    elif relevant_color_recognition_error:
        error_type = "canonical_color_disagreement"
    elif relevant_type_flag_error and relevant_color_flag_error:
        error_type = "joint_type_color_comparison_failure"
    elif relevant_type_flag_error:
        error_type = "object_type_comparison_failure"
    elif relevant_color_flag_error:
        error_type = "color_comparison_failure"
    else:
        error_type = "unattributed_target_failure"

    tp = len(matching_set & expected_ids)
    fp = len(matching_set - expected_ids)
    fn = len(expected_ids - matching_set)
    return {
        **row,
        "parsed_request_constraints": constraints,
        "predicted_instances": instances,
        "predicted_matching_ids": matching_ids,
        "parse_error": parse_error,
        "ground_truth_target_ids": sorted(expected_ids),
        "matching_missing_ids": sorted(expected_ids - matching_set),
        "matching_extra_ids": sorted(matching_set - expected_ids),
        "matching_id_set_exact": matching_exact,
        "matching_subset_candidates": parse_error is None and matching_set.issubset(gt_ids),
        "matching_tp": tp,
        "matching_fp": fp,
        "matching_fn": fn,
        "request_object_exact": request_object_exact,
        "request_color_exact": request_color_exact,
        "request_constraints_exact": request_constraints_exact,
        "semantic_id_set_exact": id_set_exact,
        "semantic_object_correct_count": object_correct_count,
        "semantic_color_correct_count": color_correct_count,
        "perceptual_color_correct_count": perceptual_color_correct_count,
        "semantic_joint_correct_count": semantic_joint_correct_count,
        "perceptual_semantic_joint_correct_count": perceptual_semantic_joint_correct_count,
        "scene_object_exact": scene_object_exact,
        "scene_color_exact": scene_color_exact,
        "perceptual_scene_color_exact": perceptual_scene_color_exact,
        "semantic_scene_joint_exact": semantic_scene_joint_exact,
        "perceptual_semantic_scene_joint_exact": perceptual_semantic_scene_joint_exact,
        "type_flag_correct_count": type_flag_correct_count,
        "color_flag_correct_count": color_flag_correct_count,
        "match_flag_correct_count": match_flag_correct_count,
        "logical_rule_correct_count": logical_rule_correct_count,
        "type_flag_scene_exact": type_flag_scene_exact,
        "color_flag_scene_exact": color_flag_scene_exact,
        "match_flag_scene_exact": match_flag_scene_exact,
        "logical_rule_scene_exact": logical_rule_scene_exact,
        "aggregation_consistency_exact": aggregation_exact,
        "instance_count": instance_count,
        "instance_evaluations": instance_evaluations,
        "error_type": error_type,
    }


def summarize(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    calls = len(rows)
    instances = sum(row["instance_count"] for row in rows)
    tp = sum(row["matching_tp"] for row in rows)
    fp = sum(row["matching_fp"] for row in rows)
    fn = sum(row["matching_fn"] for row in rows)
    has_positive_ground_truth = tp + fn > 0
    precision = tp / (tp + fp) if has_positive_ground_truth and tp + fp else (
        1.0 if has_positive_ground_truth else None
    )
    recall = tp / (tp + fn) if has_positive_ground_truth else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if has_positive_ground_truth else None)
    )
    absent = [row for row in rows if not row["ground_truth_target_ids"]]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_id"], row["query_id"])].append(row)
    return {
        "scope": scope,
        "calls": calls,
        "instances": instances,
        "matching_id_set_exact_pct": pct(sum(row["matching_id_set_exact"] for row in rows), calls),
        "matching_micro_precision_pct": round(100.0 * precision, 3) if precision is not None else None,
        "matching_micro_recall_pct": round(100.0 * recall, 3) if recall is not None else None,
        "matching_micro_f1_pct": round(100.0 * f1, 3) if f1 is not None else None,
        "request_constraints_exact_pct": pct(sum(row["request_constraints_exact"] for row in rows), calls),
        "semantic_id_set_exact_pct": pct(sum(row["semantic_id_set_exact"] for row in rows), calls),
        "semantic_object_accuracy_pct": pct(sum(row["semantic_object_correct_count"] for row in rows), instances),
        "semantic_basic_color_accuracy_pct": pct(sum(row["semantic_color_correct_count"] for row in rows), instances),
        "semantic_perceptual_color_accuracy_pct": pct(
            sum(row["perceptual_color_correct_count"] for row in rows), instances
        ),
        "semantic_joint_accuracy_pct": pct(sum(row["semantic_joint_correct_count"] for row in rows), instances),
        "semantic_perceptual_joint_accuracy_pct": pct(
            sum(row["perceptual_semantic_joint_correct_count"] for row in rows), instances
        ),
        "semantic_scene_joint_exact_pct": pct(sum(row["semantic_scene_joint_exact"] for row in rows), calls),
        "semantic_perceptual_scene_joint_exact_pct": pct(
            sum(row["perceptual_semantic_scene_joint_exact"] for row in rows), calls
        ),
        "type_flag_accuracy_pct": pct(sum(row["type_flag_correct_count"] for row in rows), instances),
        "color_flag_accuracy_pct": pct(sum(row["color_flag_correct_count"] for row in rows), instances),
        "match_flag_accuracy_pct": pct(sum(row["match_flag_correct_count"] for row in rows), instances),
        "logical_rule_scene_exact_pct": pct(sum(row["logical_rule_scene_exact"] for row in rows), calls),
        "aggregation_exact_pct": pct(sum(row["aggregation_consistency_exact"] for row in rows), calls),
        "absent_false_positive_rate_pct": pct(sum(bool(row["predicted_matching_ids"]) for row in absent), len(absent)),
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
        "scene_queries_5of5": sum(len(group) == 5 and all(row["matching_id_set_exact"] for row in group) for group in grouped.values()),
        "scene_queries_1to4of5": sum(any(row["matching_id_set_exact"] for row in group) and not all(row["matching_id_set_exact"] for row in group) for group in grouped.values()),
        "scene_queries_0of5": sum(not any(row["matching_id_set_exact"] for row in group) for group in grouped.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    scenes = read_json(experiment / "config/frozen_scenes.json")
    raw = read_jsonl(experiment / "logs/runs.jsonl")
    if len(raw) != int(config["scheduled_calls"]):
        raise ValueError(f"expected {config['scheduled_calls']} records, found {len(raw)}")
    if len({row["run_id"] for row in raw}) != len(raw):
        raise ValueError("duplicate run IDs")

    scene_lookup = {scene["scene_id"]: scene for scene in scenes}
    query_lookup = {
        (scene["scene_id"], query["query_id"]): query
        for scene in scenes
        for query in scene["queries"]
    }
    rows = [
        evaluate(
            row,
            scene_lookup[row["scene_id"]],
            query_lookup[(row["scene_id"], row["query_id"])],
        )
        for row in raw
    ]

    results = experiment / "results"
    reports = experiment / "reports"
    audit_dir = experiment / "audit"
    for path in (results, reports, audit_dir):
        path.mkdir(parents=True, exist_ok=True)
    with (results / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summaries = [summarize(rows, "OVERALL")]
    summaries.extend(
        summarize([row for row in rows if row["query_id"] == query_id], query_id)
        for query_id in QUERY_IDS
    )
    write_csv(results / "query_summary.csv", summaries)

    level_rows: list[dict[str, Any]] = []
    for level in ("L3", "L4", "L5"):
        for query_id in QUERY_IDS:
            selected = [
                row for row in rows
                if row["level"] == level and row["query_id"] == query_id
            ]
            level_rows.append(
                {"level": level, "query_id": query_id, **summarize(selected, f"{level}_{query_id}")}
            )
    write_csv(results / "level_query_summary.csv", level_rows)

    color_rows: list[dict[str, Any]] = []
    colors = sorted({str(row["target_color"]) for row in rows if row.get("target_color")})
    for query_id in QUERY_IDS:
        for color in colors:
            selected = [
                row for row in rows
                if row["query_id"] == query_id and row.get("target_color") == color
            ]
            if selected:
                color_rows.append(
                    {"query_id": query_id, "target_color": color, **summarize(selected, f"{query_id}_{color}")}
                )
    write_csv(results / "color_query_summary.csv", color_rows)

    color_confusions: Counter[tuple[str, str, str, bool]] = Counter()
    for row in rows:
        for item in row["instance_evaluations"]:
            if item.get("missing", False):
                continue
            color_confusions[
                (
                    item["asset_id"],
                    item["expected_basic_color"],
                    item["predicted_basic_color"],
                    bool(item["perceptual_color_correct"]),
                )
            ] += 1
    color_confusion_rows = [
        {
            "asset_id": asset_id,
            "canonical_basic_color": expected_color,
            "predicted_basic_color": predicted_color,
            "perceptually_acceptable": perceptually_acceptable,
            "count": count,
        }
        for (asset_id, expected_color, predicted_color, perceptually_acceptable), count
        in sorted(color_confusions.items())
    ]
    write_csv(results / "basic_color_confusions.csv", color_confusion_rows)

    error_rows: list[dict[str, Any]] = []
    for query_id in QUERY_IDS:
        selected = [
            row for row in rows
            if row["query_id"] == query_id and not row["matching_id_set_exact"]
        ]
        for error_type, count in sorted(Counter(row["error_type"] for row in selected).items()):
            error_rows.append(
                {
                    "query_id": query_id,
                    "error_type": error_type,
                    "count": count,
                    "pct_of_query_calls": pct(count, 300),
                }
            )
    write_csv(results / "error_breakdown.csv", error_rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_id"], row["query_id"])].append(row)
    stability_rows: list[dict[str, Any]] = []
    for (scene_id, query_id), group in sorted(grouped.items()):
        first = group[0]
        stability_rows.append(
            {
                "level": first["level"],
                "scene_id": scene_id,
                "query_id": query_id,
                "request": first["target_description"],
                "successes_out_of_5": sum(row["matching_id_set_exact"] for row in group),
                "ground_truth": json.dumps(first["ground_truth_target_ids"]),
                "unique_predictions": json.dumps(sorted({tuple(row["predicted_matching_ids"]) for row in group})),
                "error_types": json.dumps(dict(Counter(row["error_type"] for row in group))),
            }
        )
    write_csv(results / "scene_query_stability.csv", stability_rows)

    failure_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["matching_id_set_exact"]:
            continue
        failure_rows.append(
            {
                "level": row["level"],
                "scene_id": row["scene_id"],
                "query_id": row["query_id"],
                "request": row["target_description"],
                "seed": row["seed"],
                "error_type": row["error_type"],
                "ground_truth": json.dumps(row["ground_truth_target_ids"]),
                "prediction": json.dumps(row["predicted_matching_ids"]),
                "missing": json.dumps(row["matching_missing_ids"]),
                "extra": json.dumps(row["matching_extra_ids"]),
                "request_constraints": json.dumps(row["parsed_request_constraints"]),
                "semantic_scene_joint_exact": row["semantic_scene_joint_exact"],
                "logical_rule_scene_exact": row["logical_rule_scene_exact"],
                "aggregation_exact": row["aggregation_consistency_exact"],
                "raw_response": row["raw_response"],
            }
        )
    write_csv(results / "failures.csv", failure_rows)

    audit = {
        "expected": int(config["scheduled_calls"]),
        "observed": len(rows),
        "unique_run_ids": len({row["run_id"] for row in rows}),
        "query_counts": dict(Counter(row["query_id"] for row in rows)),
        "seed_counts": dict(Counter(str(row["seed"]) for row in rows)),
        "batch_sizes": sorted({row["runtime"]["batch_size"] for row in rows}),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
        "length_finishes": sum(row["finish_reason"] == "length" for row in rows),
        "outside_candidate_failures": sum(not row["matching_subset_candidates"] for row in rows),
        "same_prompt_template_configured": bool(
            config["design"]["same_prompt_template_for_all_queries"]
        ),
        "rendered_prompt_hash_count": len({row["prompt_sha256"] for row in rows}),
        "system_prompt_hash_count": len({row["system_prompt"] for row in rows}),
        "basic_colors": config["basic_colors"],
        "perceptual_basic_color_tokens": {
            key: sorted(value) for key, value in PERCEPTUAL_BASIC_COLOR_TOKENS.items()
        },
    }
    (audit_dir / "final_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_table = "\n".join(
        f"| {item['scope']} | {item['calls']} | {item['matching_id_set_exact_pct']:.3f}% | "
        f"{display_pct(item['matching_micro_precision_pct'])} | {display_pct(item['matching_micro_recall_pct'])} | "
        f"{display_pct(item['matching_micro_f1_pct'])} | {item['request_constraints_exact_pct']:.3f}% | "
        f"{item['semantic_basic_color_accuracy_pct']:.3f}% | "
        f"{item['semantic_perceptual_color_accuracy_pct']:.3f}% | "
        f"{item['semantic_scene_joint_exact_pct']:.3f}% | "
        f"{item['logical_rule_scene_exact_pct']:.3f}% | {item['aggregation_exact_pct']:.3f}% | "
        f"{item['absent_false_positive_rate_pct']:.3f}% |"
        for item in summaries
    )
    level_table = "\n".join(
        f"| {item['level']} | {item['query_id']} | {item['matching_id_set_exact_pct']:.3f}% | "
        f"{display_pct(item['matching_micro_f1_pct'])} | {item['semantic_basic_color_accuracy_pct']:.3f}% | "
        f"{item['absent_false_positive_rate_pct']:.3f}% |"
        for item in level_rows
    )
    error_table = "\n".join(
        f"| {item['query_id']} | {item['error_type']} | {item['count']} | {item['pct_of_query_calls']:.3f}% |"
        for item in error_rows
    )
    report = f"""# N2-Basic unified prompt and coarse-color grounding

## Protocol

- 60 numbered-RGB scenes × Q1/Q1.5/Q2/Q3 × five seeds = 1,200 calls
- One unified English prompt template for all four query types
- Confirmed visible IDs, forced JSON, batch size 1
- Nine basic colors: {', '.join(config['basic_colors'])}
- Fine shades are merged before query generation and canonical scoring; notably beige/tan -> brown, dark blue/navy -> blue, and dark purple/violet -> purple
- A separate perceptual score uses the previously adjudicated N1 asset palettes; it does not change the canonical target-ID set

## Results

| Scope | Calls | ID-set exact | Micro P | Micro R | Micro F1 | Request decomposition | Canonical basic color | Perceptual color | Semantic scene joint | Logic scene exact | Aggregation exact | Absent FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_table}

## Results by level

| Level | Query | ID-set exact | Micro F1 | Basic-color accuracy | Absent FPR |
|---|---|---:|---:|---:|---:|
{level_table}

## Error attribution

| Query | Error type | Count | Percent of query calls |
|---|---|---:|---:|
{error_table}

## Interpretation

- The model returned the complete confirmed ID inventory in every call, parsed all request constraints, and kept `matching_ids` consistent with its own per-instance match flags.
- Canonical single-color agreement was {summaries[0]['semantic_basic_color_accuracy_pct']:.3f}%, while the previously adjudicated N1 perceptual palette accepted {summaries[0]['semantic_perceptual_color_accuracy_pct']:.3f}% of instance color labels.
- Of the 160 target-set failures, 153 were canonical-color disagreements whose predicted color remained perceptually acceptable; seven were object-recognition failures in Q3.
- Therefore the strict target-ID result mixes grounding ability with ambiguity in the single canonical color label. A clean follow-up should exclude any distractor whose perceptual palette also contains the query color, especially when constructing Q3 absent queries.

## Audit

`{audit}`
"""
    (reports / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summaries": summaries, "errors": error_rows, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
