#!/usr/bin/env python3
"""Analyze explicit request-decomposition and condition-matching scaffold N2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import analyze_b0_n1_open_vocab_object_color as n1


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/b0_n2_explicit_constraint_scaffold_60scenes_20260828"
BASELINE = ROOT / "experiments/b0_n2_numbered_rgb_semantic_scaffold_60scenes_20260827"
QUERY_IDS = (
    "Q1_CATEGORY_PRESENT",
    "Q2_CATEGORY_COLOR_PRESENT",
    "Q3_HARD_NEGATIVE_ABSENT",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


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


def exact_binomial(left: int, right: int) -> float:
    total = left + right
    if not total:
        return 1.0
    low = min(left, right)
    return min(1.0, 2.0 * sum(math.comb(total, k) for k in range(low + 1)) / (2**total))


def parse(raw: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[int], str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return {}, [], [], f"JSON decode error: {error}"
    if not isinstance(value, dict) or set(value) != {"request_constraints", "instances", "matching_ids"}:
        return {}, [], [], "invalid top-level keys"

    constraints = value["request_constraints"]
    if not isinstance(constraints, dict) or set(constraints) != {"object_type", "color"}:
        return {}, [], [], "invalid request_constraints"
    if not isinstance(constraints["object_type"], str) or not constraints["object_type"].strip():
        return {}, [], [], "invalid requested object_type"
    if constraints["color"] is not None and (
        not isinstance(constraints["color"], str) or not constraints["color"].strip()
    ):
        return {}, [], [], "invalid requested color"

    instances = value["instances"]
    if not isinstance(instances, list):
        return {}, [], [], "instances must be an array"
    parsed_instances: list[dict[str, Any]] = []
    expected_keys = {
        "id", "object_type", "color", "object_type_matches", "color_matches",
        "matches_request",
    }
    for index, item in enumerate(instances):
        if not isinstance(item, dict) or set(item) != expected_keys:
            return {}, [], [], f"instances[{index}] has invalid keys"
        if isinstance(item["id"], bool) or not isinstance(item["id"], int):
            return {}, [], [], f"instances[{index}].id is invalid"
        if not isinstance(item["object_type"], str) or not item["object_type"].strip():
            return {}, [], [], f"instances[{index}].object_type is invalid"
        if not isinstance(item["color"], str) or not item["color"].strip():
            return {}, [], [], f"instances[{index}].color is invalid"
        if not isinstance(item["object_type_matches"], bool):
            return {}, [], [], f"instances[{index}].object_type_matches is invalid"
        if item["color_matches"] is not None and not isinstance(item["color_matches"], bool):
            return {}, [], [], f"instances[{index}].color_matches is invalid"
        if not isinstance(item["matches_request"], bool):
            return {}, [], [], f"instances[{index}].matches_request is invalid"
        parsed_instances.append({
            **item,
            "object_type": item["object_type"].strip(),
            "color": item["color"].strip(),
        })

    matching_ids = value["matching_ids"]
    if not isinstance(matching_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in matching_ids
    ):
        return {}, [], [], "matching_ids must be an integer array"
    return {
        "object_type": constraints["object_type"].strip(),
        "color": constraints["color"].strip() if isinstance(constraints["color"], str) else None,
    }, parsed_instances, matching_ids, None


def expected_request_object_tokens(family: str) -> list[str]:
    return ["can", "tin"] if family == "can" else ["bottle"]


def evaluate(row: dict[str, Any], scene: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    constraints, instances, matching_ids, parse_error = parse(row["raw_response"])
    semantic_instances = [
        {"id": item["id"], "object_type": item["object_type"], "color": item["color"]}
        for item in instances
    ]
    semantic = n1.evaluate_record({
        **row,
        "mode": "JSON_FORCED",
        "ground_truth": {"instances": scene["instances"]},
        "raw_response": json.dumps({"instances": semantic_instances}, ensure_ascii=False),
    })
    if parse_error is not None:
        semantic["format_valid"] = False
        semantic["parse_error"] = parse_error

    gt_ids = {int(value) for value in scene["visible_ids"]}
    expected_ids = {int(value) for value in query["expected_ids"]}
    target_family = str(query["target_family"])
    target_color = query.get("target_color")
    gt_by_id = {int(item["id"]): item for item in scene["instances"]}

    predicted_ids = [int(item["id"]) for item in instances]
    duplicate_ids = sorted(value for value, count in Counter(predicted_ids).items() if count > 1)
    pred_by_id: dict[int, dict[str, Any]] = {}
    for item in instances:
        pred_by_id.setdefault(int(item["id"]), item)
    id_set_exact = parse_error is None and not duplicate_ids and set(pred_by_id) == gt_ids

    request_object_exact = bool(
        parse_error is None
        and n1.phrase_matches(constraints.get("object_type", ""), expected_request_object_tokens(target_family))
    )
    if target_color is None:
        request_color_exact = parse_error is None and constraints.get("color") is None
    else:
        request_color_exact = bool(
            parse_error is None
            and isinstance(constraints.get("color"), str)
            and n1.normalized(str(constraints["color"])) == n1.normalized(str(target_color))
        )
    request_constraints_exact = request_object_exact and request_color_exact

    type_flag_correct = 0
    null_color_flag_correct = 0
    match_flag_correct = 0
    rule_correct = 0
    true_flag_ids: set[int] = set()
    flag_evaluations: list[dict[str, Any]] = []
    for instance_id in sorted(gt_ids):
        item = pred_by_id.get(instance_id)
        expected_type_flag = gt_by_id[instance_id]["object_family"] == target_family
        expected_match_flag = instance_id in expected_ids
        if item is None:
            flag_evaluations.append({
                "id": instance_id,
                "expected_object_type_matches": expected_type_flag,
                "expected_matches_request": expected_match_flag,
                "predicted_object_type_matches": None,
                "predicted_color_matches": None,
                "predicted_matches_request": None,
                "object_type_flag_correct": False,
                "matches_request_flag_correct": False,
                "logical_rule_correct": False,
            })
            continue
        predicted_type_flag = bool(item["object_type_matches"])
        predicted_color_flag = item["color_matches"]
        predicted_match_flag = bool(item["matches_request"])
        if predicted_match_flag:
            true_flag_ids.add(instance_id)
        object_type_flag_correct = predicted_type_flag == expected_type_flag
        match_request_flag_correct = predicted_match_flag == expected_match_flag
        if constraints.get("color") is None:
            logical_rule_correct = predicted_color_flag is None and predicted_match_flag == predicted_type_flag
            null_color_flag_correct += int(predicted_color_flag is None)
        else:
            logical_rule_correct = (
                isinstance(predicted_color_flag, bool)
                and predicted_match_flag == (predicted_type_flag and predicted_color_flag)
            )
        type_flag_correct += int(object_type_flag_correct)
        match_flag_correct += int(match_request_flag_correct)
        rule_correct += int(logical_rule_correct)
        flag_evaluations.append({
            "id": instance_id,
            "expected_object_type_matches": expected_type_flag,
            "expected_matches_request": expected_match_flag,
            "predicted_object_type_matches": predicted_type_flag,
            "predicted_color_matches": predicted_color_flag,
            "predicted_matches_request": predicted_match_flag,
            "object_type_flag_correct": object_type_flag_correct,
            "matches_request_flag_correct": match_request_flag_correct,
            "logical_rule_correct": logical_rule_correct,
        })

    matching_duplicates = sorted(value for value, count in Counter(matching_ids).items() if count > 1)
    matching_set = set(matching_ids)
    matching_exact = parse_error is None and not matching_duplicates and matching_set == expected_ids
    aggregation_exact = (
        parse_error is None and not matching_duplicates and not duplicate_ids
        and matching_set == true_flag_ids
    )
    logical_rule_scene_exact = id_set_exact and rule_correct == len(gt_ids)
    type_flag_scene_exact = id_set_exact and type_flag_correct == len(gt_ids)
    match_flag_scene_exact = id_set_exact and match_flag_correct == len(gt_ids)

    if parse_error is not None:
        error_type = "parse_failure"
    elif matching_exact:
        error_type = "correct"
    elif not request_constraints_exact:
        error_type = "request_decomposition_failure"
    elif not logical_rule_scene_exact:
        error_type = "truth_table_violation"
    elif not type_flag_scene_exact:
        error_type = "object_type_comparison_failure"
    elif semantic["scene_joint_exact"]:
        error_type = "gt_target_set_disagreement_with_semantic_inventory_exact"
    else:
        error_type = "target_failure_with_semantic_error"

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
        "request_object_exact": request_object_exact,
        "request_color_exact": request_color_exact,
        "request_constraints_exact": request_constraints_exact,
        "semantic_id_set_exact": semantic["id_set_exact"],
        "semantic_object_correct_count": semantic["object_correct_count"],
        "semantic_color_correct_count": semantic["color_correct_count"],
        "semantic_joint_correct_count": semantic["joint_correct_count"],
        "semantic_scene_joint_exact": semantic["scene_joint_exact"],
        "semantic_instance_evaluations": semantic["instance_evaluations"],
        "type_flag_correct_count": type_flag_correct,
        "type_flag_scene_exact": type_flag_scene_exact,
        "null_color_flag_correct_count": null_color_flag_correct,
        "match_flag_correct_count": match_flag_correct,
        "match_flag_scene_exact": match_flag_scene_exact,
        "logical_rule_correct_count": rule_correct,
        "logical_rule_scene_exact": logical_rule_scene_exact,
        "aggregation_consistency_exact": aggregation_exact,
        "flag_evaluations": flag_evaluations,
        "instance_count": len(gt_ids),
        "error_type": error_type,
    }


def summarize(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    calls = len(rows)
    instances = sum(row["instance_count"] for row in rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_id"], row["query_id"])].append(row)
    absent = [row for row in rows if not row["ground_truth_target_ids"]]
    return {
        "scope": scope,
        "calls": calls,
        "instances": instances,
        "matching_id_set_exact_pct": pct(sum(row["matching_id_set_exact"] for row in rows), calls),
        "request_object_exact_pct": pct(sum(row["request_object_exact"] for row in rows), calls),
        "request_color_exact_pct": pct(sum(row["request_color_exact"] for row in rows), calls),
        "request_constraints_exact_pct": pct(sum(row["request_constraints_exact"] for row in rows), calls),
        "semantic_id_set_exact_pct": pct(sum(row["semantic_id_set_exact"] for row in rows), calls),
        "semantic_object_accuracy_pct": pct(sum(row["semantic_object_correct_count"] for row in rows), instances),
        "semantic_color_accuracy_pct": pct(sum(row["semantic_color_correct_count"] for row in rows), instances),
        "semantic_joint_accuracy_pct": pct(sum(row["semantic_joint_correct_count"] for row in rows), instances),
        "semantic_scene_joint_exact_pct": pct(sum(row["semantic_scene_joint_exact"] for row in rows), calls),
        "object_type_flag_accuracy_pct": pct(sum(row["type_flag_correct_count"] for row in rows), instances),
        "object_type_flag_scene_exact_pct": pct(sum(row["type_flag_scene_exact"] for row in rows), calls),
        "matches_request_flag_accuracy_pct": pct(sum(row["match_flag_correct_count"] for row in rows), instances),
        "matches_request_flag_scene_exact_pct": pct(sum(row["match_flag_scene_exact"] for row in rows), calls),
        "logical_rule_accuracy_pct": pct(sum(row["logical_rule_correct_count"] for row in rows), instances),
        "logical_rule_scene_exact_pct": pct(sum(row["logical_rule_scene_exact"] for row in rows), calls),
        "aggregation_consistency_exact_pct": pct(sum(row["aggregation_consistency_exact"] for row in rows), calls),
        "matching_subset_candidates_pct": pct(sum(row["matching_subset_candidates"] for row in rows), calls),
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
    experiment = args.experiment.resolve()
    config = read_json(experiment / "config/experiment_config.json")
    scenes = read_json(experiment / "config/frozen_scenes.json")
    raw = read_jsonl(experiment / "logs/runs.jsonl")
    if len(raw) != int(config["scheduled_calls"]) or len({row["run_id"] for row in raw}) != len(raw):
        raise ValueError("run integrity failure")

    scene_lookup = {scene["scene_id"]: scene for scene in scenes}
    query_lookup = {
        (scene["scene_id"], query["query_id"]): query
        for scene in scenes for query in scene["queries"]
    }
    rows = [
        evaluate(row, scene_lookup[row["scene_id"]], query_lookup[(row["scene_id"], row["query_id"])])
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
    summaries.extend(summarize([row for row in rows if row["query_id"] == qid], qid) for qid in QUERY_IDS)
    write_csv(results / "query_summary.csv", summaries)

    level_summaries = []
    for level in ("L3", "L4", "L5"):
        for qid in QUERY_IDS:
            selected = [row for row in rows if row["level"] == level and row["query_id"] == qid]
            level_summaries.append({"level": level, "query_id": qid, **summarize(selected, f"{level}_{qid}")})
    write_csv(results / "level_query_summary.csv", level_summaries)

    error_rows = []
    for qid in QUERY_IDS:
        selected = [row for row in rows if row["query_id"] == qid and not row["matching_id_set_exact"]]
        for error_type, count in sorted(Counter(row["error_type"] for row in selected).items()):
            error_rows.append({
                "query_id": qid,
                "error_type": error_type,
                "count": count,
                "pct_of_query_calls": pct(count, len([row for row in rows if row["query_id"] == qid])),
            })
    write_csv(results / "target_error_breakdown.csv", error_rows)

    stability_rows = []
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["scene_id"], row["query_id"])].append(row)
    for (scene_id, query_id), group in sorted(grouped_rows.items()):
        success_count = sum(row["matching_id_set_exact"] for row in group)
        first = group[0]
        stability_rows.append({
            "level": first["level"],
            "scene_id": scene_id,
            "query_id": query_id,
            "user_request": first["target_description"],
            "successes_out_of_5": success_count,
            "ground_truth": json.dumps(first["ground_truth_target_ids"]),
            "unique_predictions": json.dumps(sorted({tuple(row["predicted_matching_ids"]) for row in group})),
            "error_types": json.dumps(dict(Counter(row["error_type"] for row in group))),
        })
    write_csv(results / "scene_query_stability.csv", stability_rows)

    failures = []
    for row in rows:
        if row["matching_id_set_exact"]:
            continue
        failures.append({
            "level": row["level"],
            "scene_id": row["scene_id"],
            "query_id": row["query_id"],
            "user_request": row["target_description"],
            "seed": row["seed"],
            "error_type": row["error_type"],
            "ground_truth": json.dumps(row["ground_truth_target_ids"]),
            "prediction": json.dumps(row["predicted_matching_ids"]),
            "missing": json.dumps(row["matching_missing_ids"]),
            "extra": json.dumps(row["matching_extra_ids"]),
            "request_constraints": json.dumps(row["parsed_request_constraints"]),
            "semantic_scene_joint_exact": row["semantic_scene_joint_exact"],
            "logical_rule_scene_exact": row["logical_rule_scene_exact"],
            "aggregation_consistency_exact": row["aggregation_consistency_exact"],
            "raw_response": row["raw_response"],
        })
    write_csv(results / "failures.csv", failures)

    baseline_rows = {row["run_id"]: row for row in read_jsonl(BASELINE / "results/evaluated_records.jsonl")}
    paired = []
    for scope in ("OVERALL", *QUERY_IDS):
        selected = rows if scope == "OVERALL" else [row for row in rows if row["query_id"] == scope]
        both = baseline_only = explicit_only = neither = 0
        for row in selected:
            old = bool(baseline_rows[row["run_id"]]["matching_id_set_exact"])
            new = bool(row["matching_id_set_exact"])
            both += int(old and new)
            baseline_only += int(old and not new)
            explicit_only += int(not old and new)
            neither += int(not old and not new)
        paired.append({
            "scope": scope,
            "paired_records": len(selected),
            "both_correct": both,
            "baseline_only_correct": baseline_only,
            "explicit_constraint_only_correct": explicit_only,
            "neither_correct": neither,
            "explicit_minus_baseline_pp": round(100.0 * (explicit_only - baseline_only) / len(selected), 3),
            "mcnemar_exact_p": exact_binomial(baseline_only, explicit_only),
        })
    write_csv(results / "paired_vs_semantic_scaffold.csv", paired)

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
    }
    (audit_dir / "final_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_table = "\n".join(
        f"| {item['scope']} | {item['matching_id_set_exact_pct']:.3f}% | {item['request_constraints_exact_pct']:.3f}% | {item['object_type_flag_accuracy_pct']:.3f}% | {item['logical_rule_scene_exact_pct']:.3f}% | {item['aggregation_consistency_exact_pct']:.3f}% | {item['semantic_scene_joint_exact_pct']:.3f}% | {item['absent_false_positive_rate_pct']:.3f}% |"
        for item in summaries
    )
    level_table = "\n".join(
        f"| {item['level']} | {item['query_id']} | {item['matching_id_set_exact_pct']:.3f}% | {item['logical_rule_scene_exact_pct']:.3f}% | {item['absent_false_positive_rate_pct']:.3f}% |"
        for item in level_summaries
    )
    paired_table = "\n".join(
        f"| {item['scope']} | {item['explicit_minus_baseline_pp']:+.3f} pp | {item['baseline_only_correct']} | {item['explicit_constraint_only_correct']} | {item['mcnemar_exact_p']:.6g} |"
        for item in paired
    )
    error_table = "\n".join(
        f"| {item['query_id']} | {item['error_type']} | {item['count']} | {item['pct_of_query_calls']:.3f}% |"
        for item in error_rows
    )
    report = f"""# Explicit request-constraint scaffold N2

## Protocol

- 60 numbered-RGB scenes × Q1/Q2/Q3 × five seeds = 900 calls
- Natural English request, confirmed visible IDs, forced JSON, batch size 1
- Controlled paired comparison with the prior semantic-scaffold run: identical scenes, requests, seeds, model, and decoding parameters

## Results

| Scope | Matching-ID exact | Request decomposition exact | Type-flag accuracy | Truth-table scene exact | Aggregation exact | Semantic scene joint exact | Absent FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary_table}

## Results by level

| Level | Query | Matching-ID exact | Truth-table scene exact | Absent FPR |
|---|---|---:|---:|---:|
{level_table}

## Paired comparison

| Scope | Explicit − baseline | Baseline-only correct | Explicit-only correct | McNemar p |
|---|---:|---:|---:|---:|
{paired_table}

## Error attribution

| Query | Error type | Count | Percent of query calls |
|---|---|---:|---:|
{error_table}

## Evaluation caveat

The target-ID ground truth was generated from asset-level canonical color labels. Some rendered objects admit different but perceptually reasonable color descriptions. For example, the evaluator expects `beige can` while the model consistently describes the visible body as `brown`, and some `dark blue bottle` queries contain additional bottles that the model also describes as dark blue although they are excluded from the asset-derived target set. Consequently, Q2 absolute Matching-ID exact should be interpreted together with a human review of `scene_query_stability.csv`, rather than treating every target-set disagreement as an unambiguous reasoning failure.

## Audit

`{audit}`
"""
    (reports / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired": paired, "errors": error_rows, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
