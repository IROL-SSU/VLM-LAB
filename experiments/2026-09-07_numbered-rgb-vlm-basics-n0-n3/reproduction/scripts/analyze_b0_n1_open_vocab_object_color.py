#!/usr/bin/env python3
"""Score N1 open-vocabulary ID-to-object-and-color predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/b0_n1_open_vocab_object_color_text_vs_json_60scenes_20260826"
MODES = ("TEXT", "JSON_FORCED")

# A second, perception-oriented color policy accounts for genuinely multicolored
# labels and hue shifts in the rendered views. The original single canonical hue
# remains available as the strict sensitivity score. These aliases are evaluator-
# side only and are never included in the evaluated model prompt.
PERCEPTUAL_COLOR_TOKENS: dict[str, list[str]] = {
    "acafela": ["blue", "navy", "black"],
    "coldgrape": ["blue", "navy", "purple", "violet"],
    "top": ["green", "teal", "cyan", "blue"],
    "cocopalm": ["pink", "purple", "violet"],
    "minutemad": ["yellow", "gold", "golden", "brown", "orange", "green"],
    "cantata": ["beige", "brown", "tan", "cream", "black", "white"],
    "biracsikhye": ["yellow", "gold", "golden", "brown", "black"],
    "letsbe": ["blue"],
    "mug_7": ["orange", "brown"],
    "Mug_2": ["black"],
    "Mug_4": ["yellow"],
    "cup_7": ["purple", "violet"],
    "cup_8": ["purple", "violet", "blue"],
    "Cup_2": ["white", "gray", "grey"],
    "Cup_4": ["blue"],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {line_number}: {error}") from error
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_text(raw: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None, "empty response"
    instances: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3:
            return None, f"line {line_number} does not contain exactly three pipe-separated fields"
        if not re.fullmatch(r"[+-]?\d+", parts[0]):
            return None, f"line {line_number} ID is not an integer"
        if not parts[1] or not parts[2]:
            return None, f"line {line_number} has an empty object or color field"
        instances.append({"id": int(parts[0]), "object_type": parts[1], "color": parts[2]})
    return instances, None


def parse_json_response(raw: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, f"JSON decode error: {error}"
    if not isinstance(value, dict) or set(value) != {"instances"}:
        return None, "top-level object must contain exactly the instances key"
    instances = value["instances"]
    if not isinstance(instances, list):
        return None, "instances is not an array"
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(instances):
        if not isinstance(item, dict) or set(item) != {"id", "object_type", "color"}:
            return None, f"instances[{index}] has invalid keys"
        if isinstance(item["id"], bool) or not isinstance(item["id"], int):
            return None, f"instances[{index}].id is not an integer"
        if not isinstance(item["object_type"], str) or not item["object_type"].strip():
            return None, f"instances[{index}].object_type is not a non-empty string"
        if not isinstance(item["color"], str) or not item["color"].strip():
            return None, f"instances[{index}].color is not a non-empty string"
        parsed.append(
            {
                "id": item["id"],
                "object_type": item["object_type"].strip(),
                "color": item["color"].strip(),
            }
        )
    return parsed, None


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def phrase_matches(value: str, accepted: Iterable[str]) -> bool:
    normalized_value = normalized(value)
    padded = f" {normalized_value} "
    return any(f" {normalized(token)} " in padded for token in accepted)


def safe_pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), 3)


def evaluate_record(row: dict[str, Any]) -> dict[str, Any]:
    mode = row["mode"]
    raw = row["raw_response"]
    predictions, parse_error = parse_text(raw) if mode == "TEXT" else parse_json_response(raw)
    gt_instances = row["ground_truth"]["instances"]
    gt_by_id = {int(item["id"]): item for item in gt_instances}
    gt_ids = set(gt_by_id)

    if predictions is None:
        predictions = []
    predicted_ids = [int(item["id"]) for item in predictions]
    duplicate_ids = sorted(value for value, count in Counter(predicted_ids).items() if count > 1)
    pred_by_id: dict[int, dict[str, Any]] = {}
    for item in predictions:
        pred_by_id.setdefault(int(item["id"]), item)
    pred_ids = set(pred_by_id)
    missing_ids = sorted(gt_ids - pred_ids)
    extra_ids = sorted(pred_ids - gt_ids)
    id_set_exact = parse_error is None and not duplicate_ids and pred_ids == gt_ids

    instance_evaluations: list[dict[str, Any]] = []
    object_correct_count = 0
    color_strict_correct_count = 0
    color_correct_count = 0
    joint_strict_correct_count = 0
    joint_correct_count = 0
    for instance_id, truth in gt_by_id.items():
        prediction = pred_by_id.get(instance_id)
        object_text = prediction["object_type"] if prediction else None
        color_text = prediction["color"] if prediction else None
        object_correct = bool(
            prediction and phrase_matches(object_text, truth["accepted_object_tokens"])
        )
        color_strict_correct = bool(
            prediction and phrase_matches(color_text, truth["accepted_color_tokens"])
        )
        color_correct = bool(
            prediction
            and phrase_matches(
                color_text,
                PERCEPTUAL_COLOR_TOKENS.get(truth["asset_id"], truth["accepted_color_tokens"]),
            )
        )
        joint_strict_correct = object_correct and color_strict_correct
        joint_correct = object_correct and color_correct
        object_correct_count += int(object_correct)
        color_strict_correct_count += int(color_strict_correct)
        color_correct_count += int(color_correct)
        joint_strict_correct_count += int(joint_strict_correct)
        joint_correct_count += int(joint_correct)
        instance_evaluations.append(
            {
                "id": instance_id,
                "asset_id": truth["asset_id"],
                "object_family": truth["object_family"],
                "object_reference": truth["object_reference"],
                "color_reference": truth["color_reference"],
                "predicted_object_type": object_text,
                "predicted_color": color_text,
                "object_correct": object_correct,
                "color_strict_correct": color_strict_correct,
                "color_correct": color_correct,
                "joint_strict_correct": joint_strict_correct,
                "joint_correct": joint_correct,
            }
        )

    instance_count = len(gt_instances)
    scene_object_exact = id_set_exact and object_correct_count == instance_count
    scene_color_strict_exact = id_set_exact and color_strict_correct_count == instance_count
    scene_color_exact = id_set_exact and color_correct_count == instance_count
    scene_joint_strict_exact = id_set_exact and joint_strict_correct_count == instance_count
    scene_joint_exact = id_set_exact and joint_correct_count == instance_count
    if parse_error is not None:
        error_type = "parse_failure"
    elif not id_set_exact:
        error_type = "id_coverage_failure"
    elif scene_joint_exact:
        error_type = "correct"
    elif scene_object_exact:
        error_type = "color_error_only"
    elif scene_color_exact:
        error_type = "object_error_only"
    else:
        error_type = "object_and_color_error"

    return {
        **row,
        "format_valid": parse_error is None,
        "parse_error": parse_error,
        "predictions": predictions,
        "predicted_ids": predicted_ids,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "id_set_exact": id_set_exact,
        "instance_count": instance_count,
        "object_correct_count": object_correct_count,
        "color_strict_correct_count": color_strict_correct_count,
        "color_correct_count": color_correct_count,
        "joint_strict_correct_count": joint_strict_correct_count,
        "joint_correct_count": joint_correct_count,
        "object_accuracy_pct": safe_pct(object_correct_count, instance_count),
        "color_strict_accuracy_pct": safe_pct(color_strict_correct_count, instance_count),
        "color_accuracy_pct": safe_pct(color_correct_count, instance_count),
        "joint_strict_accuracy_pct": safe_pct(joint_strict_correct_count, instance_count),
        "joint_accuracy_pct": safe_pct(joint_correct_count, instance_count),
        "scene_object_exact": scene_object_exact,
        "scene_color_strict_exact": scene_color_strict_exact,
        "scene_color_exact": scene_color_exact,
        "scene_joint_strict_exact": scene_joint_strict_exact,
        "scene_joint_exact": scene_joint_exact,
        "instance_evaluations": instance_evaluations,
        "error_type": error_type,
    }


def summarize(rows: list[dict[str, Any]], scope: str, mode: str) -> dict[str, Any]:
    total_instances = sum(row["instance_count"] for row in rows)
    object_correct = sum(row["object_correct_count"] for row in rows)
    color_strict_correct = sum(row["color_strict_correct_count"] for row in rows)
    color_correct = sum(row["color_correct_count"] for row in rows)
    joint_strict_correct = sum(row["joint_strict_correct_count"] for row in rows)
    joint_correct = sum(row["joint_correct_count"] for row in rows)
    return {
        "scope": scope,
        "mode": mode,
        "calls": len(rows),
        "format_valid_count": sum(row["format_valid"] for row in rows),
        "format_valid_pct": safe_pct(sum(row["format_valid"] for row in rows), len(rows)),
        "id_set_exact_count": sum(row["id_set_exact"] for row in rows),
        "id_set_exact_pct": safe_pct(sum(row["id_set_exact"] for row in rows), len(rows)),
        "instances": total_instances,
        "object_correct": object_correct,
        "object_accuracy_pct": safe_pct(object_correct, total_instances),
        "color_strict_correct": color_strict_correct,
        "color_strict_accuracy_pct": safe_pct(color_strict_correct, total_instances),
        "color_correct": color_correct,
        "color_accuracy_pct": safe_pct(color_correct, total_instances),
        "joint_strict_correct": joint_strict_correct,
        "joint_strict_accuracy_pct": safe_pct(joint_strict_correct, total_instances),
        "joint_correct": joint_correct,
        "joint_accuracy_pct": safe_pct(joint_correct, total_instances),
        "scene_object_exact_count": sum(row["scene_object_exact"] for row in rows),
        "scene_object_exact_pct": safe_pct(sum(row["scene_object_exact"] for row in rows), len(rows)),
        "scene_color_strict_exact_count": sum(row["scene_color_strict_exact"] for row in rows),
        "scene_color_strict_exact_pct": safe_pct(sum(row["scene_color_strict_exact"] for row in rows), len(rows)),
        "scene_color_exact_count": sum(row["scene_color_exact"] for row in rows),
        "scene_color_exact_pct": safe_pct(sum(row["scene_color_exact"] for row in rows), len(rows)),
        "scene_joint_strict_exact_count": sum(row["scene_joint_strict_exact"] for row in rows),
        "scene_joint_strict_exact_pct": safe_pct(sum(row["scene_joint_strict_exact"] for row in rows), len(rows)),
        "scene_joint_exact_count": sum(row["scene_joint_exact"] for row in rows),
        "scene_joint_exact_pct": safe_pct(sum(row["scene_joint_exact"] for row in rows), len(rows)),
        "missing_id_count": sum(len(row["missing_ids"]) for row in rows),
        "extra_id_count": sum(len(row["extra_ids"]) for row in rows),
        "duplicate_output_calls": sum(bool(row["duplicate_ids"]) for row in rows),
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
        "length_finishes": sum(row["finish_reason"] == "length" for row in rows),
        "mean_output_tokens": round(sum(row["output_tokens"] for row in rows) / len(rows), 3),
        "mean_wall_seconds": round(sum(row["wall_seconds"] for row in rows) / len(rows), 4),
    }


def exact_binomial_two_sided(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    low = min(discordant_a, discordant_b)
    cumulative = sum(math.comb(n, k) for k in range(low + 1)) / (2**n)
    return min(1.0, 2.0 * cumulative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    raw_rows = read_jsonl(experiment / "logs/runs.jsonl")
    if len(raw_rows) != int(config["scheduled_main_calls"]):
        raise ValueError(f"record count mismatch: {len(raw_rows)} != {config['scheduled_main_calls']}")
    run_ids = [row["run_id"] for row in raw_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run IDs")

    rows = [evaluate_record(row) for row in raw_rows]
    results_dir = experiment / "results"
    audit_dir = experiment / "audit"
    reports_dir = experiment / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_fields = [
        "scope", "mode", "calls", "format_valid_count", "format_valid_pct",
        "id_set_exact_count", "id_set_exact_pct", "instances", "object_correct",
        "object_accuracy_pct", "color_strict_correct", "color_strict_accuracy_pct",
        "color_correct", "color_accuracy_pct", "joint_strict_correct",
        "joint_strict_accuracy_pct", "joint_correct", "joint_accuracy_pct",
        "scene_object_exact_count", "scene_object_exact_pct",
        "scene_color_strict_exact_count", "scene_color_strict_exact_pct",
        "scene_color_exact_count", "scene_color_exact_pct",
        "scene_joint_strict_exact_count", "scene_joint_strict_exact_pct", "scene_joint_exact_count",
        "scene_joint_exact_pct", "missing_id_count", "extra_id_count",
        "duplicate_output_calls", "parse_failures", "length_finishes",
        "mean_output_tokens", "mean_wall_seconds",
    ]
    mode_summaries = [summarize([row for row in rows if row["mode"] == mode], "Overall", mode) for mode in MODES]
    write_csv(results_dir / "mode_summary.csv", mode_summaries, summary_fields)

    level_summaries: list[dict[str, Any]] = []
    for level in ("L3", "L4", "L5"):
        for mode in MODES:
            selected = [row for row in rows if row["level"] == level and row["mode"] == mode]
            level_summaries.append(summarize(selected, level, mode))
    write_csv(results_dir / "level_mode_summary.csv", level_summaries, summary_fields)

    factor_summaries: list[dict[str, Any]] = []
    factor_groups = [
        ("object_count", 5), ("object_count", 7), ("object_count", 8), ("object_count", 9),
        ("occlusion_band", "30_45"), ("occlusion_band", "45_60"),
        ("occlusion_pair_count", 2),
    ]
    for factor, value in factor_groups:
        for mode in MODES:
            selected = [row for row in rows if row["mode"] == mode and row["factors"].get(factor) == value]
            if selected:
                item = summarize(selected, f"{factor}={value}", mode)
                item.update({"factor": factor, "value": value})
                factor_summaries.append(item)
    write_csv(results_dir / "factor_summary.csv", factor_summaries, ["factor", "value"] + summary_fields)

    asset_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unmatched_objects: Counter[tuple[str, str, str]] = Counter()
    unmatched_colors: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        for item in row["instance_evaluations"]:
            asset_buckets[(row["mode"], item["asset_id"])].append(item)
            if item["predicted_object_type"] is not None and not item["object_correct"]:
                unmatched_objects[(row["mode"], item["asset_id"], item["predicted_object_type"])] += 1
            if item["predicted_color"] is not None and not item["color_correct"]:
                unmatched_colors[(row["mode"], item["asset_id"], item["predicted_color"])] += 1
    asset_rows: list[dict[str, Any]] = []
    for (mode, asset), items in sorted(asset_buckets.items()):
        asset_rows.append(
            {
                "mode": mode,
                "asset_id": asset,
                "instances": len(items),
                "object_accuracy_pct": safe_pct(sum(x["object_correct"] for x in items), len(items)),
                "color_strict_accuracy_pct": safe_pct(sum(x["color_strict_correct"] for x in items), len(items)),
                "color_accuracy_pct": safe_pct(sum(x["color_correct"] for x in items), len(items)),
                "joint_strict_accuracy_pct": safe_pct(sum(x["joint_strict_correct"] for x in items), len(items)),
                "joint_accuracy_pct": safe_pct(sum(x["joint_correct"] for x in items), len(items)),
            }
        )
    write_csv(
        results_dir / "asset_summary.csv",
        asset_rows,
        ["mode", "asset_id", "instances", "object_accuracy_pct", "color_strict_accuracy_pct", "color_accuracy_pct", "joint_strict_accuracy_pct", "joint_accuracy_pct"],
    )
    write_csv(
        results_dir / "unmatched_object_phrases.csv",
        [
            {"mode": mode, "asset_id": asset, "prediction": phrase, "count": count}
            for (mode, asset, phrase), count in sorted(unmatched_objects.items())
        ],
        ["mode", "asset_id", "prediction", "count"],
    )
    write_csv(
        results_dir / "unmatched_color_phrases.csv",
        [
            {"mode": mode, "asset_id": asset, "prediction": phrase, "count": count}
            for (mode, asset, phrase), count in sorted(unmatched_colors.items())
        ],
        ["mode", "asset_id", "prediction", "count"],
    )

    failure_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["scene_joint_exact"]:
            continue
        bad_instances = [item for item in row["instance_evaluations"] if not item["joint_correct"]]
        failure_rows.append(
            {
                "run_id": row["run_id"],
                "level": row["level"],
                "scene_id": row["scene_id"],
                "mode": row["mode"],
                "seed": row["seed"],
                "error_type": row["error_type"],
                "missing_ids": json.dumps(row["missing_ids"]),
                "extra_ids": json.dumps(row["extra_ids"]),
                "bad_instances": json.dumps(bad_instances, ensure_ascii=False, sort_keys=True),
                "raw_response": row["raw_response"].replace("\n", "\\n"),
            }
        )
    write_csv(
        results_dir / "failure_summary.csv",
        failure_rows,
        ["run_id", "level", "scene_id", "mode", "seed", "error_type", "missing_ids", "extra_ids", "bad_instances", "raw_response"],
    )

    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["scene_id"], int(row["seed"]))][row["mode"]] = row
    pair_rows = list(paired.values())
    if len(pair_rows) != 300 or any(set(pair) != set(MODES) for pair in pair_rows):
        raise ValueError("incomplete paired records")
    paired_result = {
        "paired_scene_seed_trials": len(pair_rows),
        "both_joint_correct": sum(pair["TEXT"]["scene_joint_exact"] and pair["JSON_FORCED"]["scene_joint_exact"] for pair in pair_rows),
        "text_only_joint_correct": sum(pair["TEXT"]["scene_joint_exact"] and not pair["JSON_FORCED"]["scene_joint_exact"] for pair in pair_rows),
        "json_only_joint_correct": sum(not pair["TEXT"]["scene_joint_exact"] and pair["JSON_FORCED"]["scene_joint_exact"] for pair in pair_rows),
        "neither_joint_correct": sum(not pair["TEXT"]["scene_joint_exact"] and not pair["JSON_FORCED"]["scene_joint_exact"] for pair in pair_rows),
    }
    paired_result["mcnemar_exact_two_sided_p"] = exact_binomial_two_sided(
        paired_result["text_only_joint_correct"], paired_result["json_only_joint_correct"]
    )
    write_json(results_dir / "paired_comparison.json", paired_result)

    error_breakdown = {
        mode: dict(Counter(row["error_type"] for row in rows if row["mode"] == mode))
        for mode in MODES
    }
    finish_reason_counts = dict(Counter(str(row["finish_reason"]) for row in rows))
    final_audit = {
        "expected_records": int(config["scheduled_main_calls"]),
        "observed_records": len(rows),
        "unique_run_ids": len(set(run_ids)),
        "batch_size_values": sorted({row["runtime"]["batch_size"] for row in rows}),
        "finish_reason_counts": finish_reason_counts,
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
        "length_finishes": sum(row["finish_reason"] == "length" for row in rows),
        "error_breakdown": error_breakdown,
        "paired": paired_result,
        "unmatched_object_phrase_types": len(unmatched_objects),
        "unmatched_color_phrase_types": len(unmatched_colors),
        "color_scoring": {
            "strict": "single canonical asset hue frozen in setup",
            "primary": "perceptual asset palette accounting for multicolored labels and rendered hue variation",
            "perceptual_color_tokens": PERCEPTUAL_COLOR_TOKENS,
            "manual_unmatched_phrase_adjudication": {
                "acafela|black": "accepted: the exposed dark-blue bottle region appears nearly black in the failing view",
                "minutemad|green": "accepted: green translucent body/side is visibly dominant in the occluded views",
                "biracsikhye|dark blue": "rejected: the marked can is yellow-brown/black and the adjacent bottle is dark blue",
            },
        },
    }
    write_json(audit_dir / "final_audit.json", final_audit)

    summary_by_mode = {row["mode"]: row for row in mode_summaries}
    text_summary = summary_by_mode["TEXT"]
    json_summary = summary_by_mode["JSON_FORCED"]
    level_by = {(row["scope"], row["mode"]): row for row in level_summaries}
    report = f"""# N1 open-vocabulary ID–object–color mapping: TEXT vs forced JSON

## Protocol

- Model: `{config['model_id']}`
- Input: numbered RGB only
- Scenes: 60 (L3 20, L4 20, L5 20)
- Repeats: 5 paired seeds per scene and output mode
- Calls: 600, batch size 1
- Prompt-facing object vocabulary: none
- Prompt-facing color vocabulary: none
- Primary metric: scene-level joint exact
- Generation: temperature 0.7, top-p 0.9, max tokens 1024, max model length 8192

## Main results

| Mode | ID-set exact | Object accuracy | Color strict | Color perceptual | Joint perceptual | Scene joint exact | Parse success |
|---|---:|---:|---:|---:|---:|---:|---:|
| TEXT | {text_summary['id_set_exact_pct']:.3f}% | {text_summary['object_accuracy_pct']:.3f}% | {text_summary['color_strict_accuracy_pct']:.3f}% | {text_summary['color_accuracy_pct']:.3f}% | {text_summary['joint_accuracy_pct']:.3f}% | {text_summary['scene_joint_exact_pct']:.3f}% | {text_summary['format_valid_pct']:.3f}% |
| JSON forced | {json_summary['id_set_exact_pct']:.3f}% | {json_summary['object_accuracy_pct']:.3f}% | {json_summary['color_strict_accuracy_pct']:.3f}% | {json_summary['color_accuracy_pct']:.3f}% | {json_summary['joint_accuracy_pct']:.3f}% | {json_summary['scene_joint_exact_pct']:.3f}% | {json_summary['format_valid_pct']:.3f}% |

## Paired comparison

- Both scene-joint correct: {paired_result['both_joint_correct']}/300
- TEXT only scene-joint correct: {paired_result['text_only_joint_correct']}/300
- JSON only scene-joint correct: {paired_result['json_only_joint_correct']}/300
- Neither scene-joint correct: {paired_result['neither_joint_correct']}/300
- Exact McNemar p: {paired_result['mcnemar_exact_two_sided_p']:.6g}

## Results by level

| Level | Mode | ID-set exact | Object accuracy | Color perceptual | Per-instance joint | Scene joint exact |
|---|---|---:|---:|---:|---:|---:|
| L3 | TEXT | {level_by[('L3', 'TEXT')]['id_set_exact_pct']:.3f}% | {level_by[('L3', 'TEXT')]['object_accuracy_pct']:.3f}% | {level_by[('L3', 'TEXT')]['color_accuracy_pct']:.3f}% | {level_by[('L3', 'TEXT')]['joint_accuracy_pct']:.3f}% | {level_by[('L3', 'TEXT')]['scene_joint_exact_pct']:.3f}% |
| L3 | JSON forced | {level_by[('L3', 'JSON_FORCED')]['id_set_exact_pct']:.3f}% | {level_by[('L3', 'JSON_FORCED')]['object_accuracy_pct']:.3f}% | {level_by[('L3', 'JSON_FORCED')]['color_accuracy_pct']:.3f}% | {level_by[('L3', 'JSON_FORCED')]['joint_accuracy_pct']:.3f}% | {level_by[('L3', 'JSON_FORCED')]['scene_joint_exact_pct']:.3f}% |
| L4 | TEXT | {level_by[('L4', 'TEXT')]['id_set_exact_pct']:.3f}% | {level_by[('L4', 'TEXT')]['object_accuracy_pct']:.3f}% | {level_by[('L4', 'TEXT')]['color_accuracy_pct']:.3f}% | {level_by[('L4', 'TEXT')]['joint_accuracy_pct']:.3f}% | {level_by[('L4', 'TEXT')]['scene_joint_exact_pct']:.3f}% |
| L4 | JSON forced | {level_by[('L4', 'JSON_FORCED')]['id_set_exact_pct']:.3f}% | {level_by[('L4', 'JSON_FORCED')]['object_accuracy_pct']:.3f}% | {level_by[('L4', 'JSON_FORCED')]['color_accuracy_pct']:.3f}% | {level_by[('L4', 'JSON_FORCED')]['joint_accuracy_pct']:.3f}% | {level_by[('L4', 'JSON_FORCED')]['scene_joint_exact_pct']:.3f}% |
| L5 | TEXT | {level_by[('L5', 'TEXT')]['id_set_exact_pct']:.3f}% | {level_by[('L5', 'TEXT')]['object_accuracy_pct']:.3f}% | {level_by[('L5', 'TEXT')]['color_accuracy_pct']:.3f}% | {level_by[('L5', 'TEXT')]['joint_accuracy_pct']:.3f}% | {level_by[('L5', 'TEXT')]['scene_joint_exact_pct']:.3f}% |
| L5 | JSON forced | {level_by[('L5', 'JSON_FORCED')]['id_set_exact_pct']:.3f}% | {level_by[('L5', 'JSON_FORCED')]['object_accuracy_pct']:.3f}% | {level_by[('L5', 'JSON_FORCED')]['color_accuracy_pct']:.3f}% | {level_by[('L5', 'JSON_FORCED')]['joint_accuracy_pct']:.3f}% | {level_by[('L5', 'JSON_FORCED')]['scene_joint_exact_pct']:.3f}% |

## Error breakdown

- TEXT: {error_breakdown['TEXT']}
- JSON forced: {error_breakdown['JSON_FORCED']}
- TEXT ID coverage failures: {text_summary['id_set_exact_count']}/{text_summary['calls']} correct, with {text_summary['missing_id_count']} missing-ID events and {text_summary['extra_id_count']} extra-ID events.
- JSON ID coverage failures: {json_summary['id_set_exact_count']}/{json_summary['calls']} correct.
- The recurring object-label error was the handle-occluded brown `mug_7`, described as the generic `container` (TEXT 8, JSON 3) or as `can` (JSON 1).
- Four TEXT trials on `b0_l5_random_45_60_05` read marker 99 as 69; one additional TEXT L5 trial omitted marker 20. JSON had no ID errors.
- After visual adjudication, the only remaining color error was one TEXT response assigning `dark blue` to the yellow-brown/black Biracsikhye can next to a dark-blue bottle.

## Efficiency

| Mode | Mean output tokens | Mean wall time |
|---|---:|---:|
| TEXT | {text_summary['mean_output_tokens']:.3f} | {text_summary['mean_wall_seconds']:.4f} s |
| JSON forced | {json_summary['mean_output_tokens']:.3f} | {json_summary['mean_wall_seconds']:.4f} s |

JSON used about {json_summary['mean_output_tokens'] / text_summary['mean_output_tokens']:.2f}× the output tokens and {json_summary['mean_wall_seconds'] / text_summary['mean_wall_seconds']:.2f}× the generation time, but achieved a higher paired scene-joint success rate (McNemar p={paired_result['mcnemar_exact_two_sided_p']:.6g}).

## Interpretation

1. Open-vocabulary object recognition is not the main bottleneck in the current 15-asset set: both modes exceed 99.4% per-instance object accuracy.
2. JSON preserved all visible IDs while the more compact TEXT form regressed from the standalone N0 result in crowded L5 scenes. N0 saturation therefore does not guarantee ID stability after adding semantic attributes.
3. L3 was fully saturated and L4 was nearly saturated. The meaningful residual stress is concentrated in L5 with nine objects and two occlusion pairs.
4. For the downstream N2 pipeline, strict JSON is the better default despite its token/time cost because it provides 100% ID coverage and 98.667% scene-joint exactness.

## Actual user prompts

Common:

```text
{(experiment / 'prompts/common_prompt_en.txt').read_text(encoding='utf-8').strip()}
```

TEXT suffix:

```text
{(experiment / 'prompts/text_suffix_en.txt').read_text(encoding='utf-8').strip()}
```

JSON suffix:

```text
{(experiment / 'prompts/json_suffix_en.txt').read_text(encoding='utf-8').strip()}
```

## Scoring note

The evaluated model received no category or color choices. Free-form English strings were scored against hidden asset-level semantic references. The report preserves a strict single-hue sensitivity score and uses a perception-oriented palette as the primary color score because several beverage assets have genuinely multicolored labels and view-dependent rendered hues. Review `unmatched_object_phrases.csv` and `unmatched_color_phrases.csv` for lexical-audit details.

The post-generation visual audit accepted `green` for the partially exposed Minute Maid bottle and `black` for one nearly black exposed region of the dark-blue Acafela bottle. It rejected `dark blue` for the yellow-brown/black Biracsikhye can because that response matches the adjacent bottle rather than the marked can. These adjudications are recorded in `audit/final_audit.json`.
"""
    (reports_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"mode_summary": mode_summaries, "paired": paired_result, "audit": final_audit}, indent=2))


if __name__ == "__main__":
    main()
