#!/usr/bin/env python3
"""Analyze the N1 explicit-visible-ID 3x2x2 ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import analyze_b0_n1_open_vocab_object_color as base


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/b0_n1_explicit_visible_ids_ablation_60scenes_20260827"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_enumerated_text(raw: str) -> tuple[list[int] | None, list[dict[str, Any]] | None, str | None]:
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None, None, "empty response"
    match = re.fullmatch(r"visible_ids\s*:\s*\[(.*?)\]", lines[0], flags=re.IGNORECASE)
    if not match:
        return None, None, "first line is not a visible_ids list"
    body = match.group(1).strip()
    visible_ids: list[int] = []
    if body:
        for index, value in enumerate(body.split(","), 1):
            value = value.strip()
            if not re.fullmatch(r"[+-]?\d+", value):
                return None, None, f"visible_ids item {index} is not an integer"
            visible_ids.append(int(value))
    instance_text = "\n".join(lines[1:])
    instances, error = base.parse_text(instance_text)
    if error:
        return visible_ids, None, error
    return visible_ids, instances, None


def parse_enumerated_json(raw: str) -> tuple[list[int] | None, list[dict[str, Any]] | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, None, f"JSON decode error: {error}"
    if not isinstance(value, dict) or set(value) != {"visible_ids", "instances"}:
        return None, None, "top-level object must contain exactly visible_ids and instances"
    visible_ids = value["visible_ids"]
    if not isinstance(visible_ids, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in visible_ids):
        return None, None, "visible_ids must be an array of integers"
    instances_only = json.dumps({"instances": value["instances"]}, ensure_ascii=False)
    instances, error = base.parse_json_response(instances_only)
    if error:
        return visible_ids, None, error
    return visible_ids, instances, None


def evaluate_record(row: dict[str, Any]) -> dict[str, Any]:
    enumeration = bool(row["enumeration"])
    if not enumeration:
        scored = base.evaluate_record(row)
        return {
            **scored,
            "visible_ids": None,
            "visible_duplicate_ids": [],
            "visible_missing_ids": [],
            "visible_extra_ids": [],
            "visible_id_set_exact": None,
            "cross_field_consistency_exact": None,
            "full_output_joint_exact": scored["scene_joint_exact"],
        }

    if row["mode"] == "TEXT":
        visible_ids, predictions, parse_error = parse_enumerated_text(row["raw_response"])
        canonical = "\n".join(
            f"{item['id']} | {item['object_type']} | {item['color']}" for item in (predictions or [])
        )
    else:
        visible_ids, predictions, parse_error = parse_enumerated_json(row["raw_response"])
        canonical = json.dumps({"instances": predictions or []}, ensure_ascii=False)

    surrogate = {**row, "raw_response": canonical}
    scored = base.evaluate_record(surrogate)
    if parse_error is not None:
        scored["format_valid"] = False
        scored["parse_error"] = parse_error
        scored["error_type"] = "parse_failure"
    scored["raw_response"] = row["raw_response"]
    visible_ids = visible_ids or []
    visible_duplicates = sorted(value for value, count in Counter(visible_ids).items() if count > 1)
    visible_set = set(visible_ids)
    gt_set = {int(item["id"]) for item in row["ground_truth"]["instances"]}
    instance_ids = scored["predicted_ids"]
    instance_duplicates = scored["duplicate_ids"]
    visible_exact = parse_error is None and not visible_duplicates and visible_set == gt_set
    cross_exact = (
        parse_error is None
        and not visible_duplicates
        and not instance_duplicates
        and visible_set == set(instance_ids)
    )
    full_exact = bool(scored["scene_joint_exact"] and visible_exact and cross_exact)
    if parse_error is None and not full_exact and scored["error_type"] == "correct":
        scored["error_type"] = "visible_id_or_cross_field_failure"
    return {
        **scored,
        "visible_ids": visible_ids,
        "visible_duplicate_ids": visible_duplicates,
        "visible_missing_ids": sorted(gt_set - visible_set),
        "visible_extra_ids": sorted(visible_set - gt_set),
        "visible_id_set_exact": visible_exact,
        "cross_field_consistency_exact": cross_exact,
        "full_output_joint_exact": full_exact,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = base.summarize(rows, "Overall", "condition")
    enumerated = [row for row in rows if row["enumeration"]]
    return {
        **{key: value for key, value in core.items() if key not in {"scope", "mode"}},
        "visible_id_set_exact_count": sum(row["visible_id_set_exact"] is True for row in enumerated),
        "visible_id_set_exact_pct": base.safe_pct(sum(row["visible_id_set_exact"] is True for row in enumerated), len(enumerated)) if enumerated else "",
        "cross_field_consistency_exact_count": sum(row["cross_field_consistency_exact"] is True for row in enumerated),
        "cross_field_consistency_exact_pct": base.safe_pct(sum(row["cross_field_consistency_exact"] is True for row in enumerated), len(enumerated)) if enumerated else "",
        "full_output_joint_exact_count": sum(row["full_output_joint_exact"] for row in rows),
        "full_output_joint_exact_pct": base.safe_pct(sum(row["full_output_joint_exact"] for row in rows), len(rows)),
    }


def paired_exact(rows: list[dict[str, Any]], left: str, right: str, metric: str, contrast: str) -> dict[str, Any]:
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["condition_id"] in {left, right}:
            pairs[(row["scene_id"], int(row["seed"]))][row["condition_id"]] = row
    if len(pairs) != 300 or any(set(pair) != {left, right} for pair in pairs.values()):
        raise ValueError(f"incomplete pairing: {left} vs {right}")
    both = left_only = right_only = neither = 0
    for pair in pairs.values():
        left_ok = bool(pair[left][metric])
        right_ok = bool(pair[right][metric])
        both += int(left_ok and right_ok)
        left_only += int(left_ok and not right_ok)
        right_only += int(not left_ok and right_ok)
        neither += int(not left_ok and not right_ok)
    return {
        "contrast": contrast,
        "metric": metric,
        "left_condition": left,
        "right_condition": right,
        "paired_trials": len(pairs),
        "both_correct": both,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "neither_correct": neither,
        "right_minus_left_percentage_points": round(100.0 * (right_only - left_only) / len(pairs), 3),
        "mcnemar_exact_two_sided_p": base.exact_binomial_two_sided(left_only, right_only),
    }


def plot_main_effects(experiment: Path, main_effects: list[dict[str, Any]]) -> None:
    panels = [
        ("temperature", "Temperature", ["0.3", "0.5", "0.7"]),
        ("enumeration", "Explicit visible-ID enumeration", ["False", "True"]),
        ("output_format", "Output format", ["TEXT", "JSON_FORCED"]),
    ]
    lookup = {(row["factor"], str(row["value"])): float(row["full_output_joint_exact_pct"]) for row in main_effects}
    grand = sum(row["full_output_joint_exact_count"] for row in main_effects if row["factor"] == "temperature") / 36.0
    label_maps = {
        "enumeration": {"False": "Off", "True": "On"},
        "output_format": {"TEXT": "TEXT", "JSON_FORCED": "JSON"},
    }
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharey=True)
    values_all = list(lookup.values())
    low, high = min(values_all) - 1.0, max(values_all) + 1.0
    for axis, (factor, title, order), color in zip(axes, panels, colors):
        values = [lookup[(factor, item)] for item in order]
        labels = [label_maps.get(factor, {}).get(item, item) for item in order]
        positions = list(range(len(order)))
        axis.plot(positions, values, marker="o", linewidth=2.4, markersize=7, color=color)
        axis.axhline(grand, color="#777777", linestyle="--", linewidth=1.2)
        axis.set_xticks(positions, labels)
        axis.set_title(title, fontsize=11, weight="bold")
        axis.grid(axis="y", alpha=0.22)
        for x, value in zip(positions, values):
            axis.annotate(f"{value:.3f}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
    axes[0].set_ylabel("Full-output joint exact (%)")
    axes[0].set_ylim(low, high)
    fig.suptitle("N1 main effects — explicit visible-ID enumeration", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, f"Dashed line: grand mean = {grand:.3f}%", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output = experiment / "figures/main_effect_full_output_joint_exact.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_color_perceptual_main_effects(
    experiment: Path, main_effects: list[dict[str, Any]]
) -> None:
    panels = [
        ("temperature", "Temperature", ["0.3", "0.5", "0.7"]),
        ("enumeration", "Explicit visible-ID enumeration", ["False", "True"]),
        ("output_format", "Output format", ["TEXT", "JSON_FORCED"]),
    ]
    lookup = {
        (row["factor"], str(row["value"])): float(row["color_accuracy_pct"])
        for row in main_effects
    }
    temperature_rows = [row for row in main_effects if row["factor"] == "temperature"]
    grand = 100.0 * sum(row["color_correct"] for row in temperature_rows) / sum(
        row["instances"] for row in temperature_rows
    )
    label_maps = {
        "enumeration": {"False": "Off", "True": "On"},
        "output_format": {"TEXT": "TEXT", "JSON_FORCED": "JSON"},
    }
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharey=True)
    values_all = list(lookup.values())
    spread = max(values_all) - min(values_all)
    padding = max(0.02, spread * 0.25)
    low, high = min(values_all) - padding, max(values_all) + padding
    for axis, (factor, title, order), color in zip(axes, panels, colors):
        values = [lookup[(factor, item)] for item in order]
        labels = [label_maps.get(factor, {}).get(item, item) for item in order]
        positions = list(range(len(order)))
        axis.plot(
            positions,
            values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=color,
        )
        axis.axhline(grand, color="#777777", linestyle="--", linewidth=1.2)
        axis.set_xticks(positions, labels)
        axis.set_title(title, fontsize=11, weight="bold")
        axis.grid(axis="y", alpha=0.22)
        for x, value in zip(positions, values):
            axis.annotate(
                f"{value:.3f}",
                (x, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
    axes[0].set_ylabel("Color perceptual accuracy (%)")
    axes[0].set_ylim(low, high)
    fig.suptitle(
        "N1 Color perceptual main effects — explicit visible-ID enumeration",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"Dashed line: grand mean = {grand:.3f}% | Zoomed y-axis",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output = experiment / "figures/main_effect_color_perceptual_accuracy.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = base.read_json(experiment / "config/experiment_config.json")
    raw_rows = base.read_jsonl(experiment / "logs/runs.jsonl")
    if len(raw_rows) != int(config["scheduled_main_calls"]):
        raise ValueError(f"record count mismatch: {len(raw_rows)}")
    if len({row["run_id"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("duplicate run IDs")
    rows = [evaluate_record(row) for row in raw_rows]

    results_dir = experiment / "results"
    reports_dir = experiment / "reports"
    audit_dir = experiment / "audit"
    for directory in (results_dir, reports_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with (results_dir / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    condition_rows: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        selected = [row for row in rows if row["condition_id"] == condition["condition_id"]]
        per_scene = Counter(row["scene_id"] for row in selected if row["full_output_joint_exact"])
        condition_rows.append(
            {
                **condition,
                **summarize(selected),
                "scenes_5of5_full_exact": sum(per_scene[scene["scene_id"]] == 5 for scene in base.read_json(experiment / "config/frozen_scenes.json")),
                "scenes_1to4of5_full_exact": sum(0 < per_scene[scene["scene_id"]] < 5 for scene in base.read_json(experiment / "config/frozen_scenes.json")),
                "scenes_0of5_full_exact": sum(per_scene[scene["scene_id"]] == 0 for scene in base.read_json(experiment / "config/frozen_scenes.json")),
                "error_breakdown": json.dumps(dict(Counter(row["error_type"] for row in selected)), sort_keys=True),
            }
        )
    write_csv(results_dir / "condition_summary.csv", condition_rows)

    level_rows: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        for level in ("L3", "L4", "L5"):
            selected = [row for row in rows if row["condition_id"] == condition["condition_id"] and row["level"] == level]
            level_rows.append({**condition, "level": level, **summarize(selected)})
    write_csv(results_dir / "level_condition_summary.csv", level_rows)

    factor_values = {
        "temperature": config["design"]["temperature"],
        "enumeration": [False, True],
        "output_format": ["TEXT", "JSON_FORCED"],
    }
    main_effect_rows: list[dict[str, Any]] = []
    for factor, values in factor_values.items():
        for value in values:
            selected = [row for row in rows if row[factor] == value]
            main_effect_rows.append({"factor": factor, "value": value, **summarize(selected)})
    write_csv(results_dir / "main_effect_summary.csv", main_effect_rows)

    paired_rows: list[dict[str, Any]] = []
    temperatures = [float(x) for x in config["design"]["temperature"]]
    for temperature in temperatures:
        code = f"{int(temperature * 10):02d}"
        for output in ("TEXT", "JSON"):
            paired_rows.append(paired_exact(rows, f"T{code}_DIRECT_{output}", f"T{code}_ENUM_{output}", "full_output_joint_exact", f"enumeration at T={temperature:.1f}, {output}"))
        for prompt in ("DIRECT", "ENUM"):
            paired_rows.append(paired_exact(rows, f"T{code}_{prompt}_TEXT", f"T{code}_{prompt}_JSON", "full_output_joint_exact", f"JSON at T={temperature:.1f}, {prompt}"))
    for left_temp, right_temp in itertools.combinations(temperatures, 2):
        for prompt in ("DIRECT", "ENUM"):
            for output in ("TEXT", "JSON"):
                paired_rows.append(paired_exact(rows, f"T{int(left_temp*10):02d}_{prompt}_{output}", f"T{int(right_temp*10):02d}_{prompt}_{output}", "full_output_joint_exact", f"temperature {left_temp:.1f} to {right_temp:.1f}, {prompt}, {output}"))
    write_csv(results_dir / "paired_comparisons.csv", paired_rows)

    failure_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["full_output_joint_exact"]:
            continue
        failure_rows.append(
            {
                "condition_id": row["condition_id"],
                "temperature": row["temperature"],
                "enumeration": row["enumeration"],
                "output_format": row["output_format"],
                "run_id": row["run_id"],
                "level": row["level"],
                "scene_id": row["scene_id"],
                "seed": row["seed"],
                "error_type": row["error_type"],
                "visible_missing_ids": json.dumps(row["visible_missing_ids"]),
                "visible_extra_ids": json.dumps(row["visible_extra_ids"]),
                "instance_missing_ids": json.dumps(row["missing_ids"]),
                "instance_extra_ids": json.dumps(row["extra_ids"]),
                "cross_field_consistency_exact": row["cross_field_consistency_exact"],
                "bad_instances": json.dumps([item for item in row["instance_evaluations"] if not item["joint_correct"]], ensure_ascii=False, sort_keys=True),
                "raw_response": row["raw_response"].replace("\n", "\\n"),
            }
        )
    write_csv(results_dir / "failures.csv", failure_rows)

    audit = {
        "expected_records": int(config["scheduled_main_calls"]),
        "observed_records": len(rows),
        "unique_run_ids": len({row["run_id"] for row in rows}),
        "condition_counts": dict(Counter(row["condition_id"] for row in rows)),
        "batch_size_values": sorted({row["runtime"]["batch_size"] for row in rows}),
        "seed_values": sorted({row["seed"] for row in rows}),
        "finish_reason_counts": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
        "length_finishes": sum(row["finish_reason"] == "length" for row in rows),
    }
    base.write_json(audit_dir / "final_audit.json", audit)
    plot_main_effects(experiment, main_effect_rows)
    plot_color_perceptual_main_effects(experiment, main_effect_rows)

    best = max(condition_rows, key=lambda row: (row["full_output_joint_exact_pct"], row["joint_accuracy_pct"]))
    condition_lines = []
    for row in condition_rows:
        vis = f"{row['visible_id_set_exact_pct']:.3f}%" if row["enumeration"] else "—"
        cross = f"{row['cross_field_consistency_exact_pct']:.3f}%" if row["enumeration"] else "—"
        condition_lines.append(
            f"| {row['temperature']:.1f} | {'On' if row['enumeration'] else 'Off'} | {'JSON' if row['output_format'] == 'JSON_FORCED' else 'TEXT'} | "
            f"{row['id_set_exact_pct']:.3f}% | {vis} | {cross} | {row['object_accuracy_pct']:.3f}% | {row['color_strict_accuracy_pct']:.3f}% | "
            f"{row['color_accuracy_pct']:.3f}% | {row['scene_joint_exact_pct']:.3f}% | {row['full_output_joint_exact_pct']:.3f}% |"
        )
    main_lines = []
    for row in main_effect_rows:
        main_lines.append(f"| {row['factor']} | {row['value']} | {row['full_output_joint_exact_pct']:.3f}% | {row['scene_joint_exact_pct']:.3f}% |")
    enum_pair_lines = []
    for row in paired_rows[:6:2]:
        enum_pair_lines.append(f"| {row['contrast']} | {row['right_minus_left_percentage_points']:+.3f} pp | {row['left_only_correct']} | {row['right_only_correct']} | {row['mcnemar_exact_two_sided_p']:.6g} |")
    # Include both TEXT and JSON enumeration comparisons (indices 0,2,4 plus 1? paired list order has enum,json,enum,json...).
    enum_pair_lines = [
        f"| {row['contrast']} | {row['right_minus_left_percentage_points']:+.3f} pp | {row['left_only_correct']} | {row['right_only_correct']} | {row['mcnemar_exact_two_sided_p']:.6g} |"
        for row in paired_rows if row["contrast"].startswith("enumeration at")
    ]
    report = f"""# N1 explicit visible-ID enumeration ablation

## Protocol

- Model: `{config['model_id']}`
- Input: 60 frozen numbered-RGB scenes (L3/L4/L5), five paired seeds
- Design: temperature (0.3/0.5/0.7) × explicit visible-ID enumeration (off/on) × output (TEXT/forced JSON)
- Calls: 3,600; batch size 1
- No object/color vocabulary and no array-length or ID-value constraint
- Primary metric: full-output joint exact. For enumeration conditions this requires the visible-ID list, cross-field ID consistency, and every instance's object/color to be correct.

## Condition results

| Temp. | Enumeration | Output | Instance ID-set | Visible-ID set | Cross-field | Object | Color strict | Color perceptual | Instance scene joint | Full-output joint |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(condition_lines)}

Best condition: `{best['condition_id']}` at {best['full_output_joint_exact_pct']:.3f}% full-output joint exact.

## Main effects

| Factor | Level | Full-output joint | Instance scene joint |
|---|---|---:|---:|
{chr(10).join(main_lines)}

## Paired enumeration contrasts

| Contrast | Enumeration minus direct | Direct-only correct | Enumeration-only correct | McNemar p |
|---|---:|---:|---:|---:|
{chr(10).join(enum_pair_lines)}

## Audit

- Records: {audit['observed_records']}/{audit['expected_records']}
- Parse failures: {audit['parse_failures']}
- Length finishes: {audit['length_finishes']}
- Batch-size values: {audit['batch_size_values']}

## Files

- `results/condition_summary.csv`
- `results/level_condition_summary.csv`
- `results/main_effect_summary.csv`
- `results/paired_comparisons.csv`
- `results/failures.csv`
- `results/evaluated_records.jsonl`
- `figures/main_effect_full_output_joint_exact.png`
- `figures/main_effect_color_perceptual_accuracy.png`
"""
    (reports_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"best_condition": best["condition_id"], "best_full_output_joint_exact_pct": best["full_output_joint_exact_pct"], "conditions": condition_rows, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
