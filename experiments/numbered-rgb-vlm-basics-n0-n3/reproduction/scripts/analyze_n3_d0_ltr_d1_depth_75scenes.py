#!/usr/bin/env python3
"""Analyze exact and pairwise accuracy for the N3 D0/D1 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/n3_d0_ltr_d1_depth_75scenes_5seeds_20260828"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def integer_array(value: Any) -> bool:
    return isinstance(value, list) and all(not isinstance(item, bool) and isinstance(item, int) for item in value)


def evaluate(row: dict[str, Any]) -> dict[str, Any]:
    parse_error: str | None = None
    parsed: dict[str, Any] = {}
    try:
        value = json.loads(row["raw_response"])
    except json.JSONDecodeError as error:
        parse_error = f"JSON decode error: {error}"
    else:
        expected_keys = {"visible_ids", row["output_key"]}
        if not isinstance(value, dict) or set(value) != expected_keys:
            parse_error = f"top-level keys must be {sorted(expected_keys)}"
        elif not integer_array(value["visible_ids"]):
            parse_error = "visible_ids must be an integer array"
        elif not integer_array(value[row["output_key"]]):
            parse_error = f"{row['output_key']} must be an integer array"
        else:
            parsed = value

    predicted_visible = parsed.get("visible_ids", [])
    predicted_order = parsed.get(row["output_key"], [])
    expected_visible = [int(value) for value in row["visible_ids"]]
    expected_order = [int(value) for value in row["expected_order"]]
    visible_exact = parse_error is None and predicted_visible == expected_visible
    visible_set_exact = (
        parse_error is None
        and len(predicted_visible) == len(set(predicted_visible))
        and set(predicted_visible) == set(expected_visible)
    )
    order_duplicates = sorted(value for value, count in Counter(predicted_order).items() if count > 1)
    order_missing = sorted(set(expected_visible) - set(predicted_order))
    order_extra = sorted(set(predicted_order) - set(expected_visible))
    permutation_valid = (
        parse_error is None
        and not order_duplicates
        and not order_missing
        and not order_extra
        and len(predicted_order) == len(expected_order)
    )
    order_exact = permutation_valid and predicted_order == expected_order

    predicted_positions: dict[int, int] = {}
    for index, instance_id in enumerate(predicted_order):
        if instance_id not in predicted_positions:
            predicted_positions[instance_id] = index
    pairwise_total = len(expected_order) * (len(expected_order) - 1) // 2
    pairwise_correct = 0
    pairwise_errors: list[dict[str, int]] = []
    for left_index, first in enumerate(expected_order):
        for second in expected_order[left_index + 1:]:
            correct = (
                first in predicted_positions
                and second in predicted_positions
                and predicted_positions[first] < predicted_positions[second]
            )
            pairwise_correct += int(correct)
            if not correct:
                pairwise_errors.append({"expected_before": first, "expected_after": second})

    if parse_error is not None:
        error_type = "parse_failure"
    elif not visible_exact:
        error_type = "visible_ids_failure"
    elif not permutation_valid:
        error_type = "output_permutation_failure"
    elif not order_exact:
        error_type = "spatial_order_failure"
    else:
        error_type = "correct"

    return {
        **row,
        "parsed_response": parsed,
        "predicted_visible_ids": predicted_visible,
        "predicted_order": predicted_order,
        "parse_error": parse_error,
        "format_valid": parse_error is None,
        "visible_ids_exact": visible_exact,
        "visible_id_set_exact": visible_set_exact,
        "order_permutation_valid": permutation_valid,
        "order_duplicates": order_duplicates,
        "order_missing": order_missing,
        "order_extra": order_extra,
        "order_exact": order_exact,
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "pairwise_errors": pairwise_errors,
        "error_type": error_type,
    }


def summarize(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    pairwise_correct = sum(row["pairwise_correct"] for row in rows)
    pairwise_total = sum(row["pairwise_total"] for row in rows)
    return {
        "scope": scope,
        "calls": len(rows),
        "scenes": len(grouped),
        "order_exact_count": sum(row["order_exact"] for row in rows),
        "order_exact_pct": pct(sum(row["order_exact"] for row in rows), len(rows)),
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "pairwise_accuracy_pct": pct(pairwise_correct, pairwise_total),
        "visible_ids_exact_pct": pct(sum(row["visible_ids_exact"] for row in rows), len(rows)),
        "permutation_valid_pct": pct(sum(row["order_permutation_valid"] for row in rows), len(rows)),
        "format_valid_pct": pct(sum(row["format_valid"] for row in rows), len(rows)),
        "scenes_5of5": sum(len(group) == 5 and all(row["order_exact"] for row in group) for group in grouped.values()),
        "scenes_1to4of5": sum(any(row["order_exact"] for row in group) and not all(row["order_exact"] for row in group) for group in grouped.values()),
        "scenes_0of5": sum(not any(row["order_exact"] for row in group) for group in grouped.values()),
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
    }


def group_summary(rows: list[dict[str, Any]], keys: Iterable[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = tuple(keys)
    for row in rows:
        values: list[Any] = []
        for key in keys:
            if key in row:
                values.append(row[key])
            else:
                values.append(row["factors"].get(key))
        groups[tuple(values)].append(row)
    output: list[dict[str, Any]] = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        labels = {key: value for key, value in zip(keys, values)}
        output.append({**labels, **summarize(group, " | ".join(f"{k}={v}" for k, v in labels.items()))})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    frozen = read_json(experiment / "config/frozen_scenes.json")
    raw_rows = read_jsonl(experiment / "logs/runs.jsonl")
    if len(raw_rows) != int(config["scheduled_calls"]):
        raise ValueError(f"expected {config['scheduled_calls']} rows, found {len(raw_rows)}")
    if len({row["run_id"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("duplicate run IDs")
    evaluated = [evaluate(row) for row in raw_rows]

    results_dir = experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    split_summary = group_summary(evaluated, ("benchmark_split",))
    cue_summary = group_summary(evaluated, ("benchmark_split", "size_cue"))
    object_count_summary = group_summary(evaluated, ("benchmark_split", "object_count"))
    d1_rows = [row for row in evaluated if row["benchmark_split"] == "N3-D1"]
    d1_occlusion_summary = group_summary(d1_rows, ("occlusion_level",))
    d1_side_summary = group_summary(d1_rows, ("occluder_side",))
    overall = summarize(evaluated, "all_tasks")
    summary = {
        "overall": overall,
        "by_split": split_summary,
        "by_split_and_size_cue": cue_summary,
        "by_split_and_object_count": object_count_summary,
        "d1_by_occlusion_level": d1_occlusion_summary,
        "d1_by_occluder_side": d1_side_summary,
        "error_counts": dict(Counter(row["error_type"] for row in evaluated)),
    }
    write_json(results_dir / "summary.json", summary)
    write_csv(results_dir / "split_summary.csv", split_summary)
    write_csv(results_dir / "cue_summary.csv", cue_summary)
    write_csv(results_dir / "object_count_summary.csv", object_count_summary)
    write_csv(results_dir / "d1_occlusion_summary.csv", d1_occlusion_summary)
    write_csv(results_dir / "d1_side_summary.csv", d1_side_summary)

    scene_rows: list[dict[str, Any]] = []
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        scene_groups[row["scene_id"]].append(row)
    for scene in frozen:
        group = scene_groups[scene["scene_id"]]
        scene_rows.append({
            "scene_id": scene["scene_id"],
            "benchmark_split": scene["benchmark_split"],
            **scene["factors"],
            "expected_order": json.dumps(scene["expected_order"]),
            "successes_out_of_5": sum(row["order_exact"] for row in group),
            "pairwise_accuracy_pct": pct(
                sum(row["pairwise_correct"] for row in group),
                sum(row["pairwise_total"] for row in group),
            ),
        })
    write_csv(results_dir / "scene_summary.csv", scene_rows)

    failures = [row for row in evaluated if not row["order_exact"]]
    failure_rows = [{
        "run_id": row["run_id"],
        "scene_id": row["scene_id"],
        "benchmark_split": row["benchmark_split"],
        "repeat_index": row["repeat_index"],
        "seed": row["seed"],
        "size_cue": row["factors"]["size_cue"],
        "occlusion_level": row["factors"]["occlusion_level"],
        "occluder_side": row["factors"]["occluder_side"],
        "expected_order": json.dumps(row["expected_order"]),
        "predicted_order": json.dumps(row["predicted_order"]),
        "predicted_visible_ids": json.dumps(row["predicted_visible_ids"]),
        "pairwise_correct": row["pairwise_correct"],
        "pairwise_total": row["pairwise_total"],
        "error_type": row["error_type"],
        "parse_error": row["parse_error"],
        "raw_response": row["raw_response"],
        "image_path": row["image_path"],
    } for row in failures]
    write_csv(
        results_dir / "failures.csv",
        failure_rows,
        fields=[
            "run_id", "scene_id", "benchmark_split", "repeat_index", "seed",
            "size_cue", "occlusion_level", "occluder_side", "expected_order",
            "predicted_order", "predicted_visible_ids", "pairwise_correct",
            "pairwise_total", "error_type", "parse_error", "raw_response", "image_path",
        ],
    )

    report_lines = [
        "# N3 D0 left-to-right / D1 front-to-back — Qwen3-VL 30B",
        "",
        f"- Scenes: {len(frozen)} (D0 27, D1 48)",
        f"- Repeats: 5 per scene ({len(evaluated)} calls)",
        "- Output: forced JSON",
        f"- Temperature: {config['generation']['temperature']}",
        f"- Seeds: {config['main_seeds']}",
        "- Batch size: 1",
        "- Candidate ID order: ascending numeric ID, independent of spatial GT",
        "- System prompt: none",
        "",
        "## Primary results",
        "",
        "| Task | Exact | Pairwise | visible_ids exact | Permutation valid | Scene 5/5 | Scene 0/5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in split_summary:
        report_lines.append(
            f"| {item['benchmark_split']} | {item['order_exact_pct']:.3f}% "
            f"({item['order_exact_count']}/{item['calls']}) | {item['pairwise_accuracy_pct']:.3f}% | "
            f"{item['visible_ids_exact_pct']:.3f}% | {item['permutation_valid_pct']:.3f}% | "
            f"{item['scenes_5of5']}/{item['scenes']} | {item['scenes_0of5']}/{item['scenes']} |"
        )
    report_lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "D0 and D1 measure different target relations and therefore their exact accuracies are reported separately. "
        "The combined value is descriptive only and is not treated as a single homogeneous N3 score.",
        "",
        "## Errors",
        "",
        f"- Total failures: {len(failures)}",
        f"- Error counts: `{json.dumps(summary['error_counts'], sort_keys=True)}`",
    ])
    (experiment / "reports/final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    audit = {
        "expected_calls": int(config["scheduled_calls"]),
        "observed_calls": len(evaluated),
        "unique_run_ids": len({row["run_id"] for row in evaluated}),
        "split_call_counts": dict(Counter(row["benchmark_split"] for row in evaluated)),
        "all_finish_reasons_stop": all(row["finish_reason"] == "stop" for row in evaluated),
        "all_context_budgets_valid": all(
            row["context_tokens_if_full_cap"] <= int(row["runtime"]["max_model_len"]) for row in evaluated
        ),
        "all_batch_size_one": all(row["runtime"]["batch_size"] == 1 for row in evaluated),
        "all_images_hash_stable": all(
            row["image_sha256"] == next(scene for scene in frozen if scene["scene_id"] == row["scene_id"])["image_sha256"]
            for row in evaluated
        ),
        "parse_failures": sum(row["parse_error"] is not None for row in evaluated),
    }
    write_json(experiment / "audit/final_audit.json", audit)
    print(json.dumps(summary, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
