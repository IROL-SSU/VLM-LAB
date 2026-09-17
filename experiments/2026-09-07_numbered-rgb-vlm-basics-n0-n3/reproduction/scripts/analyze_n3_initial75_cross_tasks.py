#!/usr/bin/env python3
"""Analyze the missing crossed tasks for the initial N3 75-scene benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import analyze_n3_d0_ltr_d1_depth_75scenes as base


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/n3_initial75_cross_tasks_75scenes_5seeds_20260906"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = base.read_json(experiment / "config/experiment_config.json")
    frozen = base.read_json(experiment / "config/frozen_scenes.json")
    raw_rows = base.read_jsonl(experiment / "logs/runs.jsonl")
    if len(raw_rows) != int(config["scheduled_calls"]):
        raise ValueError(f"expected {config['scheduled_calls']} rows, found {len(raw_rows)}")
    if len({row["run_id"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("duplicate run IDs")

    source_split_by_scene = {
        scene["scene_id"]: scene["source_benchmark_split"] for scene in frozen
    }
    evaluated: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = base.evaluate(raw)
        row["source_benchmark_split"] = source_split_by_scene[row["scene_id"]]
        evaluated.append(row)

    results_dir = experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_cross_task = base.group_summary(
        evaluated, ("source_benchmark_split", "task_id")
    )
    by_object_count = base.group_summary(
        evaluated, ("source_benchmark_split", "task_id", "object_count")
    )
    by_size_cue = base.group_summary(
        evaluated, ("source_benchmark_split", "task_id", "size_cue")
    )
    d1_lr_rows = [
        row for row in evaluated if row["source_benchmark_split"] == "N3-D1"
    ]
    d1_lr_by_occlusion = base.group_summary(d1_lr_rows, ("occlusion_level",))
    d1_lr_by_side = base.group_summary(d1_lr_rows, ("occluder_side",))
    overall = base.summarize(evaluated, "all_cross_tasks")

    original_summary = base.read_json(
        Path(config["original_experiment"]) / "results/summary.json"
    )
    original_by_split = {
        item["benchmark_split"]: item for item in original_summary["by_split"]
    }
    cross_by_split = {
        item["source_benchmark_split"]: item for item in by_cross_task
    }
    matrix = [
        {
            "source_split": "N3-D0",
            "task": "Left-to-right",
            "status": "original",
            **{key: original_by_split["N3-D0"][key] for key in (
                "calls", "scenes", "order_exact_count", "order_exact_pct",
                "pairwise_accuracy_pct", "scenes_5of5", "scenes_0of5",
            )},
        },
        {
            "source_split": "N3-D0",
            "task": "Front-to-back",
            "status": "cross-task added",
            **{key: cross_by_split["N3-D0"][key] for key in (
                "calls", "scenes", "order_exact_count", "order_exact_pct",
                "pairwise_accuracy_pct", "scenes_5of5", "scenes_0of5",
            )},
        },
        {
            "source_split": "N3-D1",
            "task": "Left-to-right",
            "status": "cross-task added",
            **{key: cross_by_split["N3-D1"][key] for key in (
                "calls", "scenes", "order_exact_count", "order_exact_pct",
                "pairwise_accuracy_pct", "scenes_5of5", "scenes_0of5",
            )},
        },
        {
            "source_split": "N3-D1",
            "task": "Front-to-back",
            "status": "original",
            **{key: original_by_split["N3-D1"][key] for key in (
                "calls", "scenes", "order_exact_count", "order_exact_pct",
                "pairwise_accuracy_pct", "scenes_5of5", "scenes_0of5",
            )},
        },
    ]

    summary = {
        "overall": overall,
        "by_cross_task": by_cross_task,
        "by_object_count": by_object_count,
        "by_size_cue": by_size_cue,
        "d1_left_to_right_by_occlusion_level": d1_lr_by_occlusion,
        "d1_left_to_right_by_occluder_side": d1_lr_by_side,
        "complete_2x2_matrix": matrix,
        "error_counts": dict(Counter(row["error_type"] for row in evaluated)),
    }
    base.write_json(results_dir / "summary.json", summary)
    base.write_csv(results_dir / "cross_task_summary.csv", by_cross_task)
    base.write_csv(results_dir / "object_count_summary.csv", by_object_count)
    base.write_csv(results_dir / "size_cue_summary.csv", by_size_cue)
    base.write_csv(results_dir / "d1_lr_occlusion_summary.csv", d1_lr_by_occlusion)
    base.write_csv(results_dir / "d1_lr_side_summary.csv", d1_lr_by_side)
    base.write_csv(results_dir / "complete_2x2_matrix.csv", matrix)

    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        scene_groups[row["scene_id"]].append(row)
    scene_rows: list[dict[str, Any]] = []
    for scene in frozen:
        group = scene_groups[scene["scene_id"]]
        scene_rows.append({
            "scene_id": scene["scene_id"],
            "source_benchmark_split": scene["source_benchmark_split"],
            "task_id": scene["task_id"],
            **scene["factors"],
            "expected_order": json.dumps(scene["expected_order"]),
            "successes_out_of_5": sum(row["order_exact"] for row in group),
            "pairwise_accuracy_pct": base.pct(
                sum(row["pairwise_correct"] for row in group),
                sum(row["pairwise_total"] for row in group),
            ),
        })
    base.write_csv(results_dir / "scene_summary.csv", scene_rows)

    failures = [row for row in evaluated if not row["order_exact"]]
    failure_rows = [{
        "run_id": row["run_id"],
        "scene_id": row["scene_id"],
        "source_benchmark_split": row["source_benchmark_split"],
        "task_id": row["task_id"],
        "repeat_index": row["repeat_index"],
        "seed": row["seed"],
        "object_count": row["factors"]["object_count"],
        "size_cue": row["factors"]["size_cue"],
        "occlusion_level": row["factors"]["occlusion_level"],
        "occluder_side": row["factors"]["occluder_side"],
        "expected_order": json.dumps(row["expected_order"]),
        "predicted_order": json.dumps(row["predicted_order"]),
        "pairwise_correct": row["pairwise_correct"],
        "pairwise_total": row["pairwise_total"],
        "error_type": row["error_type"],
    } for row in failures]
    base.write_csv(results_dir / "failures.csv", failure_rows)

    report_lines = [
        "# Initial N3 75 scenes — missing cross-task completion",
        "",
        "- Same 75 numbered RGB images as the original benchmark",
        "- Added tasks: D0 27 scenes front-to-back; D1 48 scenes left-to-right",
        f"- Repeats: 5 per scene ({len(evaluated)} new calls)",
        "- Prompts, model, seeds, generation settings, and forced-JSON schemas match the original experiment",
        "",
        "## Complete 2×2 task/dataset matrix",
        "",
        "| Source scenes | Task | Status | Exact | All-pairs | Scene 5/5 | Scene 0/5 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in matrix:
        report_lines.append(
            f"| {row['source_split']} ({row['scenes']}) | {row['task']} | {row['status']} | "
            f"{row['order_exact_pct']:.3f}% ({row['order_exact_count']}/{row['calls']}) | "
            f"{row['pairwise_accuracy_pct']:.3f}% | {row['scenes_5of5']}/{row['scenes']} | "
            f"{row['scenes_0of5']}/{row['scenes']} |"
        )

    report_lines.extend(["", "## Added-task breakdown by object count", ""])
    report_lines.extend([
        "| Source scenes | Task | Objects | Exact | All-pairs |",
        "|---|---|---:|---:|---:|",
    ])
    for row in by_object_count:
        report_lines.append(
            f"| {row['source_benchmark_split']} | {row['task_id']} | {row['object_count']} | "
            f"{row['order_exact_pct']:.3f}% | {row['pairwise_accuracy_pct']:.3f}% |"
        )

    report_lines.extend(["", "## Added-task breakdown by size cue", ""])
    report_lines.extend([
        "| Source scenes | Task | Size cue | Exact | All-pairs |",
        "|---|---|---|---:|---:|",
    ])
    for row in by_size_cue:
        report_lines.append(
            f"| {row['source_benchmark_split']} | {row['task_id']} | {row['size_cue']} | "
            f"{row['order_exact_pct']:.3f}% | {row['pairwise_accuracy_pct']:.3f}% |"
        )

    report_lines.extend([
        "",
        "## Audit",
        "",
        f"- New calls: {len(evaluated)}/{config['scheduled_calls']}",
        f"- Format failures: {sum(row['parse_error'] is not None for row in evaluated)}",
        f"- Non-stop finishes: {sum(row['finish_reason'] != 'stop' for row in evaluated)}",
        f"- Duplicate run IDs: {len(evaluated) - len({row['run_id'] for row in evaluated})}",
        f"- Total failures: {len(failures)}",
    ])
    (experiment / "reports/final_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    audit = {
        "expected_calls": int(config["scheduled_calls"]),
        "observed_calls": len(evaluated),
        "unique_run_ids": len({row["run_id"] for row in evaluated}),
        "task_call_counts": dict(Counter(row["task_id"] for row in evaluated)),
        "source_split_call_counts": dict(
            Counter(row["source_benchmark_split"] for row in evaluated)
        ),
        "all_finish_reasons_stop": all(row["finish_reason"] == "stop" for row in evaluated),
        "all_context_budgets_valid": all(
            row["context_tokens_if_full_cap"] <= int(row["runtime"]["max_model_len"])
            for row in evaluated
        ),
        "all_batch_size_one": all(row["runtime"]["batch_size"] == 1 for row in evaluated),
        "all_images_hash_stable": all(
            row["image_sha256"]
            == next(scene for scene in frozen if scene["scene_id"] == row["scene_id"])["image_sha256"]
            for row in evaluated
        ),
        "parse_failures": sum(row["parse_error"] is not None for row in evaluated),
    }
    base.write_json(experiment / "audit/final_audit.json", audit)
    print(json.dumps(summary, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
