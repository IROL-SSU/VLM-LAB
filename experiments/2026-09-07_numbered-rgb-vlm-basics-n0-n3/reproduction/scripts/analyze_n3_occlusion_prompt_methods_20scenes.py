#!/usr/bin/env python3
"""Analyze five prompt methods on the fixed 20-scene FB1 benchmark."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import analyze_n3_clear_depth_anchor_target as anchor_eval
import analyze_n3_clear_depth_pairwise as pair_eval
import analyze_n3_d0_ltr_d1_depth_75scenes as direct_eval


ROOT = Path("/home/ssu/ShelfScene")
EXPERIMENTS = ROOT / "experiments"
OUTPUT = EXPERIMENTS / "n3_occlusion_prompt_methods_comparison_20260831"
PATHS = {
    "Minimal direct": EXPERIMENTS / "n3_occlusion_minimal_direct_20scenes_5seeds_20260831",
    "Detailed direct": EXPERIMENTS / "n3_occlusion_detailed_direct_20scenes_5seeds_20260831",
    "Contact point": EXPERIMENTS / "n3_occlusion_contact_point_20scenes_5seeds_20260831",
    "Explicit pairs": EXPERIMENTS / "n3_occlusion_explicit_pairs_20scenes_5seeds_20260831",
    "Single target": EXPERIMENTS / "n3_occlusion_single_target_20scenes_5seeds_20260831",
}
DIRECT_METHODS = {"Minimal direct", "Detailed direct", "Contact point"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pct(a: int | float, b: int | float) -> float:
    return round(100.0 * a / b, 3) if b else 0.0


def side_class(row: dict) -> str:
    sides = row["factors"]["edge_sides"]
    if all(side == "left" for side in sides):
        return "all_left"
    if all(side == "right" for side in sides):
        return "all_right"
    return "mixed"


def target_from_order(row: dict) -> tuple[int, int, bool]:
    expected_order = [int(value) for value in row["expected_order"]]
    predicted_order = [int(value) for value in row.get("predicted_order", [])]
    target = int(row["target_id"])
    expected_position = {value: index for index, value in enumerate(expected_order)}
    predicted_position = {value: index for index, value in enumerate(predicted_order)}
    correct = 0
    total = len(expected_order) - 1
    for value in expected_order:
        if value == target or target not in predicted_position or value not in predicted_position:
            continue
        expected_front = expected_position[value] < expected_position[target]
        predicted_front = predicted_position[value] < predicted_position[target]
        correct += int(expected_front == predicted_front)
    return correct, total, correct == total


def target_from_pairs(row: dict) -> tuple[int, int, bool]:
    expected_order = [int(value) for value in row["expected_order"]]
    target = int(row["target_id"])
    position = {value: index for index, value in enumerate(expected_order)}
    pair_map = {
        (item["id_a"], item["id_b"]): item["front_id"]
        for item in row.get("predicted_pairs", [])
    }
    correct = 0
    total = len(expected_order) - 1
    for value in expected_order:
        if value == target:
            continue
        pair = tuple(sorted((target, value)))
        expected_front_id = value if position[value] < position[target] else target
        correct += int(pair_map.get(pair) == expected_front_id)
    return correct, total, correct == total


def target_summary(
    rows: list[dict], evaluator: Callable[[dict], tuple[int, int, bool]] | None
) -> dict:
    if evaluator is None:
        correct = sum(row["relation_correct"] for row in rows)
        total = sum(row["relation_total"] for row in rows)
        exact = sum(row["target_partition_exact"] for row in rows)
    else:
        evaluated = [evaluator(row) for row in rows]
        correct = sum(value[0] for value in evaluated)
        total = sum(value[1] for value in evaluated)
        exact = sum(value[2] for value in evaluated)
    return {
        "calls": len(rows),
        "relation_correct": correct,
        "relation_total": total,
        "target_relation_accuracy_pct": pct(correct, total),
        "target_partition_exact_count": exact,
        "target_partition_exact_pct": pct(exact, len(rows)),
    }


def grouped(
    rows: list[dict], key_fn: Callable[[dict], str], summarize_fn: Callable[[list[dict]], dict]
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return [
        {"group": key, **summarize_fn(group)}
        for key, group in sorted(groups.items())
    ]


def direct_summary(rows: list[dict]) -> dict:
    return direct_eval.summarize(rows, "occlusion")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluated_by_method: dict[str, list[dict]] = {}
    frozen_by_method: dict[str, list[dict]] = {}
    configs = {}
    for method, path in PATHS.items():
        config = read_json(path / "config/experiment_config.json")
        frozen = read_json(path / "config/frozen_scenes.json")
        raw = read_jsonl(path / "logs/runs.jsonl")
        if len(raw) != 100 or len(raw) != int(config["scheduled_calls"]):
            raise ValueError(f"{method}: expected 100 rows, found {len(raw)}")
        if method in DIRECT_METHODS:
            evaluated = [direct_eval.evaluate(row) for row in raw]
        elif method == "Explicit pairs":
            evaluated = [pair_eval.evaluate(row) for row in raw]
        else:
            evaluated = [anchor_eval.evaluate(row) for row in raw]
        evaluated_by_method[method] = evaluated
        frozen_by_method[method] = frozen
        configs[method] = config
        destination = path / "results/evaluated_records.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in evaluated:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    reference = frozen_by_method["Minimal direct"]
    reference_signature = [
        (row["scene_id"], row["image_sha256"], row["visible_ids"], row["expected_order"], row["target_id"])
        for row in reference
    ]
    all_frozen_inputs_identical = all(
        [
            (row["scene_id"], row["image_sha256"], row["visible_ids"], row["expected_order"], row["target_id"])
            for row in frozen_by_method[method]
        ]
        == reference_signature
        for method in PATHS
    )

    full_order = {}
    by_structure_full_order = {}
    by_side_full_order = {}
    for method in DIRECT_METHODS:
        rows = evaluated_by_method[method]
        full_order[method] = direct_summary(rows)
        by_structure_full_order[method] = grouped(
            rows,
            lambda row: row["factors"]["occlusion_structure"],
            direct_summary,
        )
        by_side_full_order[method] = grouped(rows, side_class, direct_summary)
    pair_rows = evaluated_by_method["Explicit pairs"]
    pair_overall = pair_eval.summarize(pair_rows)
    full_order["Explicit pairs"] = pair_overall
    by_structure_full_order["Explicit pairs"] = grouped(
        pair_rows,
        lambda row: row["factors"]["occlusion_structure"],
        pair_eval.summarize,
    )
    by_side_full_order["Explicit pairs"] = grouped(
        pair_rows, side_class, pair_eval.summarize
    )

    target_evaluators = {
        "Minimal direct": target_from_order,
        "Detailed direct": target_from_order,
        "Contact point": target_from_order,
        "Explicit pairs": target_from_pairs,
        "Single target": None,
    }
    target_relative = {
        method: target_summary(evaluated_by_method[method], evaluator)
        for method, evaluator in target_evaluators.items()
    }
    by_structure_target = {
        method: grouped(
            evaluated_by_method[method],
            lambda row: row["factors"]["occlusion_structure"],
            lambda group, evaluator=evaluator: target_summary(group, evaluator),
        )
        for method, evaluator in target_evaluators.items()
    }

    # Scene-level rows make the failures and seed stability directly auditable.
    by_run = {
        method: {row["run_id"]: row for row in rows}
        for method, rows in evaluated_by_method.items()
    }
    scene_rows = []
    for scene in reference:
        scene_id = scene["scene_id"]
        row = {
            "scene_id": scene_id,
            "object_count": scene["factors"]["object_count"],
            "structure": scene["factors"]["occlusion_structure"],
            "edge_sides": "/".join(scene["factors"]["edge_sides"]),
            "expected_order": json.dumps(scene["expected_order"]),
            "target_id": scene["target_id"],
            "image": scene["image"],
        }
        run_ids = [
            f"D1_FRONT_TO_BACK__{scene_id}__r{repeat:02d}_s{seed}"
            for repeat, seed in enumerate(configs["Minimal direct"]["main_seeds"], 1)
        ]
        for method in DIRECT_METHODS:
            row[f"{method}_order_successes"] = sum(
                by_run[method][run_id]["order_exact"] for run_id in run_ids
            )
        row["Explicit pairs_order_successes"] = sum(
            by_run["Explicit pairs"][run_id]["derived_order_exact"]
            for run_id in run_ids
        )
        for method, evaluator in target_evaluators.items():
            if evaluator is None:
                row[f"{method}_target_successes"] = sum(
                    by_run[method][run_id]["target_partition_exact"]
                    for run_id in run_ids
                )
            else:
                row[f"{method}_target_successes"] = sum(
                    evaluator(by_run[method][run_id])[2] for run_id in run_ids
                )
        scene_rows.append(row)

    results = {
        "design": {
            "scenes": 20,
            "seeds_per_scene": 5,
            "methods": 5,
            "total_calls": 500,
            "model": configs["Minimal direct"]["model_id"],
            "temperature": 0.3,
            "batch_size": 1,
            "forced_json": True,
            "all_frozen_inputs_identical": all_frozen_inputs_identical,
        },
        "full_order": full_order,
        "target_relative": target_relative,
        "by_structure_full_order": by_structure_full_order,
        "by_side_full_order": by_side_full_order,
        "by_structure_target": by_structure_target,
    }
    write_json(OUTPUT / "results/summary.json", results)

    fields = list(scene_rows[0])
    csv_path = OUTPUT / "results/scene_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scene_rows)

    direct_order = ["Minimal direct", "Detailed direct", "Contact point"]
    lines = [
        "# N3 FB1 structured occlusion — prompt-method comparison",
        "",
        "- Fixed data: the same 20 numbered RGB scenes × the same 5 seeds",
        "- Model: Qwen3-VL-30B-A3B-Instruct",
        "- Generation: temperature 0.3, batch 1, forced JSON, max_tokens 1024",
        "- Total calls: 500",
        "",
        "## Complete front-to-back order",
        "",
        "| Method | Full-order exact | Pairwise accuracy | JSON valid |",
        "|---|---:|---:|---:|",
    ]
    for method in direct_order:
        row = full_order[method]
        lines.append(
            f"| {method} | {row['order_exact_pct']:.3f}% | "
            f"{row['pairwise_accuracy_pct']:.3f}% | {row['format_valid_pct']:.3f}% |"
        )
    lines.append(
        f"| Explicit pairs | {pair_overall['derived_order_exact_pct']:.3f}% derived | "
        f"{pair_overall['pairwise_accuracy_pct']:.3f}% direct | "
        f"{pair_overall['format_valid_pct']:.3f}% |"
    )
    lines.extend([
        "",
        "Single target is excluded above because it does not output a complete order.",
        "",
        "## Same fixed-target relations",
        "",
        "| Method | Relation accuracy | All target relations exact |",
        "|---|---:|---:|",
    ])
    for method in PATHS:
        row = target_relative[method]
        lines.append(
            f"| {method} | {row['target_relation_accuracy_pct']:.3f}% | "
            f"{row['target_partition_exact_pct']:.3f}% |"
        )
    lines.extend([
        "",
        "## Full-order exact by structure",
        "",
        "| Structure | Minimal | Detailed | Contact | Explicit pairs derived |",
        "|---|---:|---:|---:|---:|",
    ])
    structure_maps = {
        method: {row["group"]: row for row in values}
        for method, values in by_structure_full_order.items()
    }
    for structure in sorted(structure_maps["Minimal direct"]):
        values = []
        for method in direct_order:
            values.append(structure_maps[method][structure]["order_exact_pct"])
        values.append(
            structure_maps["Explicit pairs"][structure]["derived_order_exact_pct"]
        )
        lines.append(
            f"| {structure} | " + " | ".join(f"{value:.3f}%" for value in values) + " |"
        )
    (OUTPUT / "reports/final_report.md").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "reports/final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    all_rows = [row for rows in evaluated_by_method.values() for row in rows]
    audit = {
        "expected_calls": 500,
        "observed_calls": len(all_rows),
        "unique_method_run_pairs": sum(len(rows) for rows in evaluated_by_method.values()),
        "all_frozen_inputs_identical": all_frozen_inputs_identical,
        "all_batch_size_one": all(row["runtime"]["batch_size"] == 1 for row in all_rows),
        "all_finish_reasons_stop": all(row["finish_reason"] == "stop" for row in all_rows),
        "all_context_budgets_valid": all(
            row["context_tokens_if_full_cap"] <= row["runtime"]["max_model_len"]
            for row in all_rows
        ),
        "maximum_output_tokens_observed": max(row["output_tokens"] for row in all_rows),
        "format_failures_by_method": {
            method: sum(not row["format_valid"] for row in rows)
            for method, rows in evaluated_by_method.items()
        },
    }
    write_json(OUTPUT / "audit/final_audit.json", audit)
    print(json.dumps(results, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
