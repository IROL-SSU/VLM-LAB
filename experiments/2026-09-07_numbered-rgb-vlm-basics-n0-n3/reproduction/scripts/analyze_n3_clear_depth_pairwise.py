#!/usr/bin/env python3
"""Analyze explicit pairwise N3 outputs and derive maximum-agreement orders."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/n3_clear_depth_pairwise_40scenes_5seeds_20260828"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def expected_pair_map(ids: list[int], expected_order: list[int]) -> dict[tuple[int, int], int]:
    position = {value: index for index, value in enumerate(expected_order)}
    result: dict[tuple[int, int], int] = {}
    for first, second in itertools.combinations(sorted(ids), 2):
        result[(first, second)] = first if position[first] < position[second] else second
    return result


def derive_orders(ids: list[int], predicted: dict[tuple[int, int], int]) -> tuple[list[list[int]], int]:
    scored: list[tuple[int, tuple[int, ...]]] = []
    for order in itertools.permutations(ids):
        position = {value: index for index, value in enumerate(order)}
        agreements = 0
        for (first, second), front in predicted.items():
            predicted_front = first if position[first] < position[second] else second
            agreements += int(predicted_front == front)
        scored.append((agreements, order))
    best_score = max(score for score, _ in scored)
    best_orders = [list(order) for score, order in scored if score == best_score]
    return sorted(best_orders), best_score


def evaluate(row: dict[str, Any]) -> dict[str, Any]:
    expected_ids = [int(value) for value in row["visible_ids"]]
    expected_order = [int(value) for value in row["expected_order"]]
    expected_pairs = expected_pair_map(expected_ids, expected_order)
    parse_error: str | None = None
    parsed: dict[str, Any] = {}
    try:
        value = json.loads(row["raw_response"])
    except json.JSONDecodeError as error:
        parse_error = f"JSON decode error: {error}"
    else:
        if not isinstance(value, dict) or set(value) != {"visible_ids", "pairwise_relations"}:
            parse_error = "top-level keys must be visible_ids and pairwise_relations"
        elif not isinstance(value["visible_ids"], list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value["visible_ids"]
        ):
            parse_error = "visible_ids must be an integer array"
        elif not isinstance(value["pairwise_relations"], list):
            parse_error = "pairwise_relations must be an array"
        else:
            parsed = value

    predicted_visible = parsed.get("visible_ids", [])
    raw_relations = parsed.get("pairwise_relations", [])
    relation_errors: list[str] = []
    predicted_pairs: dict[tuple[int, int], int] = {}
    duplicate_pairs: list[tuple[int, int]] = []
    if parse_error is None:
        for index, relation in enumerate(raw_relations):
            if not isinstance(relation, dict) or set(relation) != {"id_a", "id_b", "front_id"}:
                relation_errors.append(f"relation {index} has invalid keys")
                continue
            a, b, front = relation["id_a"], relation["id_b"], relation["front_id"]
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (a, b, front)):
                relation_errors.append(f"relation {index} contains non-integer values")
                continue
            pair = (a, b)
            if a >= b:
                relation_errors.append(f"relation {index} is not canonical: {pair}")
                continue
            if pair not in expected_pairs:
                relation_errors.append(f"relation {index} uses an invalid pair: {pair}")
                continue
            if front not in pair:
                relation_errors.append(f"relation {index} has front_id outside pair: {front}")
                continue
            if pair in predicted_pairs:
                duplicate_pairs.append(pair)
                continue
            predicted_pairs[pair] = front

    missing_pairs = sorted(set(expected_pairs) - set(predicted_pairs))
    pairwise_complete_valid = (
        parse_error is None
        and not relation_errors
        and not duplicate_pairs
        and not missing_pairs
        and len(raw_relations) == len(expected_pairs)
    )
    pairwise_correct = sum(
        predicted_pairs.get(pair) == expected_front
        for pair, expected_front in expected_pairs.items()
    )
    pairwise_total = len(expected_pairs)
    best_orders: list[list[int]] = []
    best_score = 0
    selected_order: list[int] = []
    if pairwise_complete_valid:
        best_orders, best_score = derive_orders(expected_ids, predicted_pairs)
        selected_order = best_orders[0]
    derived_exact = pairwise_complete_valid and selected_order == expected_order
    gt_among_best = pairwise_complete_valid and expected_order in best_orders
    unique_best = pairwise_complete_valid and len(best_orders) == 1
    visible_exact = parse_error is None and predicted_visible == expected_ids

    if parse_error is not None:
        error_type = "parse_failure"
    elif not visible_exact:
        error_type = "visible_ids_failure"
    elif not pairwise_complete_valid:
        error_type = "pairwise_structure_failure"
    elif pairwise_correct != pairwise_total:
        error_type = "pairwise_relation_failure"
    else:
        error_type = "correct"

    return {
        **row,
        "parsed_response": parsed,
        "predicted_visible_ids": predicted_visible,
        "predicted_pairs": [
            {"id_a": pair[0], "id_b": pair[1], "front_id": front}
            for pair, front in sorted(predicted_pairs.items())
        ],
        "parse_error": parse_error,
        "relation_errors": relation_errors,
        "duplicate_pairs": duplicate_pairs,
        "missing_pairs": missing_pairs,
        "format_valid": parse_error is None,
        "visible_ids_exact": visible_exact,
        "pairwise_complete_valid": pairwise_complete_valid,
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "best_agreement_score": best_score,
        "best_orders": best_orders,
        "selected_order": selected_order,
        "derived_order_exact": derived_exact,
        "gt_among_best_orders": gt_among_best,
        "unique_best_order": unique_best,
        "error_type": error_type,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_groups[row["scene_id"]].append(row)
    return {
        "calls": len(rows),
        "scenes": len(scene_groups),
        "pairwise_correct": sum(row["pairwise_correct"] for row in rows),
        "pairwise_total": sum(row["pairwise_total"] for row in rows),
        "pairwise_accuracy_pct": pct(
            sum(row["pairwise_correct"] for row in rows),
            sum(row["pairwise_total"] for row in rows),
        ),
        "pairwise_complete_valid_pct": pct(sum(row["pairwise_complete_valid"] for row in rows), len(rows)),
        "visible_ids_exact_pct": pct(sum(row["visible_ids_exact"] for row in rows), len(rows)),
        "format_valid_pct": pct(sum(row["format_valid"] for row in rows), len(rows)),
        "derived_order_exact_count": sum(row["derived_order_exact"] for row in rows),
        "derived_order_exact_pct": pct(sum(row["derived_order_exact"] for row in rows), len(rows)),
        "gt_among_best_count": sum(row["gt_among_best_orders"] for row in rows),
        "gt_among_best_pct": pct(sum(row["gt_among_best_orders"] for row in rows), len(rows)),
        "unique_best_pct": pct(sum(row["unique_best_order"] for row in rows), len(rows)),
        "scenes_5of5": sum(all(row["derived_order_exact"] for row in group) for group in scene_groups.values()),
        "scenes_1to4of5": sum(
            any(row["derived_order_exact"] for row in group)
            and not all(row["derived_order_exact"] for row in group)
            for group in scene_groups.values()
        ),
        "scenes_0of5": sum(not any(row["derived_order_exact"] for row in group) for group in scene_groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    frozen = read_json(experiment / "config/frozen_scenes.json")
    rows = read_jsonl(experiment / "logs/runs.jsonl")
    if len(rows) != int(config["scheduled_calls"]):
        raise ValueError(f"expected {config['scheduled_calls']} rows, found {len(rows)}")
    evaluated = [evaluate(row) for row in rows]

    results_dir = experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    overall = summarize(evaluated)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[(row["factors"]["object_count"], row["factors"]["size_level"])].append(row)
    condition_rows = []
    for (object_count, level), group in sorted(groups.items()):
        condition_rows.append({"object_count": object_count, "size_level": level, **summarize(group)})

    summary = {
        "overall": overall,
        "by_condition": condition_rows,
        "error_counts": dict(Counter(row["error_type"] for row in evaluated)),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (results_dir / "condition_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)

    failure_rows = [row for row in evaluated if row["error_type"] != "correct"]
    with (results_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failure_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = [
        "# N3 clear-depth explicit pairwise front-to-back — Qwen3-VL 30B",
        "",
        f"- Scenes: {len(frozen)}; repeats: 5; calls: {len(evaluated)}",
        "- VLM output: every unordered pair only; forced JSON; batch size 1",
        "- Derived order: maximum pairwise agreement over every 3! or 4! permutation",
        "- Deterministic tie-break: lexicographically smallest maximum-agreement order",
        "",
        "## Overall",
        "",
        f"- Direct pairwise accuracy: {overall['pairwise_accuracy_pct']:.3f}% "
        f"({overall['pairwise_correct']}/{overall['pairwise_total']})",
        f"- Pairwise response complete/valid: {overall['pairwise_complete_valid_pct']:.3f}%",
        f"- Derived total-order exact: {overall['derived_order_exact_pct']:.3f}% "
        f"({overall['derived_order_exact_count']}/{overall['calls']})",
        f"- Ground-truth order among maximum-agreement orders: {overall['gt_among_best_pct']:.3f}%",
        f"- Unique maximum-agreement order: {overall['unique_best_pct']:.3f}%",
        f"- visible_ids exact / valid JSON: {overall['visible_ids_exact_pct']:.3f}% / "
        f"{overall['format_valid_pct']:.3f}%",
        f"- Scene 5/5 / 1-4/5 / 0/5: {overall['scenes_5of5']} / "
        f"{overall['scenes_1to4of5']} / {overall['scenes_0of5']}",
        "",
        "## By condition",
        "",
        "| Objects | Level | Pairwise | Derived exact | GT among best | Complete pairs |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in condition_rows:
        report.append(
            f"| {row['object_count']} | {row['size_level']} | {row['pairwise_accuracy_pct']:.3f}% | "
            f"{row['derived_order_exact_pct']:.3f}% | {row['gt_among_best_pct']:.3f}% | "
            f"{row['pairwise_complete_valid_pct']:.3f}% |"
        )
    report.extend(["", f"- Error counts: `{json.dumps(summary['error_counts'], sort_keys=True)}`"])
    reports_dir = experiment / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    audit = {
        "expected_calls": int(config["scheduled_calls"]),
        "observed_calls": len(rows),
        "unique_run_ids": len({row["run_id"] for row in rows}),
        "all_finish_reasons_stop": all(row["finish_reason"] == "stop" for row in rows),
        "all_batch_size_one": all(row["runtime"]["batch_size"] == 1 for row in rows),
        "all_context_budgets_valid": all(
            row["context_tokens_if_full_cap"] <= int(row["runtime"]["max_model_len"])
            for row in rows
        ),
        "parse_failures": sum(row["parse_error"] is not None for row in evaluated),
    }
    audit_dir = experiment / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
