#!/usr/bin/env python3
"""Analyze target-anchored N3 front/behind classifications."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/n3_clear_depth_anchor_target_40scenes_5seeds_20260828"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(a: int | float, b: int | float) -> float:
    return round(100 * a / b, 3) if b else 0.0


def expected_relations(row: dict[str, Any]) -> dict[int, str]:
    order = [int(value) for value in row["expected_order"]]
    target = int(row["target_id"])
    target_position = order.index(target)
    return {
        value: "FRONT_OF_TARGET" if index < target_position else "BEHIND_TARGET"
        for index, value in enumerate(order)
        if value != target
    }


def evaluate(row: dict[str, Any]) -> dict[str, Any]:
    ids = [int(value) for value in row["visible_ids"]]
    target = int(row["target_id"])
    expected = expected_relations(row)
    parse_error: str | None = None
    parsed: dict[str, Any] = {}
    try:
        value = json.loads(row["raw_response"])
    except json.JSONDecodeError as error:
        parse_error = f"JSON decode error: {error}"
    else:
        keys = {"visible_ids", "target_id", "relations_to_target"}
        if not isinstance(value, dict) or set(value) != keys:
            parse_error = f"top-level keys must be {sorted(keys)}"
        elif not isinstance(value["visible_ids"], list):
            parse_error = "visible_ids must be an array"
        elif not isinstance(value["target_id"], int) or isinstance(value["target_id"], bool):
            parse_error = "target_id must be an integer"
        elif not isinstance(value["relations_to_target"], list):
            parse_error = "relations_to_target must be an array"
        else:
            parsed = value

    predicted: dict[int, str] = {}
    structure_errors: list[str] = []
    duplicates: list[int] = []
    relations = parsed.get("relations_to_target", [])
    if parse_error is None:
        for index, item in enumerate(relations):
            if not isinstance(item, dict) or set(item) != {"id", "relation"}:
                structure_errors.append(f"relation {index} has invalid keys")
                continue
            instance_id, relation = item["id"], item["relation"]
            if instance_id not in expected:
                structure_errors.append(f"relation {index} has invalid non-target ID {instance_id}")
                continue
            if relation not in {"FRONT_OF_TARGET", "BEHIND_TARGET"}:
                structure_errors.append(f"relation {index} has invalid label {relation!r}")
                continue
            if instance_id in predicted:
                duplicates.append(instance_id)
                continue
            predicted[instance_id] = relation
    missing = sorted(set(expected) - set(predicted))
    complete = (
        parse_error is None
        and not structure_errors
        and not duplicates
        and not missing
        and len(relations) == len(expected)
    )
    visible_exact = parse_error is None and parsed.get("visible_ids") == ids
    target_exact = parse_error is None and parsed.get("target_id") == target
    correct = sum(predicted.get(instance_id) == relation for instance_id, relation in expected.items())
    total = len(expected)
    partition_exact = complete and visible_exact and target_exact and correct == total
    if parse_error:
        error_type = "parse_failure"
    elif not visible_exact:
        error_type = "visible_ids_failure"
    elif not target_exact:
        error_type = "target_id_failure"
    elif not complete:
        error_type = "relation_structure_failure"
    elif not partition_exact:
        error_type = "target_relation_failure"
    else:
        error_type = "correct"
    return {
        **row,
        "parsed_response": parsed,
        "expected_relations": expected,
        "predicted_relations": predicted,
        "parse_error": parse_error,
        "structure_errors": structure_errors,
        "duplicate_ids": duplicates,
        "missing_ids": missing,
        "format_valid": parse_error is None,
        "visible_ids_exact": visible_exact,
        "target_id_exact": target_exact,
        "relations_complete_valid": complete,
        "relation_correct": correct,
        "relation_total": total,
        "target_partition_exact": partition_exact,
        "error_type": error_type,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_groups[row["scene_id"]].append(row)
    correct = sum(row["relation_correct"] for row in rows)
    total = sum(row["relation_total"] for row in rows)
    exact = sum(row["target_partition_exact"] for row in rows)
    return {
        "calls": len(rows),
        "scenes": len(scene_groups),
        "relation_correct": correct,
        "relation_total": total,
        "target_relation_accuracy_pct": pct(correct, total),
        "target_partition_exact_count": exact,
        "target_partition_exact_pct": pct(exact, len(rows)),
        "relations_complete_valid_pct": pct(sum(row["relations_complete_valid"] for row in rows), len(rows)),
        "visible_ids_exact_pct": pct(sum(row["visible_ids_exact"] for row in rows), len(rows)),
        "target_id_exact_pct": pct(sum(row["target_id_exact"] for row in rows), len(rows)),
        "format_valid_pct": pct(sum(row["format_valid"] for row in rows), len(rows)),
        "scenes_5of5": sum(all(row["target_partition_exact"] for row in group) for group in scene_groups.values()),
        "scenes_1to4of5": sum(
            any(row["target_partition_exact"] for row in group)
            and not all(row["target_partition_exact"] for row in group)
            for group in scene_groups.values()
        ),
        "scenes_0of5": sum(not any(row["target_partition_exact"] for row in group) for group in scene_groups.values()),
    }


def relation_correct_from_order(row: dict[str, Any], source: dict[str, Any]) -> tuple[int, int, bool]:
    expected = expected_relations(row)
    order = source.get("predicted_order", [])
    position = {value: index for index, value in enumerate(order)}
    target = row["target_id"]
    correct = 0
    for instance_id, label in expected.items():
        if target not in position or instance_id not in position:
            continue
        predicted = "FRONT_OF_TARGET" if position[instance_id] < position[target] else "BEHIND_TARGET"
        correct += int(predicted == label)
    return correct, len(expected), correct == len(expected)


def relation_correct_from_pairs(row: dict[str, Any], source: dict[str, Any]) -> tuple[int, int, bool]:
    expected = expected_relations(row)
    pair_map = {
        (item["id_a"], item["id_b"]): item["front_id"]
        for item in source.get("predicted_pairs", [])
    }
    target = row["target_id"]
    correct = 0
    for instance_id, label in expected.items():
        pair = tuple(sorted((target, instance_id)))
        front = pair_map.get(pair)
        predicted = "FRONT_OF_TARGET" if front == instance_id else "BEHIND_TARGET" if front == target else None
        correct += int(predicted == label)
    return correct, len(expected), correct == len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    frozen = read_json(experiment / "config/frozen_scenes.json")
    frozen_by_scene = {scene["scene_id"]: scene for scene in frozen}
    rows = read_jsonl(experiment / "logs/runs.jsonl")
    if len(rows) != int(config["scheduled_calls"]):
        raise ValueError(f"expected {config['scheduled_calls']} rows, found {len(rows)}")
    enriched_rows = []
    for row in rows:
        scene = frozen_by_scene[row["scene_id"]]
        enriched_rows.append({
            **row,
            "target_id": int(scene["target_id"]),
            "target_rank_front_zero_based": int(scene["target_rank_front_zero_based"]),
        })
    evaluated = [evaluate(row) for row in enriched_rows]
    overall = summarize(evaluated)

    condition_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    rank_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        condition_groups[(row["factors"]["object_count"], row["factors"]["size_level"])].append(row)
        rank_groups[(row["factors"]["object_count"], row["target_rank_front_zero_based"])].append(row)
    by_condition = [
        {"object_count": key[0], "size_level": key[1], **summarize(group)}
        for key, group in sorted(condition_groups.items())
    ]
    by_target_rank = [
        {"object_count": key[0], "target_rank_front_zero_based": key[1], **summarize(group)}
        for key, group in sorted(rank_groups.items())
    ]
    label_counts: dict[str, dict[str, int | float]] = {}
    for label in ("FRONT_OF_TARGET", "BEHIND_TARGET"):
        label_total = 0
        label_correct = 0
        for row in evaluated:
            for instance_id, expected_label in row["expected_relations"].items():
                if expected_label != label:
                    continue
                label_total += 1
                label_correct += int(row["predicted_relations"].get(instance_id) == label)
        label_counts[label] = {
            "correct": label_correct,
            "total": label_total,
            "accuracy_pct": pct(label_correct, label_total),
        }

    if experiment.name.startswith("n3_packed_anchor_target_"):
        comparison_sources = {
            "minimal_total_order": (
                ROOT / "experiments/n3_packed_minimal_direct_40scenes_5seeds_20260829/results/evaluated_records.jsonl",
                relation_correct_from_order,
            ),
            "detailed_total_order": (
                ROOT / "experiments/n3_packed_detailed_direct_40scenes_5seeds_20260829/results/evaluated_records.jsonl",
                relation_correct_from_order,
            ),
            "contact_total_order": (
                ROOT / "experiments/n3_packed_contact_prompt_40scenes_5seeds_20260829/results/evaluated_records.jsonl",
                relation_correct_from_order,
            ),
            "explicit_all_pairs": (
                ROOT / "experiments/n3_packed_pairwise_40scenes_5seeds_20260829/results/evaluated_records.jsonl",
                relation_correct_from_pairs,
            ),
        }
    else:
        comparison_sources = {
            "minimal_total_order": (
                ROOT / "experiments/n3_clear_depth_minimal_direct_40scenes_5seeds_20260831/results/evaluated_records.jsonl",
                relation_correct_from_order,
            ),
            "baseline_total_order": (
                ROOT / "experiments/n3_clear_depth_40scenes_5seeds_20260828/results/evaluated_records.jsonl",
                relation_correct_from_order,
            ),
            "contact_total_order": (
                ROOT / "experiments/n3_clear_depth_contact_prompt_40scenes_5seeds_20260828/results/evaluated_records.jsonl",
                relation_correct_from_order,
            ),
            "explicit_all_pairs": (
                ROOT / "experiments/n3_clear_depth_pairwise_40scenes_5seeds_20260828/results/evaluated_records.jsonl",
                relation_correct_from_pairs,
            ),
        }
    comparison = [{"method": "single_target_anchor", **overall}]
    for method, (path, evaluator) in comparison_sources.items():
        source_by_run = {source["run_id"]: source for source in read_jsonl(path)}
        correct = total = exact = 0
        for row in evaluated:
            c, t, e = evaluator(row, source_by_run[row["run_id"]])
            correct += c
            total += t
            exact += int(e)
        comparison.append({
            "method": method,
            "relation_correct": correct,
            "relation_total": total,
            "target_relation_accuracy_pct": pct(correct, total),
            "target_partition_exact_count": exact,
            "target_partition_exact_pct": pct(exact, len(evaluated)),
        })

    summary = {
        "overall": overall,
        "by_condition": by_condition,
        "by_target_rank": by_target_rank,
        "by_relation_label": label_counts,
        "same_anchor_pair_comparison": comparison,
        "error_counts": dict(Counter(row["error_type"] for row in evaluated)),
    }
    results = experiment / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (results / "evaluated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    failures = [row for row in evaluated if not row["target_partition_exact"]]
    with (results / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for filename, data in (("condition_summary.csv", by_condition), ("target_rank_summary.csv", by_target_rank)):
        with (results / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    report = [
        "# N3 clear-depth single-target anchor relations — Qwen3-VL 30B",
        "",
        f"- Scenes: 40; repeats: 5; calls: {len(evaluated)}",
        "- One fixed target ID per scene; every other object classified as FRONT_OF_TARGET or BEHIND_TARGET",
        "- Forced JSON; batch size 1; temperature 0.3",
        "",
        "## Overall",
        "",
        f"- Target-relative relation accuracy: {overall['target_relation_accuracy_pct']:.3f}% "
        f"({overall['relation_correct']}/{overall['relation_total']})",
        f"- Target partition exact: {overall['target_partition_exact_pct']:.3f}% "
        f"({overall['target_partition_exact_count']}/{overall['calls']})",
        f"- Complete relations / visible IDs / target ID / JSON: "
        f"{overall['relations_complete_valid_pct']:.3f}% / {overall['visible_ids_exact_pct']:.3f}% / "
        f"{overall['target_id_exact_pct']:.3f}% / {overall['format_valid_pct']:.3f}%",
        f"- FRONT_OF_TARGET: {label_counts['FRONT_OF_TARGET']['accuracy_pct']:.3f}% "
        f"({label_counts['FRONT_OF_TARGET']['correct']}/{label_counts['FRONT_OF_TARGET']['total']})",
        f"- BEHIND_TARGET: {label_counts['BEHIND_TARGET']['accuracy_pct']:.3f}% "
        f"({label_counts['BEHIND_TARGET']['correct']}/{label_counts['BEHIND_TARGET']['total']})",
        "",
        "## Same anchor-pair comparison",
        "",
        "| Method | Relation accuracy | All target relations exact |",
        "|---|---:|---:|",
    ]
    for row in comparison:
        report.append(
            f"| {row['method']} | {row['target_relation_accuracy_pct']:.3f}% "
            f"({row['relation_correct']}/{row['relation_total']}) | "
            f"{row['target_partition_exact_pct']:.3f}% "
            f"({row['target_partition_exact_count']}/{len(evaluated)}) |"
        )
    report.extend([
        "",
        "## By condition",
        "",
        "| Objects | Level | Relation accuracy | Partition exact |",
        "|---:|---|---:|---:|",
    ])
    for row in by_condition:
        report.append(
            f"| {row['object_count']} | {row['size_level']} | "
            f"{row['target_relation_accuracy_pct']:.3f}% | {row['target_partition_exact_pct']:.3f}% |"
        )
    reports = experiment / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

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
    (audit_dir / "final_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
