#!/usr/bin/env python3
"""Analyze the 3-scene-type N3 RGB-D prompt-method experiment.

The report keeps two questions separate:

1. Can the model recover the complete front-to-back order?
2. For one fixed target, can it classify every other object as front/behind?

It also performs paired RGB-only versus RGB-D comparisons on the same scene/seed
calls.  The structured-occlusion RGB baseline contains the original 20 scenes;
the remaining RGB-D summaries use all 40 newly frozen scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import analyze_n3_clear_depth_anchor_target as anchor_eval
import analyze_n3_clear_depth_pairwise as pair_eval
import analyze_n3_d0_ltr_d1_depth_75scenes as order_eval


ROOT = Path("/home/ssu/ShelfScene")
DEFAULT_EXPERIMENT = ROOT / "experiments/n3_rgbd_3scene_40each_5methods_5seeds_20260831"

ORDER_METHODS = ("minimal_direct", "detailed_direct", "contact_point", "explicit_pairs")
METHOD_LABELS = {
    "minimal_direct": "Minimal direct",
    "detailed_direct": "Detailed direct",
    "contact_point": "Contact point",
    "explicit_pairs": "Explicit all pairs",
    "single_target": "Single target",
}
SCENE_LABELS = {
    "separated_nonoverlap": "Separated non-occlusion",
    "packed_nonoverlap": "Packed non-occlusion",
    "structured_occlusion": "Structured occlusion",
}

RGB_BASELINES = {
    "separated_nonoverlap": {
        "minimal_direct": "n3_clear_depth_minimal_direct_40scenes_5seeds_20260831",
        "detailed_direct": "n3_clear_depth_40scenes_5seeds_20260828",
        "contact_point": "n3_clear_depth_contact_prompt_40scenes_5seeds_20260828",
        "explicit_pairs": "n3_clear_depth_pairwise_40scenes_5seeds_20260828",
        "single_target": "n3_clear_depth_anchor_target_40scenes_5seeds_20260828",
    },
    "packed_nonoverlap": {
        "minimal_direct": "n3_packed_minimal_direct_40scenes_5seeds_20260829",
        "detailed_direct": "n3_packed_detailed_direct_40scenes_5seeds_20260829",
        "contact_point": "n3_packed_contact_prompt_40scenes_5seeds_20260829",
        "explicit_pairs": "n3_packed_pairwise_40scenes_5seeds_20260829",
        "single_target": "n3_packed_anchor_target_40scenes_5seeds_20260829",
    },
    "structured_occlusion": {
        "minimal_direct": "n3_occlusion_minimal_direct_20scenes_5seeds_20260831",
        "detailed_direct": "n3_occlusion_detailed_direct_20scenes_5seeds_20260831",
        "contact_point": "n3_occlusion_contact_point_20scenes_5seeds_20260831",
        "explicit_pairs": "n3_occlusion_explicit_pairs_20scenes_5seeds_20260831",
        "single_target": "n3_occlusion_single_target_20scenes_5seeds_20260831",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(a: int | float, b: int | float) -> float:
    return round(100.0 * a / b, 3) if b else 0.0


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def evaluate_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated = []
    for row in raw_rows:
        method = row["method"]
        # Existing evaluators group on scene_id.  Use scene_uid here because the
        # separated and packed datasets intentionally reuse scene IDs.
        working = {**row, "source_scene_id": row["scene_id"], "scene_id": row["scene_uid"]}
        if method in {"minimal_direct", "detailed_direct", "contact_point"}:
            working["output_key"] = "front_to_back"
            result = order_eval.evaluate(working)
        elif method == "explicit_pairs":
            result = pair_eval.evaluate(working)
        elif method == "single_target":
            result = anchor_eval.evaluate(working)
        else:
            raise ValueError(f"unknown method: {method}")
        evaluated.append(result)
    return evaluated


def order_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_groups[row["scene_uid"]].append(row)
    if method == "explicit_pairs":
        exact_key = "derived_order_exact"
        structure_key = "pairwise_complete_valid"
    else:
        exact_key = "order_exact"
        structure_key = "order_permutation_valid"
    correct = sum(row["pairwise_correct"] for row in rows)
    total = sum(row["pairwise_total"] for row in rows)
    return {
        "calls": len(rows),
        "scenes": len(scene_groups),
        "full_order_exact_count": sum(bool(row[exact_key]) for row in rows),
        "full_order_exact_pct": pct(sum(bool(row[exact_key]) for row in rows), len(rows)),
        "all_object_pairs_correct": correct,
        "all_object_pairs_total": total,
        "all_object_pairs_accuracy_pct": pct(correct, total),
        "response_structure_valid_pct": pct(sum(bool(row[structure_key]) for row in rows), len(rows)),
        "visible_ids_exact_pct": pct(sum(bool(row["visible_ids_exact"]) for row in rows), len(rows)),
        "format_valid_pct": pct(sum(bool(row["format_valid"]) for row in rows), len(rows)),
        "scenes_5of5": sum(len(group) == 5 and all(bool(row[exact_key]) for row in group) for group in scene_groups.values()),
        "scenes_1to4of5": sum(any(bool(row[exact_key]) for row in group) and not all(bool(row[exact_key]) for row in group) for group in scene_groups.values()),
        "scenes_0of5": sum(not any(bool(row[exact_key]) for row in group) for group in scene_groups.values()),
    }


def native_target_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_groups[row["scene_uid"]].append(row)
    correct = sum(row["relation_correct"] for row in rows)
    total = sum(row["relation_total"] for row in rows)
    exact = sum(bool(row["target_partition_exact"]) for row in rows)
    return {
        "calls": len(rows),
        "scenes": len(scene_groups),
        "target_relation_correct": correct,
        "target_relation_total": total,
        "target_relation_accuracy_pct": pct(correct, total),
        "target_partition_exact_count": exact,
        "target_partition_exact_pct": pct(exact, len(rows)),
        "response_structure_valid_pct": pct(sum(bool(row["relations_complete_valid"]) for row in rows), len(rows)),
        "visible_ids_exact_pct": pct(sum(bool(row["visible_ids_exact"]) for row in rows), len(rows)),
        "format_valid_pct": pct(sum(bool(row["format_valid"]) for row in rows), len(rows)),
    }


def same_target_summary(
    source_rows: list[dict[str, Any]],
    target_rows: dict[tuple[str, int], dict[str, Any]],
    evaluator: Callable[[dict[str, Any], dict[str, Any]], tuple[int, int, bool]],
) -> dict[str, Any]:
    correct = total = exact = 0
    matched = 0
    for source in source_rows:
        target = target_rows.get((source["scene_uid"], int(source["seed"])))
        if target is None:
            continue
        c, t, e = evaluator(target, source)
        correct += c
        total += t
        exact += int(e)
        matched += 1
    return {
        "calls": matched,
        "target_relation_correct": correct,
        "target_relation_total": total,
        "target_relation_accuracy_pct": pct(correct, total),
        "target_partition_exact_count": exact,
        "target_partition_exact_pct": pct(exact, matched),
    }


def baseline_rows(scene_type: str, method: str) -> list[dict[str, Any]]:
    path = ROOT / "experiments" / RGB_BASELINES[scene_type][method] / "results/evaluated_records.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def rgbd_subset_for_baseline(
    rows: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_key = {(row["scene_id"], int(row["seed"])): row for row in baseline}
    rgbd_by_key = {(row["source_scene_id"], int(row["seed"])): row for row in rows}
    common = sorted(set(baseline_by_key) & set(rgbd_by_key))
    if len(common) != len(baseline):
        raise ValueError(f"paired baseline mismatch: common={len(common)} baseline={len(baseline)}")
    return [baseline_by_key[key] for key in common], [rgbd_by_key[key] for key in common]


def paired_order_comparison(
    scene_type: str,
    method: str,
    rgbd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rgb, rgbd = rgbd_subset_for_baseline(rgbd_rows, baseline_rows(scene_type, method))
    exact_key = "derived_order_exact" if method == "explicit_pairs" else "order_exact"
    rgb_exact = sum(bool(row[exact_key]) for row in rgb)
    rgbd_exact = sum(bool(row[exact_key]) for row in rgbd)
    rgb_pair_c = sum(row["pairwise_correct"] for row in rgb)
    rgb_pair_t = sum(row["pairwise_total"] for row in rgb)
    rgbd_pair_c = sum(row["pairwise_correct"] for row in rgbd)
    rgbd_pair_t = sum(row["pairwise_total"] for row in rgbd)
    return {
        "scene_type": scene_type,
        "method": method,
        "paired_calls": len(rgb),
        "paired_scenes": len({row["scene_id"] for row in rgb}),
        "rgb_full_order_exact_pct": pct(rgb_exact, len(rgb)),
        "rgbd_full_order_exact_pct": pct(rgbd_exact, len(rgbd)),
        "full_order_delta_pp": round(pct(rgbd_exact, len(rgbd)) - pct(rgb_exact, len(rgb)), 3),
        "rgb_all_pairs_accuracy_pct": pct(rgb_pair_c, rgb_pair_t),
        "rgbd_all_pairs_accuracy_pct": pct(rgbd_pair_c, rgbd_pair_t),
        "all_pairs_delta_pp": round(pct(rgbd_pair_c, rgbd_pair_t) - pct(rgb_pair_c, rgb_pair_t), 3),
        "rgbd_wins": sum(not bool(a[exact_key]) and bool(b[exact_key]) for a, b in zip(rgb, rgbd)),
        "rgb_wins": sum(bool(a[exact_key]) and not bool(b[exact_key]) for a, b in zip(rgb, rgbd)),
    }


def paired_native_target_comparison(
    scene_type: str, rgbd_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rgb, rgbd = rgbd_subset_for_baseline(rgbd_rows, baseline_rows(scene_type, "single_target"))
    rgb_c = sum(row["relation_correct"] for row in rgb)
    rgb_t = sum(row["relation_total"] for row in rgb)
    rgbd_c = sum(row["relation_correct"] for row in rgbd)
    rgbd_t = sum(row["relation_total"] for row in rgbd)
    rgb_exact = sum(bool(row["target_partition_exact"]) for row in rgb)
    rgbd_exact = sum(bool(row["target_partition_exact"]) for row in rgbd)
    return {
        "scene_type": scene_type,
        "method": "single_target",
        "paired_calls": len(rgb),
        "paired_scenes": len({row["scene_id"] for row in rgb}),
        "rgb_target_relation_accuracy_pct": pct(rgb_c, rgb_t),
        "rgbd_target_relation_accuracy_pct": pct(rgbd_c, rgbd_t),
        "target_relation_delta_pp": round(pct(rgbd_c, rgbd_t) - pct(rgb_c, rgb_t), 3),
        "rgb_target_partition_exact_pct": pct(rgb_exact, len(rgb)),
        "rgbd_target_partition_exact_pct": pct(rgbd_exact, len(rgbd)),
        "target_partition_delta_pp": round(pct(rgbd_exact, len(rgbd)) - pct(rgb_exact, len(rgb)), 3),
        "rgbd_wins": sum(not bool(a["target_partition_exact"]) and bool(b["target_partition_exact"]) for a, b in zip(rgb, rgbd)),
        "rgb_wins": sum(bool(a["target_partition_exact"]) and not bool(b["target_partition_exact"]) for a, b in zip(rgb, rgbd)),
    }


def paired_derived_target_comparison(
    scene_type: str,
    method: str,
    rgbd_source_rows: list[dict[str, Any]],
    rgbd_target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rgb_source, rgbd_source = rgbd_subset_for_baseline(rgbd_source_rows, baseline_rows(scene_type, method))
    rgb_target, rgbd_target = rgbd_subset_for_baseline(rgbd_target_rows, baseline_rows(scene_type, "single_target"))
    rgb_source_by_key = {(row["scene_id"], int(row["seed"])): row for row in rgb_source}
    rgbd_source_by_key = {(row["source_scene_id"], int(row["seed"])): row for row in rgbd_source}
    rgb_target_by_key = {(row["scene_id"], int(row["seed"])): row for row in rgb_target}
    rgbd_target_by_key = {(row["source_scene_id"], int(row["seed"])): row for row in rgbd_target}
    common = sorted(set(rgb_source_by_key) & set(rgb_target_by_key) & set(rgbd_source_by_key) & set(rgbd_target_by_key))
    evaluator = anchor_eval.relation_correct_from_pairs if method == "explicit_pairs" else anchor_eval.relation_correct_from_order
    rgb_c = rgb_t = rgb_exact = rgbd_c = rgbd_t = rgbd_exact = 0
    for key in common:
        c, t, e = evaluator(rgb_target_by_key[key], rgb_source_by_key[key])
        rgb_c += c
        rgb_t += t
        rgb_exact += int(e)
        c, t, e = evaluator(rgbd_target_by_key[key], rgbd_source_by_key[key])
        rgbd_c += c
        rgbd_t += t
        rgbd_exact += int(e)
    return {
        "scene_type": scene_type,
        "method": method,
        "paired_calls": len(common),
        "rgb_target_relation_accuracy_pct": pct(rgb_c, rgb_t),
        "rgbd_target_relation_accuracy_pct": pct(rgbd_c, rgbd_t),
        "target_relation_delta_pp": round(pct(rgbd_c, rgbd_t) - pct(rgb_c, rgb_t), 3),
        "rgb_target_partition_exact_pct": pct(rgb_exact, len(common)),
        "rgbd_target_partition_exact_pct": pct(rgbd_exact, len(common)),
        "target_partition_delta_pp": round(pct(rgbd_exact, len(common)) - pct(rgb_exact, len(common)), 3),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    config = read_json(experiment / "config/experiment_config.json")
    frozen = read_json(experiment / "config/frozen_scenes.json")
    configured_methods = list(config["methods"])
    order_methods = [method for method in ORDER_METHODS if method in configured_methods]
    scene_labels = config.get("scene_labels") or {
        key: SCENE_LABELS.get(key, key.replace("_", " ").title())
        for key in sorted({row["scene_type"] for row in frozen})
    }
    raw = read_jsonl(experiment / "logs/runs.jsonl")
    expected_calls = int(config["scheduled_calls"])
    if len(raw) != expected_calls:
        raise ValueError(f"expected {expected_calls} rows, found {len(raw)}")
    if len({row["run_id"] for row in raw}) != len(raw):
        raise ValueError("duplicate run IDs")
    evaluated = evaluate_rows(raw)
    results = experiment / "results"
    reports = experiment / "reports"
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_jsonl(results / "evaluated_records.jsonl", evaluated)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[(row["scene_type"], row["method"])].append(row)

    order_rows = []
    target_rows = []
    target_maps = {
        scene_type: {
            (row["scene_uid"], int(row["seed"])): row
            for row in groups[(scene_type, "single_target")]
        }
        for scene_type in scene_labels
    }
    for scene_type in scene_labels:
        for method in order_methods:
            source = groups[(scene_type, method)]
            order_rows.append({"scene_type": scene_type, "method": method, **order_summary(source, method)})
            evaluator = anchor_eval.relation_correct_from_pairs if method == "explicit_pairs" else anchor_eval.relation_correct_from_order
            target_rows.append({
                "scene_type": scene_type,
                "method": method,
                **same_target_summary(source, target_maps[scene_type], evaluator),
            })
        if "single_target" in configured_methods:
            target_rows.append({
                "scene_type": scene_type,
                "method": "single_target",
                **native_target_summary(groups[(scene_type, "single_target")]),
            })

    paired_order = []
    paired_target = []
    if not config.get("skip_external_baseline_comparison", False):
        for scene_type in scene_labels:
            for method in order_methods:
                paired_order.append(paired_order_comparison(scene_type, method, groups[(scene_type, method)]))
                paired_target.append(paired_derived_target_comparison(
                    scene_type,
                    method,
                    groups[(scene_type, method)],
                    groups[(scene_type, "single_target")],
                ))
            paired_target.append(paired_native_target_comparison(scene_type, groups[(scene_type, "single_target")]))

    scene_method_rows = []
    scene_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        scene_groups[(row["scene_uid"], row["method"])].append(row)
    for (scene_uid, method), rows in sorted(scene_groups.items()):
        first = rows[0]
        common = {
            "scene_uid": scene_uid,
            "scene_type": first["scene_type"],
            "source_scene_id": first["source_scene_id"],
            "method": method,
            "visible_ids": json.dumps(first["visible_ids"]),
            "expected_order": json.dumps(first["expected_order"]),
            "target_id": first["target_id"],
        }
        if method == "single_target":
            scene_method_rows.append({
                **common,
                "successes_out_of_5": sum(bool(row["target_partition_exact"]) for row in rows),
                "component_accuracy_pct": pct(
                    sum(row["relation_correct"] for row in rows),
                    sum(row["relation_total"] for row in rows),
                ),
                "metric": "target relations",
            })
        else:
            exact_key = "derived_order_exact" if method == "explicit_pairs" else "order_exact"
            scene_method_rows.append({
                **common,
                "successes_out_of_5": sum(bool(row[exact_key]) for row in rows),
                "component_accuracy_pct": pct(
                    sum(row["pairwise_correct"] for row in rows),
                    sum(row["pairwise_total"] for row in rows),
                ),
                "metric": "all object pairs",
            })

    expected_images = int(config["runtime"].get("images_per_prompt", 2))
    expected_order = ["numbered_rgb"]
    if expected_images == 2:
        expected_order.append(
            "numbered_grayscale_depth"
            if config.get("depth_markers", {}).get("present")
            else "grayscale_depth"
        )
    audit = {
        "scheduled_calls": expected_calls,
        "observed_calls": len(raw),
        "unique_run_ids": len({row["run_id"] for row in raw}),
        "scene_count": len(frozen),
        "scene_type_counts": {
            scene_type: sum(row["scene_type"] == scene_type for row in frozen)
            for scene_type in scene_labels
        },
        "method_counts": {
            method: sum(row["method"] == method for row in raw)
            for method in METHOD_LABELS
        },
        "all_batch_size_one": all(row["runtime"]["batch_size"] == 1 for row in raw),
        "images_per_prompt": expected_images,
        "all_images_in_expected_order": all(
            row["runtime"]["images_per_prompt"] == expected_images
            and row["image_order"] == expected_order
            for row in raw
        ),
        "depth_annotator": config.get("depth_convention", {}).get("annotator"),
        "depth_measure": config.get("depth_convention", {}).get("measure"),
        "all_finished_stop": all(row["finish_reason"] == "stop" for row in raw),
        "max_input_tokens": max(row["input_tokens"] for row in raw),
        "max_output_tokens": max(row["output_tokens"] for row in raw),
        "max_context_tokens_if_full_cap": max(row["context_tokens_if_full_cap"] for row in raw),
        "max_model_len": config["runtime"]["max_model_len"],
        "format_failures": sum(not row["format_valid"] for row in evaluated),
    }
    summary = {
        "order_summary_all_40_scenes": order_rows,
        "same_target_summary_all_40_scenes": target_rows,
        "paired_rgb_vs_rgbd_order": paired_order,
        "paired_rgb_vs_rgbd_same_target": paired_target,
        "audit": audit,
    }
    write_json(results / "summary.json", summary)
    write_csv(results / "order_summary_all40.csv", order_rows)
    write_csv(results / "same_target_summary_all40.csv", target_rows)
    write_csv(results / "paired_rgb_vs_rgbd_order.csv", paired_order)
    write_csv(results / "paired_rgb_vs_rgbd_same_target.csv", paired_target)
    write_csv(results / "scene_method_summary.csv", scene_method_rows)
    write_json(experiment / "audit/final_audit.json", audit)

    input_label = config.get("input_condition", "rgb_plus_gray")
    report = [
        f"# {config.get('report_title', 'N3 prompt comparison')} — {input_label}",
        "",
        f"- Model: `{config['model_id']}`",
        f"- Design: {len(frozen)} scenes × {len(config['methods'])} prompts × {len(config['main_seeds'])} seeds = {len(raw):,} calls",
        f"- Input: {config.get('input', input_label)}",
        "- Forced JSON; temperature 0.3; batch size 1",
        "",
        "## Result — complete order",
        "",
    ]
    report += markdown_table(
        ["Scene type", "Prompt", "Full-order exact", "All-pairs accuracy", "5/5 scenes", "0/5 scenes"],
        [[
            scene_labels[row["scene_type"]], METHOD_LABELS[row["method"]],
            f"{row['full_order_exact_pct']:.3f}%", f"{row['all_object_pairs_accuracy_pct']:.3f}%",
            row["scenes_5of5"], row["scenes_0of5"],
        ] for row in order_rows],
    )
    report += ["", "## Output compliance — complete-order prompts", ""]
    report += markdown_table(
        ["Scene type", "Prompt", "visible_ids exact", "Response structure valid", "JSON valid"],
        [[
            scene_labels[row["scene_type"]], METHOD_LABELS[row["method"]],
            f"{row['visible_ids_exact_pct']:.3f}%", f"{row['response_structure_valid_pct']:.3f}%",
            f"{row['format_valid_pct']:.3f}%",
        ] for row in order_rows],
    )
    report += ["", "## Result — same fixed target", ""]
    report += markdown_table(
        ["Scene type", "Prompt", "Relation accuracy", "All relations exact"],
        [[
            scene_labels[row["scene_type"]], METHOD_LABELS[row["method"]],
            f"{row['target_relation_accuracy_pct']:.3f}%", f"{row['target_partition_exact_pct']:.3f}%",
        ] for row in target_rows],
    )
    if paired_order:
        report += [
            "",
            "## Paired RGB-only → RGB-D comparison — complete order",
            "",
            "Occlusion comparisons use the original 20 shared scenes; the other two use all 40 shared scenes.",
            "",
        ]
        report += markdown_table(
            ["Scene type", "Prompt", "Calls", "Exact RGB→RGB-D", "Δ exact", "Pairs RGB→RGB-D", "Δ pairs"],
            [[
                scene_labels[row["scene_type"]], METHOD_LABELS[row["method"]], row["paired_calls"],
                f"{row['rgb_full_order_exact_pct']:.3f}→{row['rgbd_full_order_exact_pct']:.3f}%",
                f"{row['full_order_delta_pp']:+.3f} pp",
                f"{row['rgb_all_pairs_accuracy_pct']:.3f}→{row['rgbd_all_pairs_accuracy_pct']:.3f}%",
                f"{row['all_pairs_delta_pp']:+.3f} pp",
            ] for row in paired_order],
        )
        report += ["", "## Paired RGB-only → RGB-D comparison — same fixed target", ""]
        report += markdown_table(
            ["Scene type", "Prompt", "Calls", "Relations RGB→RGB-D", "Δ relations", "Exact RGB→RGB-D", "Δ exact"],
            [[
                scene_labels[row["scene_type"]], METHOD_LABELS[row["method"]], row["paired_calls"],
                f"{row['rgb_target_relation_accuracy_pct']:.3f}→{row['rgbd_target_relation_accuracy_pct']:.3f}%",
                f"{row['target_relation_delta_pp']:+.3f} pp",
                f"{row['rgb_target_partition_exact_pct']:.3f}→{row['rgbd_target_partition_exact_pct']:.3f}%",
                f"{row['target_partition_delta_pp']:+.3f} pp",
            ] for row in paired_target],
        )
    report += [
        "",
        "## Metric meaning",
        "",
        "- Full-order exact: the entire closest-to-farthest ID array must match; one inversion makes the call fail.",
        "- All-pairs accuracy: among every unordered object pair, the predicted closer object is correct.",
        "- Target relation accuracy: for the same fixed target, each other object is correctly labeled front/behind.",
        "- All relations exact: every relation to that target must be correct in the call.",
        "",
        "## Audit",
        "",
        f"- Calls/run IDs: {audit['observed_calls']}/{audit['unique_run_ids']}",
        f"- Batch size 1: {audit['all_batch_size_one']}",
        f"- Images per prompt / fixed order: {audit['images_per_prompt']} / {audit['all_images_in_expected_order']}",
        f"- All finish reason `stop`: {audit['all_finished_stop']}",
        f"- Format failures: {audit['format_failures']}",
        f"- Maximum input/output tokens: {audit['max_input_tokens']}/{audit['max_output_tokens']}",
        f"- Maximum context with full output cap: {audit['max_context_tokens_if_full_cap']} / {audit['max_model_len']}",
        "",
    ]
    (reports / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "evaluated_calls": len(evaluated),
        "order_rows": len(order_rows),
        "target_rows": len(target_rows),
        "paired_order_rows": len(paired_order),
        "paired_target_rows": len(paired_target),
        "report": str(reports / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
