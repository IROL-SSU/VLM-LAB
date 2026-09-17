#!/usr/bin/env python3
"""Build the complete 40-scene RGB/Gray/Numbered-Gray N3 comparison."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import analyze_n3_clear_depth_anchor_target as anchor_eval
import analyze_n3_rgbd_prompt_methods as base


ROOT = Path("/home/ssu/ShelfScene")
EXPERIMENTS = ROOT / "experiments"
SUPPLEMENT = EXPERIMENTS / "n3_occlusion_rgb_missing20_5methods_5seeds_20260907"
OUTPUT = EXPERIMENTS / "n3_depth_input_ablation_3conditions_full40_20260907"
GRAY = EXPERIMENTS / "n3_rgbd_3scene_40each_5methods_5seeds_20260831"
NUMBERED_GRAY = (
    EXPERIMENTS / "n3_numbered_depth_3scene_40each_5methods_5seeds_20260831"
)
ORDER_METHODS = ["minimal_direct", "detailed_direct", "contact_point", "explicit_pairs"]
TARGET_METHODS = ORDER_METHODS + ["single_target"]
SCENE_TYPES = ["separated_nonoverlap", "packed_nonoverlap", "structured_occlusion"]
INPUTS = ["numbered_rgb", "rgb_plus_gray", "rgb_plus_numbered_gray"]
INPUT_LABELS = {
    "numbered_rgb": "Numbered RGB",
    "rgb_plus_gray": "RGB + Gray",
    "rgb_plus_numbered_gray": "RGB + Numbered Gray",
}
SCENE_LABELS = {
    "separated_nonoverlap": "Separated non-occlusion",
    "packed_nonoverlap": "Packed non-occlusion",
    "structured_occlusion": "Structured occlusion",
}
METHOD_LABELS = base.METHOD_LABELS

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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def normalize_rgb_row(row: dict[str, Any], scene_type: str, method: str) -> dict[str, Any]:
    result = dict(row)
    result["method"] = method
    result["scene_type"] = scene_type
    result["source_scene_id"] = row["scene_id"]
    result["scene_uid"] = f"{scene_type}__{row['scene_id']}"
    return result


def load_rgb_groups() -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    supplement = read_jsonl(SUPPLEMENT / "results/evaluated_records.jsonl")
    supplement_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in supplement:
        supplement_groups[row["method"]].append(row)

    for scene_type in SCENE_TYPES:
        for method in TARGET_METHODS:
            baseline_path = (
                EXPERIMENTS
                / RGB_BASELINES[scene_type][method]
                / "results/evaluated_records.jsonl"
            )
            rows = [
                normalize_rgb_row(row, scene_type, method)
                for row in read_jsonl(baseline_path)
            ]
            if scene_type == "structured_occlusion":
                rows.extend(supplement_groups[method])
            groups[(scene_type, method)] = rows
    return groups


def load_multi_input_groups(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path / "results/evaluated_records.jsonl"):
        groups[(row["scene_type"], row["method"])].append(row)
    return dict(groups)


def summaries(
    groups: dict[tuple[str, str], list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order_rows = []
    target_rows = []
    for scene_type in SCENE_TYPES:
        target_map = {
            (row["scene_uid"], int(row["seed"])): row
            for row in groups[(scene_type, "single_target")]
        }
        for method in ORDER_METHODS:
            rows = groups[(scene_type, method)]
            order_rows.append(
                {
                    "scene_type": scene_type,
                    "method": method,
                    **base.order_summary(rows, method),
                }
            )
            evaluator = (
                anchor_eval.relation_correct_from_pairs
                if method == "explicit_pairs"
                else anchor_eval.relation_correct_from_order
            )
            target_rows.append(
                {
                    "scene_type": scene_type,
                    "method": method,
                    **base.same_target_summary(rows, target_map, evaluator),
                }
            )
        target_rows.append(
            {
                "scene_type": scene_type,
                "method": "single_target",
                **base.native_target_summary(groups[(scene_type, "single_target")]),
            }
        )
    return order_rows, target_rows


def group_signature(
    rows: list[dict[str, Any]], *, include_target: bool = False
) -> set[tuple[Any, ...]]:
    signatures = set()
    for row in rows:
        signature: tuple[Any, ...] = (
            row.get("source_scene_id", row["scene_id"]),
            tuple(row["visible_ids"]),
            tuple(row["expected_order"]),
        )
        if include_target:
            signature += (int(row["target_id"]),)
        signatures.add(signature)
    return signatures


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped_by_input = {
        "numbered_rgb": load_rgb_groups(),
        "rgb_plus_gray": load_multi_input_groups(GRAY),
        "rgb_plus_numbered_gray": load_multi_input_groups(NUMBERED_GRAY),
    }

    order_by_input = {}
    target_by_input = {}
    for input_name, groups in grouped_by_input.items():
        order_rows, target_rows = summaries(groups)
        order_by_input[input_name] = {
            (row["scene_type"], row["method"]): row for row in order_rows
        }
        target_by_input[input_name] = {
            (row["scene_type"], row["method"]): row for row in target_rows
        }

    order_matrix = []
    for scene_type in SCENE_TYPES:
        for method in ORDER_METHODS:
            row: dict[str, Any] = {"scene_type": scene_type, "method": method}
            for input_name in INPUTS:
                value = order_by_input[input_name][(scene_type, method)]
                prefix = input_name
                row[f"{prefix}_calls"] = value["calls"]
                row[f"{prefix}_scenes"] = value["scenes"]
                row[f"{prefix}_full_order_exact_pct"] = value[
                    "full_order_exact_pct"
                ]
                row[f"{prefix}_all_pairs_accuracy_pct"] = value[
                    "all_object_pairs_accuracy_pct"
                ]
            order_matrix.append(row)

    target_matrix = []
    for scene_type in SCENE_TYPES:
        for method in TARGET_METHODS:
            row = {"scene_type": scene_type, "method": method}
            for input_name in INPUTS:
                value = target_by_input[input_name][(scene_type, method)]
                prefix = input_name
                row[f"{prefix}_calls"] = value["calls"]
                row[f"{prefix}_target_partition_exact_pct"] = value[
                    "target_partition_exact_pct"
                ]
                row[f"{prefix}_target_relation_accuracy_pct"] = value[
                    "target_relation_accuracy_pct"
                ]
            target_matrix.append(row)

    pooled = []
    for input_name in INPUTS:
        order_values = list(order_by_input[input_name].values())
        target_values = list(target_by_input[input_name].values())
        order_calls = sum(row["calls"] for row in order_values)
        pair_correct = sum(row["all_object_pairs_correct"] for row in order_values)
        pair_total = sum(row["all_object_pairs_total"] for row in order_values)
        target_calls = sum(row["calls"] for row in target_values)
        relation_correct = sum(row["target_relation_correct"] for row in target_values)
        relation_total = sum(row["target_relation_total"] for row in target_values)
        pooled.append(
            {
                "input": input_name,
                "full_order_calls": order_calls,
                "full_order_exact_pct": base.pct(
                    sum(row["full_order_exact_count"] for row in order_values),
                    order_calls,
                ),
                "all_pairs_accuracy_pct": base.pct(pair_correct, pair_total),
                "fixed_target_calls": target_calls,
                "target_partition_exact_pct": base.pct(
                    sum(row["target_partition_exact_count"] for row in target_values),
                    target_calls,
                ),
                "target_relation_accuracy_pct": base.pct(
                    relation_correct, relation_total
                ),
            }
        )

    signatures_match = {}
    for scene_type in SCENE_TYPES:
        scene_reference = group_signature(
            grouped_by_input["numbered_rgb"][(scene_type, "single_target")]
        )
        target_reference = group_signature(
            grouped_by_input["numbered_rgb"][(scene_type, "single_target")],
            include_target=True,
        )
        signatures_match[scene_type] = {
            "scene_and_gt": all(
            group_signature(grouped_by_input[input_name][(scene_type, method)])
            == scene_reference
            for input_name in INPUTS
            for method in TARGET_METHODS
            ),
            "single_target_id": all(
                group_signature(
                    grouped_by_input[input_name][(scene_type, "single_target")],
                    include_target=True,
                )
                == target_reference
                for input_name in INPUTS
            ),
        }

    all_rows = [
        row
        for groups in grouped_by_input.values()
        for rows in groups.values()
        for row in rows
    ]
    supplement_audit = read_json(SUPPLEMENT / "audit/final_audit.json")
    audit = {
        "supplement_expected_calls": 500,
        "supplement_observed_calls": supplement_audit["observed_calls"],
        "supplement_unique_run_ids": supplement_audit["unique_run_ids"],
        "all_order_cells_have_40_scenes_200_calls": all(
            row[f"{input_name}_scenes"] == 40
            and row[f"{input_name}_calls"] == 200
            for row in order_matrix
            for input_name in INPUTS
        ),
        "all_target_cells_have_200_calls": all(
            row[f"{input_name}_calls"] == 200
            for row in target_matrix
            for input_name in INPUTS
        ),
        "scene_gt_target_signatures_match_across_inputs_and_methods": signatures_match,
        "all_finish_reasons_stop": all(row["finish_reason"] == "stop" for row in all_rows),
        "all_batch_size_one": all(row["runtime"]["batch_size"] == 1 for row in all_rows),
        "format_failures": sum(not bool(row["format_valid"]) for row in all_rows),
        "full_order_matrix_rows": len(order_matrix),
        "fixed_target_matrix_rows": len(target_matrix),
    }

    result = {
        "design": {
            "scene_types": SCENE_TYPES,
            "scenes_per_type": 40,
            "seeds_per_scene": 5,
            "inputs": INPUTS,
            "full_order_methods": ORDER_METHODS,
            "fixed_target_methods": TARGET_METHODS,
            "historical_reason_for_20_scene_subset": (
                "The first RGB-only structured-occlusion prompt comparison was run on "
                "20 scenes; later Gray and Numbered-Gray experiments expanded the source "
                "to 40, so prior paired reporting used only the shared 20."
            ),
        },
        "pooled_full40": pooled,
        "full_order_matrix": order_matrix,
        "fixed_target_matrix": target_matrix,
        "audit": audit,
    }
    write_json(OUTPUT / "results/summary.json", result)
    write_json(OUTPUT / "audit/final_audit.json", audit)
    write_json(
        OUTPUT / "config/source_experiments.json",
        {
            "rgb_missing20_supplement": str(SUPPLEMENT),
            "gray": str(GRAY),
            "numbered_gray": str(NUMBERED_GRAY),
            "original_rgb_baselines": RGB_BASELINES,
        },
    )
    write_csv(OUTPUT / "results/full_order_matrix.csv", order_matrix)
    write_csv(OUTPUT / "results/fixed_target_matrix.csv", target_matrix)
    write_csv(OUTPUT / "results/pooled_full40.csv", pooled)

    lines = [
        "# N3 depth-input ablation — complete 40-scene comparison",
        "",
        "The prior 20-scene occlusion restriction was historical: the first RGB-only "
        "prompt comparison stopped at 20 scenes, while both depth-input experiments "
        "already contained all 40. The missing 20 RGB-only scenes have now been run.",
        "",
        "- Three scene families × 40 scenes each",
        "- Three input conditions",
        "- Four full-order prompts; five fixed-target comparison methods",
        "- Five seeds per scene/method/input",
        "",
        "## Pooled 40-scene results",
        "",
        "| Input | Full-order exact | All-pairs | Target exact | Target relations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {INPUT_LABELS[row['input']]} | {row['full_order_exact_pct']:.3f}% | "
            f"{row['all_pairs_accuracy_pct']:.3f}% | "
            f"{row['target_partition_exact_pct']:.3f}% | "
            f"{row['target_relation_accuracy_pct']:.3f}% |"
        )
    lines += [
        "",
        "## Full-order: Exact / All-pairs",
        "",
        "| Scene family | Prompt | Numbered RGB | RGB + Gray | RGB + Numbered Gray |",
        "|---|---|---:|---:|---:|",
    ]
    for row in order_matrix:
        cells = []
        for input_name in INPUTS:
            cells.append(
                f"{row[f'{input_name}_full_order_exact_pct']:.3f} / "
                f"{row[f'{input_name}_all_pairs_accuracy_pct']:.3f}%"
            )
        lines.append(
            f"| {SCENE_LABELS[row['scene_type']]} | {METHOD_LABELS[row['method']]} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Same fixed target: Partition exact / Relation accuracy",
        "",
        "| Scene family | Prompt | Numbered RGB | RGB + Gray | RGB + Numbered Gray |",
        "|---|---|---:|---:|---:|",
    ]
    for row in target_matrix:
        cells = []
        for input_name in INPUTS:
            cells.append(
                f"{row[f'{input_name}_target_partition_exact_pct']:.3f} / "
                f"{row[f'{input_name}_target_relation_accuracy_pct']:.3f}%"
            )
        lines.append(
            f"| {SCENE_LABELS[row['scene_type']]} | {METHOD_LABELS[row['method']]} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Audit",
        "",
        f"- Missing RGB-only calls: {audit['supplement_observed_calls']}/500",
        f"- Every table cell has 40 scenes × 5 seeds: {audit['all_order_cells_have_40_scenes_200_calls']}",
        f"- Scene/GT/target signatures match: {all(all(value.values()) for value in signatures_match.values())}",
        f"- Format failures across source records: {audit['format_failures']}",
        "",
    ]
    report_path = OUTPUT / "reports/final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
