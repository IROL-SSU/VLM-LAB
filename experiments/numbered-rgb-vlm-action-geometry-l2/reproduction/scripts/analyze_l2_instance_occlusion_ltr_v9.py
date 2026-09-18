#!/usr/bin/env python3
"""Analyze left-to-right per-instance visual occlusion experiment v9."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_l2_grasp_zone_v3 import CONDITIONS, FAMILIES, NAMES, pct, read_json, read_jsonl
from analyze_l2_instance_interference_v8 import binary_metric, safe_div


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_instance_occlusion_ltr_v9"
V8_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_instance_interference_v8"
V7_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_potential_interference_v7"
OUTPUT = EXPERIMENT / "results/l2_instance_occlusion_ltr_v9"


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(candidate_rows: list[dict], run_rows: list[dict]) -> dict:
    candidate_metric = binary_metric(
        [bool(item["truth_occludes"]) for item in candidate_rows],
        [bool(item["predicted_occludes"]) for item in candidate_rows],
    )
    any_metric = binary_metric(
        [bool(item["truth_any_occlusion"]) for item in run_rows],
        [bool(item["predicted_any_occlusion"]) for item in run_rows],
    )
    return {
        "candidate_instance": candidate_metric,
        "run_count": len(run_rows),
        "full_output_exact_matches": sum(item["full_output_exact_match"] for item in run_rows),
        "full_output_exact_match_accuracy": safe_div(
            sum(item["full_output_exact_match"] for item in run_rows), len(run_rows)
        ),
        "occluder_set_exact_matches": sum(item["occluder_set_exact_match"] for item in run_rows),
        "occluder_set_exact_match_accuracy": safe_div(
            sum(item["occluder_set_exact_match"] for item in run_rows), len(run_rows)
        ),
        "visible_order_correct": sum(item["visible_order_correct"] for item in run_rows),
        "visible_order_accuracy": safe_div(
            sum(item["visible_order_correct"] for item in run_rows), len(run_rows)
        ),
        "target_id_correct": sum(item["target_id_correct"] for item in run_rows),
        "target_id_accuracy": safe_div(
            sum(item["target_id_correct"] for item in run_rows), len(run_rows)
        ),
        "scene_any_occlusion": any_metric,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(EXPERIMENT / "logs/runs.jsonl")
    schedule = [
        row for row in read_jsonl(EXPERIMENT / "config/run_table.jsonl")
        if row["level"] == "L2"
    ]
    attempts = read_jsonl(EXPERIMENT / "logs/attempts.jsonl")
    gt_cache = {}
    mapping_cache = {}
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id not in gt_cache:
            gt_cache[scene_id] = read_json(EXPERIMENT / row["ground_truth"])
            mapping_cache[scene_id] = read_json(EXPERIMENT / row["id_mapping"])

    candidate_rows = []
    run_rows = []
    invariant_failures = []
    for row in rows:
        gt = gt_cache[row["scene_id"]]["L2"]
        expected_rows = gt["occlusion_assessments_left_to_right"]
        actual_rows = row["parsed_response"]["occlusion_assessments_left_to_right"]
        expected = {
            int(item["object_id"]): bool(item["occludes_target"])
            for item in expected_rows
        }
        actual = {
            int(item["object_id"]): bool(item["occludes_target"])
            for item in actual_rows
        }
        expected_order = gt["visible_object_ids_left_to_right"]
        actual_order = row["parsed_response"]["visible_object_ids_left_to_right"]
        target_id = int(gt["target_id"])
        expected_occluders = sorted(object_id for object_id, value in expected.items() if value)
        predicted_occluders = sorted(object_id for object_id, value in actual.items() if value)
        if (
            actual_order != expected_order
            or [item["object_id"] for item in actual_rows]
            != [item["object_id"] for item in expected_rows]
            or row["parsed_response"]["target_id"] != target_id
        ):
            invariant_failures.append(row["run_id"])
        correct_count = 0
        for expected_item in expected_rows:
            object_id = int(expected_item["object_id"])
            truth = bool(expected_item["occludes_target"])
            predicted = actual.get(object_id)
            correct = predicted == truth
            correct_count += int(correct)
            candidate_rows.append({
                "run_id": row["run_id"],
                "scene_id": row["scene_id"],
                "scene_family": row["scene_family"],
                "condition": row["geometry_condition"],
                "resolution": row.get("direction_resolution"),
                "seed": row["seed"],
                "target_id": target_id,
                "object_id": object_id,
                "candidate_occlusion_pixels": expected_item["candidate_occlusion_pixels"],
                "candidate_occlusion_ratio": expected_item["candidate_occlusion_ratio"],
                "truth_occludes": truth,
                "predicted_occludes": predicted,
                "correct": correct,
            })
        full_exact = (
            actual == expected
            and actual_order == expected_order
            and row["parsed_response"]["target_id"] == target_id
        )
        run_rows.append({
            "run_id": row["run_id"],
            "scene_id": row["scene_id"],
            "scene_family": row["scene_family"],
            "condition": row["geometry_condition"],
            "resolution": row.get("direction_resolution"),
            "seed": row["seed"],
            "target_id": target_id,
            "visible_object_ids_left_to_right": json.dumps(expected_order),
            "candidate_count": len(expected),
            "correct_candidate_count": correct_count,
            "expected_occluder_ids": json.dumps(expected_occluders),
            "predicted_occluder_ids": json.dumps(predicted_occluders),
            "visible_order_correct": actual_order == expected_order,
            "target_id_correct": row["parsed_response"]["target_id"] == target_id,
            "full_output_exact_match": full_exact,
            "occluder_set_exact_match": predicted_occluders == expected_occluders,
            "truth_any_occlusion": bool(expected_occluders),
            "predicted_any_occlusion": bool(predicted_occluders),
            "any_occlusion_correct": bool(expected_occluders) == bool(predicted_occluders),
        })

    integrity_failures = []
    run_ids = [row["run_id"] for row in rows]
    scheduled_ids = {row["run_id"] for row in schedule}
    if len(rows) != 1500:
        integrity_failures.append(f"completed={len(rows)} expected=1500")
    if len(set(run_ids)) != len(run_ids):
        integrity_failures.append("duplicate run IDs")
    if set(run_ids) != scheduled_ids:
        integrity_failures.append("completed run IDs differ from L2 schedule")
    if invariant_failures:
        integrity_failures.append(f"output invariant failures={len(invariant_failures)}")
    cell_counts = Counter((row["scene_id"], condition_key(row)) for row in rows)
    if set(cell_counts.values()) != {5}:
        integrity_failures.append("scene-condition cells do not all have five seeds")

    overall = summarize(candidate_rows, run_rows)
    row_ids_by_condition = defaultdict(set)
    row_ids_by_family = defaultdict(set)
    for row in rows:
        row_ids_by_condition[condition_key(row)].add(row["run_id"])
        row_ids_by_family[row["scene_family"]].add(row["run_id"])

    condition_results = {}
    condition_csv = []
    for condition in CONDITIONS:
        ids = row_ids_by_condition[condition]
        result = summarize(
            [item for item in candidate_rows if item["run_id"] in ids],
            [item for item in run_rows if item["run_id"] in ids],
        )
        condition_results[f"{condition[0]}_{condition[1] or 'NA'}"] = result
        metric = result["candidate_instance"]
        condition_csv.append({
            "condition": condition[0], "resolution": condition[1],
            "name": NAMES[condition], "run_count": result["run_count"],
            "candidate_count": metric["n"],
            "tp": metric["tp"], "fn": metric["fn"],
            "tn": metric["tn"], "fp": metric["fp"],
            "candidate_accuracy": metric["accuracy"],
            "candidate_balanced_accuracy": metric["balanced_accuracy"],
            "occlusion_recall": metric["positive_recall"],
            "non_occlusion_specificity": metric["specificity"],
            "precision": metric["precision"],
            "full_output_exact_match_accuracy": result["full_output_exact_match_accuracy"],
            "any_occlusion_accuracy": result["scene_any_occlusion"]["accuracy"],
            "any_occlusion_balanced_accuracy": result["scene_any_occlusion"]["balanced_accuracy"],
        })

    family_results = {}
    family_csv = []
    for family in FAMILIES:
        ids = row_ids_by_family[family]
        result = summarize(
            [item for item in candidate_rows if item["run_id"] in ids],
            [item for item in run_rows if item["run_id"] in ids],
        )
        family_results[family] = result
        metric = result["candidate_instance"]
        family_csv.append({
            "scene_family": family, "run_count": result["run_count"],
            "candidate_count": metric["n"],
            "tp": metric["tp"], "fn": metric["fn"],
            "tn": metric["tn"], "fp": metric["fp"],
            "candidate_accuracy": metric["accuracy"],
            "candidate_balanced_accuracy": metric["balanced_accuracy"],
            "occlusion_recall": metric["positive_recall"],
            "non_occlusion_specificity": metric["specificity"],
            "full_output_exact_match_accuracy": result["full_output_exact_match_accuracy"],
            "any_occlusion_accuracy": result["scene_any_occlusion"]["accuracy"],
        })

    condition_family_csv = []
    for condition in CONDITIONS:
        for family in FAMILIES:
            ids = row_ids_by_condition[condition] & row_ids_by_family[family]
            result = summarize(
                [item for item in candidate_rows if item["run_id"] in ids],
                [item for item in run_rows if item["run_id"] in ids],
            )
            metric = result["candidate_instance"]
            condition_family_csv.append({
                "condition": condition[0], "resolution": condition[1],
                "name": NAMES[condition], "scene_family": family,
                "run_count": result["run_count"], "candidate_count": metric["n"],
                "candidate_accuracy": metric["accuracy"],
                "candidate_balanced_accuracy": metric["balanced_accuracy"],
                "occlusion_recall": metric["positive_recall"],
                "non_occlusion_specificity": metric["specificity"],
                "full_output_exact_match_accuracy": result["full_output_exact_match_accuracy"],
                "any_occlusion_accuracy": result["scene_any_occlusion"]["accuracy"],
            })

    scene_csv = []
    for scene_id in sorted(gt_cache):
        scene_candidates = [item for item in candidate_rows if item["scene_id"] == scene_id]
        scene_runs = [item for item in run_rows if item["scene_id"] == scene_id]
        result = summarize(scene_candidates, scene_runs)
        metric = result["candidate_instance"]
        scene_csv.append({
            "scene_id": scene_id, "scene_family": scene_runs[0]["scene_family"],
            "target_id": scene_runs[0]["target_id"],
            "visible_object_ids_left_to_right": scene_runs[0]["visible_object_ids_left_to_right"],
            "gt_occluder_ids": scene_runs[0]["expected_occluder_ids"],
            "run_count": result["run_count"], "candidate_count": metric["n"],
            "candidate_accuracy": metric["accuracy"],
            "occlusion_recall": metric["positive_recall"],
            "non_occlusion_specificity": metric["specificity"],
            "full_output_exact_match_accuracy": result["full_output_exact_match_accuracy"],
            "any_occlusion_accuracy": result["scene_any_occlusion"]["accuracy"],
        })

    grouped_candidates = defaultdict(list)
    for item in candidate_rows:
        grouped_candidates[(item["scene_id"], item["object_id"])].append(item)
    candidate_object_csv = []
    for (scene_id, object_id), items in sorted(grouped_candidates.items()):
        truth = bool(items[0]["truth_occludes"])
        predicted_count = sum(bool(item["predicted_occludes"]) for item in items)
        candidate_object_csv.append({
            "scene_id": scene_id, "scene_family": items[0]["scene_family"],
            "target_id": items[0]["target_id"], "object_id": object_id,
            "candidate_occlusion_pixels": items[0]["candidate_occlusion_pixels"],
            "candidate_occlusion_ratio": items[0]["candidate_occlusion_ratio"],
            "truth_occludes": truth, "run_count": len(items),
            "predicted_occlusion_count": predicted_count,
            "predicted_occlusion_rate": predicted_count / len(items),
            "accuracy": predicted_count / len(items) if truth else (len(items) - predicted_count) / len(items),
            "numbered_rgb": f"images/numbered_rgb/{scene_id}.png",
        })
    positive_objects = [item for item in candidate_object_csv if item["truth_occludes"]]
    worst_detected_positive = min(
        positive_objects, key=lambda item: item["predicted_occlusion_rate"]
    )
    best_detected_positive = max(
        positive_objects, key=lambda item: item["predicted_occlusion_rate"]
    )

    v8_rows = read_jsonl(V8_EXPERIMENT / "logs/runs.jsonl")
    v8_by_id = {row["run_id"]: row for row in v8_rows}
    v7_rows = read_jsonl(V7_EXPERIMENT / "logs/l2_potential_interference_v7/runs.jsonl")
    v7_by_id = {row["run_id"]: row for row in v7_rows}
    v8_candidate_predictions = {}
    for run_id, row in v8_by_id.items():
        v8_candidate_predictions[run_id] = {
            int(item["object_id"]): item["relation_to_target"] == "INTERFERES"
            for item in row["parsed_response"]["instance_assessments"]
            if item["relation_to_target"] != "TARGET"
        }
    v8_predictions = [
        v8_candidate_predictions[item["run_id"]][item["object_id"]]
        for item in candidate_rows
    ]
    labels = [bool(item["truth_occludes"]) for item in candidate_rows]
    v9_predictions = [bool(item["predicted_occludes"]) for item in candidate_rows]
    v8_run_any = {
        run_id: any(values.values()) for run_id, values in v8_candidate_predictions.items()
    }
    run_labels = [bool(item["truth_any_occlusion"]) for item in run_rows]
    prior_relaxed_results = read_json(
        V7_EXPERIMENT / "results/l2_potential_interference_v7/results.json"
    )["overall_same_relaxed_gt"]
    comparison = {
        "candidate_instance": {
            "v9_occlusion_ltr": binary_metric(labels, v9_predictions),
            "v8_interference": binary_metric(labels, v8_predictions),
            "v9_correct_v8_wrong": sum(
                prediction9 == label and prediction8 != label
                for label, prediction9, prediction8 in zip(labels, v9_predictions, v8_predictions)
            ),
            "v8_correct_v9_wrong": sum(
                prediction8 == label and prediction9 != label
                for label, prediction9, prediction8 in zip(labels, v9_predictions, v8_predictions)
            ),
        },
        "scene_any": {
            "v9_occlusion_ltr": binary_metric(
                run_labels, [bool(item["predicted_any_occlusion"]) for item in run_rows]
            ),
            "v8_interference": binary_metric(
                run_labels, [v8_run_any[item["run_id"]] for item in run_rows]
            ),
            "v7_boolean": binary_metric(
                run_labels,
                [
                    bool(v7_by_id[item["run_id"]]["parsed_response"]["potential_grasp_interference"])
                    for item in run_rows
                ],
            ),
            "v5_direct_occlusion": prior_relaxed_results["direct_occlusion_v5_rescored"],
            "original_l1_non_full": prior_relaxed_results["original_l1_non_FULL_rescored"],
        },
        "same_numbered_rgb_hash_v8": sum(
            row["numbered_rgb_sha256"] == v8_by_id[row["run_id"]]["numbered_rgb_sha256"]
            for row in rows
        ),
        "same_geometry_hash_v8": sum(
            row["geometry_json_sha256"] == v8_by_id[row["run_id"]]["geometry_json_sha256"]
            for row in rows
        ),
    }

    errors = [item for item in candidate_rows if not item["correct"]]
    integrity = {
        "status": "pass" if not integrity_failures else "fail",
        "scheduled": len(schedule), "completed": len(rows),
        "unique_run_ids": len(set(run_ids)),
        "parse_errors": sum(row["parse_error"] is not None for row in rows),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in rows),
        "output_invariant_failures": len(invariant_failures),
        "attempt_records": len(attempts),
        "infrastructure_errors": sum(item["status"] != "success" for item in attempts),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "failures": integrity_failures,
    }
    results = {
        "prompt_version": read_json(EXPERIMENT / "config/experiment_config.json")["l2_prompt_version"],
        "gt_definition": {
            "scope": "every numbered visible non-target instance",
            "ordering": "visible instance mask centroid x, then display ID",
            "candidate_rule": "candidate_occlusion_ratio >= 0.05",
            "candidate_instances_per_scene_set": 89,
            "positive_candidate_instances_per_scene_set": 15,
            "negative_candidate_instances_per_scene_set": 74,
        },
        "integrity": integrity, "overall": overall,
        "condition_results": condition_results, "family_results": family_results,
        "comparison": comparison, "candidate_error_count": len(errors),
        "representative_positive_objects": {
            "lowest_detection_rate": worst_detected_positive,
            "highest_detection_rate": best_detected_positive,
        },
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT / "condition_metrics.csv", condition_csv)
    write_csv(OUTPUT / "family_metrics.csv", family_csv)
    write_csv(OUTPUT / "condition_family_metrics.csv", condition_family_csv)
    write_csv(OUTPUT / "scene_metrics.csv", scene_csv)
    write_csv(OUTPUT / "run_outcomes.csv", sorted(run_rows, key=lambda item: item["run_id"]))
    write_csv(OUTPUT / "candidate_instance_outcomes.csv", sorted(candidate_rows, key=lambda item: (item["run_id"], item["object_id"])))
    write_csv(OUTPUT / "candidate_object_metrics.csv", candidate_object_csv)
    write_csv(OUTPUT / "candidate_instance_errors.csv", sorted(errors, key=lambda item: (item["run_id"], item["object_id"])))

    candidate = overall["candidate_instance"]
    any_scene = overall["scene_any_occlusion"]
    report = [
        "# L2 instance occlusion left-to-right v9", "", "## 실험 정의", "",
        "보이는 번호 물체를 visible-mask 중심의 x좌표로 왼쪽부터 나열하고, target을 제외한 각 instance가 target을 시각적으로 가리는지만 Boolean으로 평가한다.",
        "", "프롬프트에는 수치 임계값을 노출하지 않았다. GT는 후보 instance가 target-only silhouette의 5% 이상을 가리면 양성이다.",
        "", "## 무결성", "",
        f"- 완료: {integrity['completed']}/{integrity['scheduled']}",
        f"- 파싱 오류 / 후검증 실패 / 출력 불변식 실패: {integrity['parse_errors']} / {integrity['post_validation_failures']} / {integrity['output_invariant_failures']}",
        f"- 인프라 오류: {integrity['infrastructure_errors']}",
        "", "## 전체 결과", "",
        "| 평가 단위 | Accuracy | BA | Occlusion recall | Non-occlusion specificity | Precision | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 후보 instance | {pct(candidate['accuracy'])} | {pct(candidate['balanced_accuracy'])} | {pct(candidate['positive_recall'])} | {pct(candidate['specificity'])} | {pct(candidate['precision'])} | {candidate['tp']}/{candidate['fn']}/{candidate['tn']}/{candidate['fp']} |",
        f"| 장면 내 하나라도 가림 | {pct(any_scene['accuracy'])} | {pct(any_scene['balanced_accuracy'])} | {pct(any_scene['positive_recall'])} | {pct(any_scene['specificity'])} | {pct(any_scene['precision'])} | {any_scene['tp']}/{any_scene['fn']}/{any_scene['tn']}/{any_scene['fp']} |",
        "",
        f"- 전체 출력 완전일치: {overall['full_output_exact_matches']}/{overall['run_count']} ({pct(overall['full_output_exact_match_accuracy'])})",
        f"- Occluder ID 집합 완전일치: {overall['occluder_set_exact_matches']}/{overall['run_count']} ({pct(overall['occluder_set_exact_match_accuracy'])})",
        f"- 왼쪽→오른쪽 배열 및 target ID 정확도: {pct(overall['visible_order_accuracy'])} / {pct(overall['target_id_accuracy'])}",
        "", "## 조건별 결과", "",
        "| 조건 | 후보 정확도 | BA | 가림 recall | 비가림 specificity | 전체 출력 완전일치 | 장면 Boolean 정확도 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in condition_csv:
        report.append(
            f"| {item['condition']} {item['resolution'] or ''} {item['name']} | "
            f"{pct(item['candidate_accuracy'])} | {pct(item['candidate_balanced_accuracy'])} | "
            f"{pct(item['occlusion_recall'])} | {pct(item['non_occlusion_specificity'])} | "
            f"{pct(item['full_output_exact_match_accuracy'])} | {pct(item['any_occlusion_accuracy'])} |"
        )
    report.extend([
        "", "## 장면 종류별 결과", "",
        "| 장면 종류 | 후보 정확도 | BA | 가림 recall | 비가림 specificity | 전체 출력 완전일치 | 장면 Boolean 정확도 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in family_csv:
        report.append(
            f"| {item['scene_family']} | {pct(item['candidate_accuracy'])} | "
            f"{pct(item['candidate_balanced_accuracy']) if item['candidate_balanced_accuracy'] is not None else 'N/A'} | "
            f"{pct(item['occlusion_recall']) if item['occlusion_recall'] is not None else 'N/A'} | "
            f"{pct(item['non_occlusion_specificity'])} | {pct(item['full_output_exact_match_accuracy'])} | "
            f"{pct(item['any_occlusion_accuracy'])} |"
        )
    report.extend([
        "", "## 이전 formulation과 비교", "",
        "### 후보 instance 단위", "",
        "| 방식 | Accuracy | BA | Recall | Specificity | Precision | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("v9_occlusion_ltr", "v9 visual occlusion LTR"), ("v8_interference", "v8 grasp interference")):
        metric = comparison["candidate_instance"][key]
        report.append(
            f"| {label} | {pct(metric['accuracy'])} | {pct(metric['balanced_accuracy'])} | "
            f"{pct(metric['positive_recall'])} | {pct(metric['specificity'])} | {pct(metric['precision'])} | "
            f"{metric['tp']}/{metric['fn']}/{metric['tn']}/{metric['fp']} |"
        )
    report.extend([
        "", "### 장면 Boolean 단위", "",
        "| 방식 | Accuracy | BA | Recall | Specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, label in (
        ("v9_occlusion_ltr", "v9 visual occlusion LTR"),
        ("v8_interference", "v8 grasp interference"),
        ("v7_boolean", "v7 single Boolean"),
        ("v5_direct_occlusion", "v5 direct occlusion Boolean"),
        ("original_l1_non_full", "original L1 non-FULL"),
    ):
        metric = comparison["scene_any"][key]
        report.append(
            f"| {label} | {pct(metric['accuracy'])} | {pct(metric['balanced_accuracy'])} | "
            f"{pct(metric['positive_recall'])} | {pct(metric['specificity'])} | "
            f"{metric['tp']}/{metric['fn']}/{metric['tn']}/{metric['fp']} |"
        )
    report.extend([
        "", "## 대표 양성 instance", "",
        f"- 최저 검출: `{worst_detected_positive['scene_id']}`의 object {worst_detected_positive['object_id']}는 GT overlap ratio {worst_detected_positive['candidate_occlusion_ratio']:.3f}이지만 60회 중 {worst_detected_positive['predicted_occlusion_count']}회만 true였다.",
        f"- 최고 검출: `{best_detected_positive['scene_id']}`의 object {best_detected_positive['object_id']}는 GT overlap ratio {best_detected_positive['candidate_occlusion_ratio']:.3f}이고 60회 중 {best_detected_positive['predicted_occlusion_count']}회 true였다.",
        "",
        f"오답 candidate 행 {len(errors)}개는 `candidate_instance_errors.csv`에 모두 기록했다.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    source_paths = {
        "runs": EXPERIMENT / "logs/runs.jsonl",
        "attempts": EXPERIMENT / "logs/attempts.jsonl",
        "runtime": EXPERIMENT / "logs/runtime.json",
        "pilot_console": EXPERIMENT / "logs/pilot_console.log",
        "full_console": EXPERIMENT / "logs/full_console.log",
        "preparation_console": EXPERIMENT / "logs/preparation_console.log",
        "system_prompt": EXPERIMENT / "prompts/system_en.txt",
        "l2_prompt": EXPERIMENT / "prompts/l2_en.txt",
        "config": EXPERIMENT / "config/experiment_config.json",
        "run_table": EXPERIMENT / "config/run_table.jsonl",
        "preflight": EXPERIMENT / "audit/preflight_report.json",
    }
    for schema_path in sorted((EXPERIMENT / "schemas/l2_instances").glob("*.json")):
        source_paths[f"schema/{schema_path.stem}"] = schema_path
    (OUTPUT / "source_artifact_hashes.json").write_text(
        json.dumps({
            name: {"path": str(path.relative_to(EXPERIMENT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in source_paths.items()
        }, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (OUTPUT / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUTPUT), "integrity": integrity, "overall": overall,
        "comparison": comparison, "candidate_error_count": len(errors),
    }, indent=2))


if __name__ == "__main__":
    main()
