#!/usr/bin/env python3
"""Analyze the all-visible-instance L2 interference experiment (v8)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_l2_grasp_zone_v3 import CONDITIONS, FAMILIES, NAMES, pct, read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_instance_interference_v8"
V7_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_potential_interference_v7"
OUTPUT = EXPERIMENT / "results/l2_instance_interference_v8"


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def binary_metric(labels: list[bool], predictions: list[bool]) -> dict:
    if len(labels) != len(predictions):
        raise ValueError("Label/prediction length mismatch")
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n": len(labels),
        "positive": tp + fn,
        "negative": tn + fp,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "accuracy": safe_div(tp + tn, len(labels)),
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "positive_recall": recall,
        "specificity": specificity,
        "precision": safe_div(tp, tp + fp),
    }


def relation_map(items: object) -> dict[int, str]:
    if not isinstance(items, list):
        return {}
    output = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("object_id"), int):
            output[int(item["object_id"])] = item.get("relation_to_target")
    return output


def summarize_runs(rows: list[dict], gt_cache: dict[str, dict]) -> tuple[dict, list[dict], list[dict]]:
    candidate_rows = []
    run_rows = []
    for row in rows:
        gt = gt_cache[row["scene_id"]]
        expected = relation_map(gt["L2"]["instance_assessments"])
        actual = relation_map(row["parsed_response"]["instance_assessments"])
        target_id = int(gt["L2"]["target_id"])
        expected_blockers = sorted(
            object_id for object_id, relation in expected.items()
            if relation == "INTERFERES"
        )
        predicted_blockers = sorted(
            object_id for object_id, relation in actual.items()
            if relation == "INTERFERES"
        )
        correct_candidates = 0
        candidate_count = 0
        for item in gt["L2"]["instance_assessments"]:
            object_id = int(item["object_id"])
            if object_id == target_id:
                continue
            expected_relation = item["relation_to_target"]
            predicted_relation = actual.get(object_id)
            correct = expected_relation == predicted_relation
            candidate_count += 1
            correct_candidates += int(correct)
            candidate_rows.append({
                "run_id": row["run_id"],
                "scene_id": row["scene_id"],
                "scene_family": row["scene_family"],
                "condition": row["geometry_condition"],
                "resolution": row.get("direction_resolution"),
                "seed": row["seed"],
                "target_id": target_id,
                "object_id": object_id,
                "candidate_occlusion_pixels": item["candidate_occlusion_pixels"],
                "candidate_occlusion_ratio": item["candidate_occlusion_ratio"],
                "truth_relation": expected_relation,
                "predicted_relation": predicted_relation,
                "truth_interferes": expected_relation == "INTERFERES",
                "predicted_interferes": predicted_relation == "INTERFERES",
                "correct": correct,
            })
        exact = actual == expected and row["parsed_response"].get("target_id") == target_id
        blocker_exact = predicted_blockers == expected_blockers
        run_rows.append({
            "run_id": row["run_id"],
            "scene_id": row["scene_id"],
            "scene_family": row["scene_family"],
            "condition": row["geometry_condition"],
            "resolution": row.get("direction_resolution"),
            "seed": row["seed"],
            "target_id": target_id,
            "numbered_object_count": len(expected),
            "candidate_count": candidate_count,
            "correct_candidate_count": correct_candidates,
            "target_role_correct": actual.get(target_id) == "TARGET",
            "expected_blocker_ids": json.dumps(expected_blockers),
            "predicted_blocker_ids": json.dumps(predicted_blockers),
            "instance_exact_match": exact,
            "blocker_set_exact_match": blocker_exact,
            "truth_any_interference": bool(expected_blockers),
            "predicted_any_interference": bool(predicted_blockers),
            "any_interference_correct": bool(expected_blockers) == bool(predicted_blockers),
        })

    candidate_metric = binary_metric(
        [bool(item["truth_interferes"]) for item in candidate_rows],
        [bool(item["predicted_interferes"]) for item in candidate_rows],
    )
    any_metric = binary_metric(
        [bool(item["truth_any_interference"]) for item in run_rows],
        [bool(item["predicted_any_interference"]) for item in run_rows],
    )
    summary = {
        "candidate_instance": candidate_metric,
        "run_count": len(run_rows),
        "instance_exact_matches": sum(item["instance_exact_match"] for item in run_rows),
        "instance_exact_match_accuracy": safe_div(
            sum(item["instance_exact_match"] for item in run_rows), len(run_rows)
        ),
        "blocker_set_exact_matches": sum(item["blocker_set_exact_match"] for item in run_rows),
        "blocker_set_exact_match_accuracy": safe_div(
            sum(item["blocker_set_exact_match"] for item in run_rows), len(run_rows)
        ),
        "target_role_correct": sum(item["target_role_correct"] for item in run_rows),
        "target_role_accuracy": safe_div(
            sum(item["target_role_correct"] for item in run_rows), len(run_rows)
        ),
        "scene_any_interference": any_metric,
    }
    return summary, candidate_rows, run_rows


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        if row["scene_id"] not in gt_cache:
            gt_cache[row["scene_id"]] = read_json(EXPERIMENT / row["ground_truth"])
            mapping_cache[row["scene_id"]] = read_json(EXPERIMENT / row["id_mapping"])

    integrity_failures = []
    run_ids = [row["run_id"] for row in rows]
    scheduled_ids = {row["run_id"] for row in schedule}
    if len(rows) != 1500:
        integrity_failures.append(f"completed_rows={len(rows)} expected=1500")
    if len(set(run_ids)) != len(run_ids):
        integrity_failures.append("duplicate run IDs")
    if set(run_ids) != scheduled_ids:
        integrity_failures.append("completed run IDs do not equal scheduled L2 run IDs")

    output_invariant_failures = []
    for row in rows:
        mapping = mapping_cache[row["scene_id"]]
        visible_ids = sorted(
            int(item["display_id"]) for item in mapping["instances"] if item["visible"]
        )
        parsed = row["parsed_response"]
        items = parsed.get("instance_assessments", [])
        output_ids = [item.get("object_id") for item in items if isinstance(item, dict)]
        if (
            output_ids != visible_ids
            or len(items) != len(visible_ids)
            or parsed.get("target_id") != row["target_object_id"]
        ):
            output_invariant_failures.append(row["run_id"])
    if output_invariant_failures:
        integrity_failures.append(
            f"output invariant failures={len(output_invariant_failures)}"
        )

    expected_counts = Counter((row["scene_id"], condition_key(row)) for row in rows)
    if set(expected_counts.values()) != {5}:
        integrity_failures.append("scene-condition cells do not all contain five seeds")
    if Counter(row["seed"] for row in rows) != Counter({seed: 300 for seed in range(28101, 28106)}):
        integrity_failures.append("seed balance mismatch")

    overall, candidate_rows, run_rows = summarize_runs(rows, gt_cache)
    row_by_id = {row["run_id"]: row for row in rows}
    run_outcome_by_id = {row["run_id"]: row for row in run_rows}

    condition_results = {}
    condition_csv = []
    for condition in CONDITIONS:
        subset = [row_by_id[item["run_id"]] for item in run_rows if (
            item["condition"], item["resolution"] or None
        ) == condition]
        result, _, _ = summarize_runs(subset, gt_cache)
        key = f"{condition[0]}_{condition[1] or 'NA'}"
        condition_results[key] = result
        metric = result["candidate_instance"]
        condition_csv.append({
            "condition": condition[0],
            "resolution": condition[1],
            "name": NAMES[condition],
            "run_count": result["run_count"],
            "candidate_count": metric["n"],
            "tp": metric["tp"], "fn": metric["fn"],
            "tn": metric["tn"], "fp": metric["fp"],
            "candidate_accuracy": metric["accuracy"],
            "candidate_balanced_accuracy": metric["balanced_accuracy"],
            "interference_recall": metric["positive_recall"],
            "non_interference_specificity": metric["specificity"],
            "precision": metric["precision"],
            "instance_exact_match_accuracy": result["instance_exact_match_accuracy"],
            "blocker_set_exact_match_accuracy": result["blocker_set_exact_match_accuracy"],
            "any_interference_accuracy": result["scene_any_interference"]["accuracy"],
            "any_interference_balanced_accuracy": result["scene_any_interference"]["balanced_accuracy"],
        })

    family_results = {}
    family_csv = []
    for family in FAMILIES:
        subset = [row for row in rows if row["scene_family"] == family]
        result, _, _ = summarize_runs(subset, gt_cache)
        family_results[family] = result
        metric = result["candidate_instance"]
        family_csv.append({
            "scene_family": family,
            "run_count": result["run_count"],
            "candidate_count": metric["n"],
            "tp": metric["tp"], "fn": metric["fn"],
            "tn": metric["tn"], "fp": metric["fp"],
            "candidate_accuracy": metric["accuracy"],
            "candidate_balanced_accuracy": metric["balanced_accuracy"],
            "interference_recall": metric["positive_recall"],
            "non_interference_specificity": metric["specificity"],
            "instance_exact_match_accuracy": result["instance_exact_match_accuracy"],
            "any_interference_accuracy": result["scene_any_interference"]["accuracy"],
        })

    condition_family_csv = []
    for condition in CONDITIONS:
        for family in FAMILIES:
            subset = [
                row for row in rows
                if condition_key(row) == condition and row["scene_family"] == family
            ]
            result, _, _ = summarize_runs(subset, gt_cache)
            metric = result["candidate_instance"]
            condition_family_csv.append({
                "condition": condition[0],
                "resolution": condition[1],
                "name": NAMES[condition],
                "scene_family": family,
                "run_count": result["run_count"],
                "candidate_count": metric["n"],
                "candidate_accuracy": metric["accuracy"],
                "candidate_balanced_accuracy": metric["balanced_accuracy"],
                "interference_recall": metric["positive_recall"],
                "non_interference_specificity": metric["specificity"],
                "instance_exact_match_accuracy": result["instance_exact_match_accuracy"],
                "any_interference_accuracy": result["scene_any_interference"]["accuracy"],
            })

    scene_csv = []
    for scene_id in sorted(gt_cache):
        scene_runs = [item for item in run_rows if item["scene_id"] == scene_id]
        scene_candidates = [item for item in candidate_rows if item["scene_id"] == scene_id]
        candidate_metric = binary_metric(
            [bool(item["truth_interferes"]) for item in scene_candidates],
            [bool(item["predicted_interferes"]) for item in scene_candidates],
        )
        scene_csv.append({
            "scene_id": scene_id,
            "scene_family": scene_runs[0]["scene_family"],
            "target_id": scene_runs[0]["target_id"],
            "run_count": len(scene_runs),
            "candidate_count": candidate_metric["n"],
            "gt_blocker_ids": scene_runs[0]["expected_blocker_ids"],
            "candidate_accuracy": candidate_metric["accuracy"],
            "interference_recall": candidate_metric["positive_recall"],
            "non_interference_specificity": candidate_metric["specificity"],
            "instance_exact_match_accuracy": safe_div(
                sum(item["instance_exact_match"] for item in scene_runs), len(scene_runs)
            ),
            "any_interference_accuracy": safe_div(
                sum(item["any_interference_correct"] for item in scene_runs), len(scene_runs)
            ),
        })

    candidate_object_csv = []
    grouped_candidates = defaultdict(list)
    for item in candidate_rows:
        grouped_candidates[(item["scene_id"], item["object_id"])].append(item)
    for (scene_id, object_id), items in sorted(grouped_candidates.items()):
        truth_interferes = bool(items[0]["truth_interferes"])
        predicted_count = sum(item["predicted_interferes"] for item in items)
        candidate_object_csv.append({
            "scene_id": scene_id,
            "scene_family": items[0]["scene_family"],
            "target_id": items[0]["target_id"],
            "object_id": object_id,
            "candidate_occlusion_pixels": items[0]["candidate_occlusion_pixels"],
            "candidate_occlusion_ratio": items[0]["candidate_occlusion_ratio"],
            "truth_relation": items[0]["truth_relation"],
            "run_count": len(items),
            "predicted_interference_count": predicted_count,
            "predicted_interference_rate": predicted_count / len(items),
            "accuracy": (
                predicted_count / len(items)
                if truth_interferes
                else (len(items) - predicted_count) / len(items)
            ),
            "numbered_rgb": f"images/numbered_rgb/{scene_id}.png",
        })
    worst_false_negative = min(
        (item for item in candidate_object_csv if item["truth_relation"] == "INTERFERES"),
        key=lambda item: item["predicted_interference_rate"],
    )
    worst_false_positive = max(
        (
            item for item in candidate_object_csv
            if item["truth_relation"] == "DOES_NOT_INTERFERE"
        ),
        key=lambda item: item["predicted_interference_rate"],
    )

    v7_rows = read_jsonl(V7_EXPERIMENT / "logs/l2_potential_interference_v7/runs.jsonl")
    v7_by_id = {row["run_id"]: row for row in v7_rows}
    v8_any_labels = [bool(run_outcome_by_id[run_id]["truth_any_interference"]) for run_id in run_ids]
    v8_any_predictions = [
        bool(run_outcome_by_id[run_id]["predicted_any_interference"]) for run_id in run_ids
    ]
    v7_predictions = [
        bool(v7_by_id[run_id]["parsed_response"]["potential_grasp_interference"])
        for run_id in run_ids
    ]
    comparison = {
        "v8_any_instance_interferes": binary_metric(v8_any_labels, v8_any_predictions),
        "v7_single_boolean": binary_metric(v8_any_labels, v7_predictions),
        "same_numbered_rgb_hash": sum(
            row_by_id[run_id]["numbered_rgb_sha256"] == v7_by_id[run_id]["numbered_rgb_sha256"]
            for run_id in run_ids
        ),
        "same_geometry_hash": sum(
            row_by_id[run_id]["geometry_json_sha256"] == v7_by_id[run_id]["geometry_json_sha256"]
            for run_id in run_ids
        ),
    }

    error_rows = [item for item in candidate_rows if not item["correct"]]
    integrity = {
        "status": "pass" if not integrity_failures else "fail",
        "scheduled": len(schedule),
        "completed": len(rows),
        "unique_run_ids": len(set(run_ids)),
        "parse_errors": sum(row["parse_error"] is not None for row in rows),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in rows),
        "output_invariant_failures": len(output_invariant_failures),
        "attempt_records": len(attempts),
        "infrastructure_errors": sum(item["status"] != "success" for item in attempts),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "failures": integrity_failures,
    }
    results = {
        "prompt_version": read_json(EXPERIMENT / "config/experiment_config.json")["l2_prompt_version"],
        "gt_definition": {
            "scope": "every numbered visible instance",
            "candidate_rule": "candidate_occlusion_ratio >= 0.05",
            "numbered_instances_per_scene": sum(
                len(gt["L2"]["instance_assessments"]) for gt in gt_cache.values()
            ),
            "candidate_instances_per_scene_set": sum(
                len(gt["L2"]["instance_assessments"]) - 1 for gt in gt_cache.values()
            ),
            "positive_candidate_instances_per_scene_set": sum(
                item["relation_to_target"] == "INTERFERES"
                for gt in gt_cache.values()
                for item in gt["L2"]["instance_assessments"]
            ),
        },
        "integrity": integrity,
        "overall": overall,
        "condition_results": condition_results,
        "family_results": family_results,
        "v7_scene_boolean_comparison": comparison,
        "error_candidate_count": len(error_rows),
        "representative_errors": {
            "worst_false_negative": worst_false_negative,
            "worst_false_positive": worst_false_positive,
        },
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT / "condition_metrics.csv", condition_csv)
    write_csv(OUTPUT / "family_metrics.csv", family_csv)
    write_csv(OUTPUT / "condition_family_metrics.csv", condition_family_csv)
    write_csv(OUTPUT / "scene_metrics.csv", scene_csv)
    write_csv(OUTPUT / "candidate_object_metrics.csv", candidate_object_csv)
    write_csv(OUTPUT / "run_outcomes.csv", sorted(run_rows, key=lambda item: item["run_id"]))
    write_csv(
        OUTPUT / "candidate_instance_outcomes.csv",
        sorted(candidate_rows, key=lambda item: (item["run_id"], item["object_id"])),
    )
    if error_rows:
        write_csv(
            OUTPUT / "candidate_instance_errors.csv",
            sorted(error_rows, key=lambda item: (item["run_id"], item["object_id"])),
        )
    else:
        (OUTPUT / "candidate_instance_errors.csv").write_text(
            "run_id,scene_id,object_id,truth_relation,predicted_relation\n",
            encoding="utf-8",
        )

    overall_candidate = overall["candidate_instance"]
    overall_any = overall["scene_any_interference"]
    report = [
        "# L2 all-instance interference v8", "", "## 실험 정의", "",
        "한 장면을 한 번 호출하고, 보이는 모든 번호 물체를 오름차순으로 한 번씩 출력한다. Target은 `TARGET`, 나머지는 `INTERFERES` 또는 `DOES_NOT_INTERFERE`로 분류한다.",
        "", "프롬프트에는 수치 임계값을 노출하지 않았다. GT는 각 후보가 target-only silhouette의 5% 이상을 가린 경우 `INTERFERES`로 정의한다.",
        "", "## 무결성", "",
        f"- 완료: {integrity['completed']}/{integrity['scheduled']}",
        f"- 파싱 오류 / 후검증 실패 / 출력 불변식 실패: {integrity['parse_errors']} / {integrity['post_validation_failures']} / {integrity['output_invariant_failures']}",
        f"- 인프라 오류: {integrity['infrastructure_errors']}",
        "", "## 전체 결과", "",
        "| 평가 단위 | Accuracy | Balanced accuracy | Interference recall | Non-interference specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 후보 instance | {pct(overall_candidate['accuracy'])} | {pct(overall_candidate['balanced_accuracy'])} | {pct(overall_candidate['positive_recall'])} | {pct(overall_candidate['specificity'])} | {overall_candidate['tp']}/{overall_candidate['fn']}/{overall_candidate['tn']}/{overall_candidate['fp']} |",
        f"| 장면 내 하나라도 방해 | {pct(overall_any['accuracy'])} | {pct(overall_any['balanced_accuracy'])} | {pct(overall_any['positive_recall'])} | {pct(overall_any['specificity'])} | {overall_any['tp']}/{overall_any['fn']}/{overall_any['tn']}/{overall_any['fp']} |",
        "",
        f"- 모든 번호 관계 완전일치: {overall['instance_exact_matches']}/{overall['run_count']} ({pct(overall['instance_exact_match_accuracy'])})",
        f"- 방해 물체 ID 집합 완전일치: {overall['blocker_set_exact_matches']}/{overall['run_count']} ({pct(overall['blocker_set_exact_match_accuracy'])})",
        f"- Target 역할 정확도: {overall['target_role_correct']}/{overall['run_count']} ({pct(overall['target_role_accuracy'])})",
        "", "## 조건별 결과", "",
        "| 조건 | 후보 정확도 | BA | 방해 recall | 비방해 specificity | 전체 관계 완전일치 | 장면 Boolean 정확도 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in condition_csv:
        report.append(
            f"| {item['condition']} {item['resolution'] or ''} {item['name']} | "
            f"{pct(item['candidate_accuracy'])} | {pct(item['candidate_balanced_accuracy'])} | "
            f"{pct(item['interference_recall'])} | {pct(item['non_interference_specificity'])} | "
            f"{pct(item['instance_exact_match_accuracy'])} | {pct(item['any_interference_accuracy'])} |"
        )
    report.extend([
        "", "## 장면 종류별 결과", "",
        "| 장면 종류 | 후보 정확도 | BA | 방해 recall | 비방해 specificity | 전체 관계 완전일치 | 장면 Boolean 정확도 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in family_csv:
        report.append(
            f"| {item['scene_family']} | {pct(item['candidate_accuracy'])} | "
            f"{pct(item['candidate_balanced_accuracy']) if item['candidate_balanced_accuracy'] is not None else 'N/A'} | "
            f"{pct(item['interference_recall']) if item['interference_recall'] is not None else 'N/A'} | "
            f"{pct(item['non_interference_specificity'])} | "
            f"{pct(item['instance_exact_match_accuracy'])} | {pct(item['any_interference_accuracy'])} |"
        )
    v8_compare = comparison["v8_any_instance_interferes"]
    v7_compare = comparison["v7_single_boolean"]
    report.extend([
        "", "## v7 단일 Boolean과 비교", "",
        "| 방식 | Accuracy | BA | Recall | Specificity | TP/FN/TN/FP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| v8 instance별 출력의 OR | {pct(v8_compare['accuracy'])} | {pct(v8_compare['balanced_accuracy'])} | {pct(v8_compare['positive_recall'])} | {pct(v8_compare['specificity'])} | {v8_compare['tp']}/{v8_compare['fn']}/{v8_compare['tn']}/{v8_compare['fp']} |",
        f"| v7 단일 Boolean | {pct(v7_compare['accuracy'])} | {pct(v7_compare['balanced_accuracy'])} | {pct(v7_compare['positive_recall'])} | {pct(v7_compare['specificity'])} | {v7_compare['tp']}/{v7_compare['fn']}/{v7_compare['tn']}/{v7_compare['fp']} |",
        "",
        f"오답 후보 instance 행은 총 {len(error_rows)}개이며 `candidate_instance_errors.csv`에 모두 기록했다.",
        "", "## 대표 오류", "",
        f"- False negative: `{worst_false_negative['scene_id']}`에서 target {worst_false_negative['target_id']}를 가리는 object {worst_false_negative['object_id']}의 GT overlap ratio는 {worst_false_negative['candidate_occlusion_ratio']:.3f}이지만, 60회 중 {worst_false_negative['predicted_interference_count']}회만 `INTERFERES`로 출력했다.",
        f"- False positive: `{worst_false_positive['scene_id']}`에서 object {worst_false_positive['object_id']}의 GT overlap ratio는 {worst_false_positive['candidate_occlusion_ratio']:.3f}이지만, 60회 중 {worst_false_positive['predicted_interference_count']}회 `INTERFERES`로 출력했다. 이 물체는 target {worst_false_positive['target_id']} 바로 옆에 있어, 프롬프트의 gripper-contact 표현이 시각적 가림 GT보다 넓게 해석된 사례다.",
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
            name: {
                "path": str(path.relative_to(EXPERIMENT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in source_paths.items()
        }, indent=2) + "\n",
        encoding="utf-8",
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
        "output": str(OUTPUT),
        "integrity": integrity,
        "overall": overall,
        "comparison": comparison,
        "error_candidate_count": len(error_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
