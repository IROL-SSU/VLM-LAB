#!/usr/bin/env python3
"""Analyze the single primary grasp-blocker ID experiment v10."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_l2_grasp_zone_v3 import CONDITIONS, FAMILIES, NAMES, read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10"
V9_EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_l2_instance_occlusion_ltr_v9"
OUTPUT = EXPERIMENT / "results/l2_primary_grasp_blocker_v10"


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def condition_key(row: dict) -> tuple[str, str | None]:
    return row["geometry_condition"], row.get("direction_resolution")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    gt_positive = sum(item["truth_id"] is not None for item in rows)
    gt_null = n - gt_positive
    predicted_positive = sum(item["predicted_id"] is not None for item in rows)
    predicted_null = n - predicted_positive
    tp = sum(
        item["truth_id"] is not None and item["predicted_id"] is not None
        for item in rows
    )
    fn = sum(
        item["truth_id"] is not None and item["predicted_id"] is None
        for item in rows
    )
    tn = sum(
        item["truth_id"] is None and item["predicted_id"] is None
        for item in rows
    )
    fp = sum(
        item["truth_id"] is None and item["predicted_id"] is not None
        for item in rows
    )
    exact = sum(item["exact"] for item in rows)
    positive_id_correct = sum(
        item["truth_id"] is not None and item["exact"] for item in rows
    )
    wrong_non_null_id = sum(
        item["truth_id"] is not None
        and item["predicted_id"] is not None
        and not item["exact"]
        for item in rows
    )
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n": n,
        "gt_positive": gt_positive,
        "gt_null": gt_null,
        "predicted_positive": predicted_positive,
        "predicted_null": predicted_null,
        "exact_matches": exact,
        "exact_accuracy": safe_div(exact, n),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "presence_accuracy": safe_div(tp + tn, n),
        "presence_balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None else None
        ),
        "presence_recall": recall,
        "null_specificity": specificity,
        "presence_precision": safe_div(tp, tp + fp),
        "positive_id_correct": positive_id_correct,
        "positive_id_accuracy": safe_div(positive_id_correct, gt_positive),
        "id_accuracy_given_non_null_prediction_on_positive": safe_div(
            positive_id_correct, tp
        ),
        "wrong_non_null_id_on_positive": wrong_non_null_id,
    }


def collapsed_v9_metrics(truth_by_scene: dict[str, int | None]) -> dict:
    rows = [
        row for row in read_jsonl(V9_EXPERIMENT / "logs/runs.jsonl")
        if row["level"] == "L2"
    ]
    exact = positive_exact = null_exact = 0
    positive = null = 0
    any_tp = any_fn = any_tn = any_fp = 0
    for row in rows:
        predicted = {
            int(item["object_id"])
            for item in row["parsed_response"]["occlusion_assessments_left_to_right"]
            if item["occludes_target"]
        }
        truth_id = truth_by_scene[row["scene_id"]]
        expected = {truth_id} if truth_id is not None else set()
        exact += predicted == expected
        if truth_id is None:
            null += 1
            null_exact += not predicted
        else:
            positive += 1
            positive_exact += predicted == expected
        truth_any = bool(expected)
        predicted_any = bool(predicted)
        any_tp += truth_any and predicted_any
        any_fn += truth_any and not predicted_any
        any_tn += not truth_any and not predicted_any
        any_fp += not truth_any and predicted_any
    recall = safe_div(any_tp, any_tp + any_fn)
    specificity = safe_div(any_tn, any_tn + any_fp)
    return {
        "n": len(rows),
        "single_set_exact_matches": exact,
        "single_set_exact_accuracy": safe_div(exact, len(rows)),
        "positive_single_set_exact": positive_exact,
        "positive_single_set_accuracy": safe_div(positive_exact, positive),
        "null_set_exact": null_exact,
        "null_set_accuracy": safe_div(null_exact, null),
        "presence_tp": any_tp,
        "presence_fn": any_fn,
        "presence_tn": any_tn,
        "presence_fp": any_fp,
        "presence_recall": recall,
        "null_specificity": specificity,
        "presence_balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None else None
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in read_jsonl(EXPERIMENT / "logs/runs.jsonl")
        if row["level"] == "L2"
    ]
    schedule = [
        row for row in read_jsonl(EXPERIMENT / "config/run_table.jsonl")
        if row["level"] == "L2"
    ]
    attempts = read_jsonl(EXPERIMENT / "logs/attempts.jsonl")
    truth_by_scene = {}
    candidate_count_by_scene = {}
    for path in sorted((EXPERIMENT / "geometry/ground_truth").glob("*.json")):
        gt = read_json(path)
        truth_by_scene[gt["scene_id"]] = gt["L2"]["object_to_remove_id"]
        candidate_count_by_scene[gt["scene_id"]] = len(
            gt["L2"]["primary_blocker_candidates"]
        )

    outcome_rows = []
    for row in rows:
        truth_id = truth_by_scene[row["scene_id"]]
        predicted_id = row["parsed_response"]["object_to_remove_id"]
        outcome_rows.append({
            "run_id": row["run_id"],
            "scene_id": row["scene_id"],
            "scene_family": row["scene_family"],
            "condition": row["geometry_condition"],
            "resolution": row.get("direction_resolution"),
            "condition_name": NAMES[condition_key(row)],
            "seed": row["seed"],
            "target_id": row["target_object_id"],
            "truth_id": truth_id,
            "predicted_id": predicted_id,
            "truth_has_blocker": truth_id is not None,
            "predicted_has_blocker": predicted_id is not None,
            "presence_correct": (truth_id is None) == (predicted_id is None),
            "exact": predicted_id == truth_id,
            "numbered_rgb": row["numbered_rgb"],
        })

    failures = []
    run_ids = [row["run_id"] for row in rows]
    scheduled_ids = {row["run_id"] for row in schedule}
    if len(rows) != 1500:
        failures.append(f"completed={len(rows)} expected=1500")
    if len(set(run_ids)) != len(run_ids):
        failures.append("duplicate run IDs")
    if set(run_ids) != scheduled_ids:
        failures.append("completed run IDs differ from L2 schedule")
    cell_counts = Counter((row["scene_id"], condition_key(row)) for row in rows)
    if set(cell_counts.values()) != {5}:
        failures.append("scene-condition cells do not all have five seeds")
    if max(candidate_count_by_scene.values()) > 1:
        failures.append("this scene set unexpectedly contains multiple GT candidates")

    overall = summarize(outcome_rows)
    condition_rows = []
    condition_results = {}
    for condition in CONDITIONS:
        subset = [row for row in outcome_rows if (row["condition"], row["resolution"]) == condition]
        result = summarize(subset)
        condition_results[f"{condition[0]}_{condition[1] or 'NA'}"] = result
        condition_rows.append({
            "condition": condition[0],
            "resolution": condition[1],
            "name": NAMES[condition],
            **result,
        })

    family_rows = []
    family_results = {}
    for family in FAMILIES:
        subset = [row for row in outcome_rows if row["scene_family"] == family]
        result = summarize(subset)
        family_results[family] = result
        family_rows.append({"scene_family": family, **result})

    condition_family_rows = []
    for condition in CONDITIONS:
        for family in FAMILIES:
            subset = [
                row for row in outcome_rows
                if (row["condition"], row["resolution"]) == condition
                and row["scene_family"] == family
            ]
            condition_family_rows.append({
                "condition": condition[0],
                "resolution": condition[1],
                "name": NAMES[condition],
                "scene_family": family,
                **summarize(subset),
            })

    scene_rows = []
    for scene_id in sorted(truth_by_scene):
        subset = [row for row in outcome_rows if row["scene_id"] == scene_id]
        counts = Counter(row["predicted_id"] for row in subset)
        scene_rows.append({
            "scene_id": scene_id,
            "scene_family": subset[0]["scene_family"],
            "target_id": subset[0]["target_id"],
            "truth_id": truth_by_scene[scene_id],
            "prediction_counts": json.dumps(
                {str(key): value for key, value in counts.most_common()}, sort_keys=True
            ),
            **summarize(subset),
        })

    errors = [row for row in outcome_rows if not row["exact"]]
    v9 = collapsed_v9_metrics(truth_by_scene)
    integrity = {
        "status": "pass" if not failures else "fail",
        "scheduled": len(schedule),
        "completed": len(rows),
        "unique_run_ids": len(set(run_ids)),
        "parse_errors": sum(row["parse_error"] is not None for row in rows),
        "post_validation_failures": sum(not row["post_validation_pass"] for row in rows),
        "attempt_records": len(attempts),
        "infrastructure_errors": sum(item["status"] != "success" for item in attempts),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        "failures": failures,
    }
    comparison = {
        "v10_single_id": overall,
        "v9_instance_array_collapsed": v9,
        "exact_accuracy_delta_v10_minus_v9": (
            overall["exact_accuracy"] - v9["single_set_exact_accuracy"]
        ),
        "positive_id_accuracy_delta_v10_minus_v9": (
            overall["positive_id_accuracy"] - v9["positive_single_set_accuracy"]
        ),
        "null_accuracy_delta_v10_minus_v9": (
            overall["null_specificity"] - v9["null_set_accuracy"]
        ),
    }
    results = {
        "prompt_version": read_json(EXPERIMENT / "config/experiment_config.json")["l2_prompt_version"],
        "gt_definition": {
            "positive_scenes": sum(value is not None for value in truth_by_scene.values()),
            "null_scenes": sum(value is None for value in truth_by_scene.values()),
            "candidate_rule": "candidate occlusion ratio >= 0.05",
            "selection_rule": "largest ratio, then smallest display ID",
            "maximum_candidate_count_per_scene": max(candidate_count_by_scene.values()),
            "note": "Isaac Sim instance-mask oracle; not a robot-arm grasp rollout",
        },
        "integrity": integrity,
        "overall": overall,
        "condition_results": condition_results,
        "family_results": family_results,
        "comparison": comparison,
        "error_count": len(errors),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT / "condition_metrics.csv", condition_rows)
    write_csv(OUTPUT / "family_metrics.csv", family_rows)
    write_csv(OUTPUT / "condition_family_metrics.csv", condition_family_rows)
    write_csv(OUTPUT / "scene_metrics.csv", scene_rows)
    write_csv(OUTPUT / "run_outcomes.csv", sorted(outcome_rows, key=lambda row: row["run_id"]))
    write_csv(OUTPUT / "errors.csv", sorted(errors, key=lambda row: row["run_id"]))

    prompt = (EXPERIMENT / "prompts/l2_en.txt").read_text(encoding="utf-8").strip()
    system = (EXPERIMENT / "prompts/system_en.txt").read_text(encoding="utf-8").strip()
    report = [
        "# L2 single primary grasp blocker v10", "", "## 실행 정의", "",
        "System prompt:", "", "```text", system, "```", "",
        "User task prompt:", "", "```text", prompt, "```", "",
        "출력은 `object_to_remove_id` 하나이며 정답 방해물이 없으면 `null`이다.",
        "GT는 Isaac Sim instance mask로 target-only silhouette의 5% 이상을 가리는 물체 중 가림 비율이 가장 큰 ID다. 이 장면 집합에는 양성 장면마다 후보가 정확히 하나 있다.",
        "", "## 무결성", "",
        f"- 완료: {integrity['completed']}/{integrity['scheduled']}",
        f"- 파싱 오류 / 후검증 실패 / 인프라 오류: {integrity['parse_errors']} / {integrity['post_validation_failures']} / {integrity['infrastructure_errors']}",
        "", "## 전체 결과", "",
        "| 지표 | 결과 |", "|---|---:|",
        f"| 정확한 ID 또는 null 완전일치 | {overall['exact_matches']}/{overall['n']} ({pct(overall['exact_accuracy'])}) |",
        f"| 방해물 존재 여부 정확도 | {pct(overall['presence_accuracy'])} |",
        f"| 존재 여부 balanced accuracy | {pct(overall['presence_balanced_accuracy'])} |",
        f"| 방해물 존재 recall | {pct(overall['presence_recall'])} |",
        f"| null specificity | {pct(overall['null_specificity'])} |",
        f"| 양성 장면 정확한 ID | {overall['positive_id_correct']}/{overall['gt_positive']} ({pct(overall['positive_id_accuracy'])}) |",
        f"| 출력 null 횟수 | {overall['predicted_null']}/{overall['n']} |",
        "", "모델은 1,500회 모두 non-null ID를 출력했다. 따라서 양성 존재 recall은 100%지만 null specificity는 0%다.",
        "", "## 조건별", "",
        "| 조건 | 정확한 ID/null | 양성 ID 정확도 | null specificity |",
        "|---|---:|---:|---:|",
    ]
    for item in condition_rows:
        report.append(
            f"| {item['condition']} {item['resolution'] or ''} {item['name']} | "
            f"{pct(item['exact_accuracy'])} | {pct(item['positive_id_accuracy'])} | "
            f"{pct(item['null_specificity'])} |"
        )
    report.extend([
        "", "## 장면 종류별", "",
        "| 장면 종류 | 정확한 ID/null | 양성 ID 정확도 | null specificity |",
        "|---|---:|---:|---:|",
    ])
    for item in family_rows:
        report.append(
            f"| {item['scene_family']} | {pct(item['exact_accuracy'])} | "
            f"{pct(item['positive_id_accuracy'])} | {pct(item['null_specificity'])} |"
        )
    report.extend([
        "", "## v9 대비", "",
        "| 방식 | 전체 exact | 양성 정답 blocker | null 정답 |",
        "|---|---:|---:|---:|",
        f"| v9 instance별 occlusion 배열 | {pct(v9['single_set_exact_accuracy'])} | {pct(v9['positive_single_set_accuracy'])} | {pct(v9['null_set_accuracy'])} |",
        f"| v10 단일 blocker ID | {pct(overall['exact_accuracy'])} | {pct(overall['positive_id_accuracy'])} | {pct(overall['null_specificity'])} |",
        "", "v10은 양성 blocker ID 선택은 크게 좋아졌지만, `null`을 한 번도 선택하지 않아 전체 exact는 v9보다 낮다.",
        "", "## 해석", "",
        "짧은 문구의 `Select the single numbered object`가 선택 행동을 강하게 유도해, 뒤의 `If no object ... return null`보다 우세하게 작동했다. 따라서 이 결과는 target 앞의 물체를 하나 고르는 능력은 높지만, 먼저 제거할 물체가 실제로 필요한지 판단하는 능력은 전혀 분리되지 않았음을 보여준다.",
        "", "주의: 이번 GT는 Isaac Sim의 instance-mask 가림 oracle이다. 실제 로봇 팔의 IK·충돌·gripper 폐쇄를 재생한 grasp rollout 평가는 아니다.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    hash_paths = [
        EXPERIMENT / "config/experiment_config.json",
        EXPERIMENT / "config/run_table.jsonl",
        EXPERIMENT / "prompts/system_en.txt",
        EXPERIMENT / "prompts/l2_en.txt",
        EXPERIMENT / "schemas/l2.json",
        EXPERIMENT / "audit/preflight_report.json",
        EXPERIMENT / "logs/preparation_console.log",
        EXPERIMENT / "logs/pilot_console.log",
        EXPERIMENT / "logs/full_console.log",
        EXPERIMENT / "logs/runs.jsonl",
        EXPERIMENT / "logs/attempts.jsonl",
        EXPERIMENT / "logs/runtime.json",
        EXPERIMENT / "logs/progress.json",
        EXPERIMENT / "logs/last_invocation_summary.json",
        OUTPUT / "results.json",
        OUTPUT / "report.md",
    ]
    (OUTPUT / "artifact_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(EXPERIMENT)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in hash_paths
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
