#!/usr/bin/env python3
"""Audit and summarize the completed VLM action/geometry v1 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import vlm_action_geometry_v1_common as common


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/vlm_action_geometry_single_info_v1"
CONDITION_NAMES = {
    "C0": "Numbered RGB only",
    "C1": "R numeric",
    "C2": "R qualitative",
    "C3": "F numeric",
    "C4": "F qualitative",
    "C5": "M numeric",
    "C6": "M qualitative",
}
SCENE_FAMILIES = (
    "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE", "FC_CLEAR", "FC_BLOCKED",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def latest(rows: list[dict]) -> dict[str, dict]:
    return {row["run_id"]: row for row in rows}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def exact_two_sided_binomial(discordant_first: int, discordant_second: int) -> float:
    total = discordant_first + discordant_second
    if total == 0:
        return 1.0
    extreme = min(discordant_first, discordant_second)
    lower = sum(math.comb(total, index) for index in range(extreme + 1)) / (2**total)
    return min(1.0, 2.0 * lower)


def main() -> None:
    global EXPERIMENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    args = parser.parse_args()
    EXPERIMENT = args.experiment if args.experiment.is_absolute() else ROOT / args.experiment
    config = read_json(EXPERIMENT / "config/experiment_config.json")
    schedule = latest(read_jsonl(EXPERIMENT / "config/run_table.jsonl"))
    inference_raw = read_jsonl(EXPERIMENT / "logs/runs.jsonl")
    replay_raw = read_jsonl(EXPERIMENT / "logs/replay.jsonl")
    inference = latest(inference_raw)
    replay = latest(replay_raw)
    expected = int(config["scheduled_calls"])
    expected_l4 = sum(row["level"] == "L4" for row in schedule.values())
    missing_inference = sorted(set(schedule) - set(inference))
    extra_inference = sorted(set(inference) - set(schedule))
    expected_l4_ids = {run_id for run_id, row in schedule.items() if row["level"] == "L4"}
    missing_replay = sorted(expected_l4_ids - set(replay))
    if missing_inference or extra_inference or len(inference) != expected:
        raise RuntimeError(
            f"Inference audit failed: got={len(inference)} expected={expected} "
            f"missing={len(missing_inference)} extra={len(extra_inference)}"
        )
    if missing_replay or len(replay) != expected_l4:
        raise RuntimeError(
            f"Replay audit failed: got={len(replay)} expected={expected_l4} missing={len(missing_replay)}"
        )

    joined = []
    for run_id, record in inference.items():
        score = record.get("scoring", {}).get("task_correct")
        if record["level"] == "L4":
            score = bool(replay[run_id]["action_valid"])
        joined.append({**record, "final_task_correct": bool(score), "replay": replay.get(run_id)})

    group_fields = ("level", "geometry_condition", "direction_resolution")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in joined:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    metrics = []
    for key, rows in sorted(grouped.items()):
        n = len(rows)
        correct = sum(row["final_task_correct"] for row in rows)
        parsed = sum(not row.get("parse_error") for row in rows)
        semantic = sum(bool(row.get("post_validation_pass")) for row in rows)
        metric = {
            "level": key[0], "geometry_condition": key[1],
            "condition_name": CONDITION_NAMES[key[1]],
            "direction_resolution": key[2], "n": n,
            "parsed_n": parsed, "post_validation_pass_n": semantic,
            "correct_n": correct, "accuracy": correct / n,
        }
        for family in SCENE_FAMILIES:
            family_rows = [row for row in rows if row["scene_family"] == family]
            family_correct = sum(row["final_task_correct"] for row in family_rows)
            prefix = family.lower()
            metric[f"{prefix}_n"] = len(family_rows)
            metric[f"{prefix}_correct_n"] = family_correct
            metric[f"{prefix}_accuracy"] = (
                family_correct / len(family_rows) if family_rows else None
            )
        metrics.append(metric)

    family_metrics = []
    by_family: dict[tuple, list[dict]] = defaultdict(list)
    for row in joined:
        by_family[(row["level"], row["scene_family"])].append(row)
    for (level, family), rows in sorted(by_family.items()):
        correct = sum(row["final_task_correct"] for row in rows)
        family_metrics.append({
            "level": level, "scene_family": family, "n": len(rows),
            "correct_n": correct, "accuracy": correct / len(rows),
        })

    # Paired exact McNemar comparisons against C0 for the same scene and seed.
    lookup = {
        (
            row["level"], row["scene_id"], row["seed"],
            row["geometry_condition"], row["direction_resolution"],
        ): row["final_task_correct"]
        for row in joined
    }
    comparisons = []
    for metric in metrics:
        condition = metric["geometry_condition"]
        if condition == "C0":
            continue
        level = metric["level"]
        resolution = metric["direction_resolution"]
        baseline_resolution = resolution if level == "L4" else None
        pairs = []
        for row in grouped[(level, condition, resolution)]:
            baseline_key = (level, row["scene_id"], row["seed"], "C0", baseline_resolution)
            if baseline_key in lookup:
                pairs.append((lookup[baseline_key], row["final_task_correct"]))
        if not pairs:
            continue
        baseline_only = sum(first and not second for first, second in pairs)
        treatment_only = sum(second and not first for first, second in pairs)
        comparisons.append({
            "level": level, "geometry_condition": condition,
            "condition_name": CONDITION_NAMES[condition],
            "direction_resolution": resolution, "paired_n": len(pairs),
            "baseline_accuracy": sum(first for first, _ in pairs) / len(pairs),
            "condition_accuracy": sum(second for _, second in pairs) / len(pairs),
            "accuracy_delta": (
                sum(second for _, second in pairs) - sum(first for first, _ in pairs)
            ) / len(pairs),
            "c0_only_correct": baseline_only,
            "condition_only_correct": treatment_only,
            "mcnemar_exact_two_sided_p": exact_two_sided_binomial(
                baseline_only, treatment_only
            ),
        })

    artifact_documents = {}
    artifact_hash_mismatches = []
    for row in replay.values():
        relative = row.get("cache_artifact")
        if not relative:
            continue
        path = EXPERIMENT / relative
        actual_hash = common.sha256_path(path)
        if actual_hash != row.get("cache_artifact_sha256"):
            artifact_hash_mismatches.append(row["run_id"])
        artifact_documents[relative] = read_json(path)
    if artifact_hash_mismatches:
        raise RuntimeError(
            f"Replay artifact hash mismatches: {artifact_hash_mismatches[:5]}"
        )

    inference_attempts = read_jsonl(EXPERIMENT / "logs/attempts.jsonl")
    replay_attempts = read_jsonl(EXPERIMENT / "logs/replay_attempts.jsonl")
    validation_error_distribution = Counter(
        message
        for row in inference.values()
        for message in (row.get("post_validation_errors") or [])
    )
    errors = {
        "parse_error_runs": sum(bool(row.get("parse_error")) for row in inference.values()),
        "post_validation_failed_runs": sum(
            not bool(row.get("post_validation_pass")) for row in inference.values()
        ),
        "inference_attempt_errors": sum(row.get("status") == "error" for row in inference_attempts),
        "replay_attempt_errors": sum(row.get("status") == "error" for row in replay_attempts),
        "replay_not_executed_due_to_invalid_model_output": sum(
            not bool(row.get("simulator_executed")) for row in replay.values()
        ),
        "post_validation_error_distribution": dict(
            sorted(validation_error_distribution.items())
        ),
    }
    action_counts = Counter(
        (row.get("model_response") or {}).get("action", "INVALID") for row in replay.values()
    )
    replay_counts = {
        "unique_simulator_artifacts": len({
            row.get("cache_key") for row in replay.values() if row.get("cache_key")
        }),
        "simulator_executed_runs": sum(bool(row.get("simulator_executed")) for row in replay.values()),
        "action_valid_runs": sum(bool(row.get("action_valid")) for row in replay.values()),
        "collision_runs": sum(row.get("collision") is True for row in replay.values()),
        "boundary_violation_runs": sum(
            row.get("boundary_violation") is True for row in replay.values()
        ),
        "action_distribution": dict(sorted(action_counts.items())),
        "all_artifacts_direct_stage_execution": all(
            document.get("direct_stage_execution") is True
            for document in artifact_documents.values()
        ),
        "all_artifact_baselines_match_preparation": all(
            document.get("baseline_target_mask_matches_preparation") is True
            for document in artifact_documents.values()
        ),
        "isaac_sim_versions": sorted({
            document.get("runtime", {}).get("isaac_sim", "unknown")
            for document in artifact_documents.values()
        }),
    }
    latency = [float(row["latency_seconds"]) for row in inference.values()]
    tokens_in = sum(int(row.get("input_token_usage") or 0) for row in inference.values())
    tokens_out = sum(int(row.get("output_token_usage") or 0) for row in inference.values())
    audit = {
        "status": "pass",
        "experiment_version": config["experiment_version"],
        "generated_at": utc_now(),
        "scheduled_runs": len(schedule), "completed_inference_runs": len(inference),
        "raw_inference_log_rows": len(inference_raw),
        "scheduled_l4_runs": expected_l4, "completed_replay_runs": len(replay),
        "raw_append_only_replay_log_rows": len(replay_raw),
        "run_table_sha256": common.sha256_path(EXPERIMENT / "config/run_table.jsonl"),
        "inference_log_sha256": common.sha256_path(EXPERIMENT / "logs/runs.jsonl"),
        "replay_log_sha256": common.sha256_path(EXPERIMENT / "logs/replay.jsonl"),
        "total_input_tokens": tokens_in, "total_output_tokens": tokens_out,
        "latency_seconds": {
            "sum": round(sum(latency), 3), "mean": round(sum(latency) / len(latency), 6),
            "min": round(min(latency), 6), "max": round(max(latency), 6),
        },
        "errors": errors, "replay": replay_counts,
    }
    results = {
        "audit": audit, "condition_metrics": metrics,
        "scene_family_metrics": family_metrics, "paired_comparisons_vs_c0": comparisons,
    }
    result_dir = EXPERIMENT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(result_dir / "results.json", results)
    common.write_json(EXPERIMENT / "audit/final_audit.json", audit)

    with (result_dir / "condition_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (result_dir / "paired_comparisons_vs_c0.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        f"# {config['experiment_version']} — L4 action × geometry results",
        "", f"Generated: {audit['generated_at']}", "",
        "## Completion audit", "",
        f"- Inference: {len(inference):,}/{expected:,}",
        f"- L4 Isaac Sim replay: {len(replay):,}/{expected_l4:,}",
        f"- Unique Isaac Sim artifacts: {replay_counts['unique_simulator_artifacts']:,}",
        f"- Parse failures: {errors['parse_error_runs']:,}",
        f"- Semantic post-validation failures: {errors['post_validation_failed_runs']:,}",
        "", "## Accuracy by condition", "",
        "| Level | Condition | Resolution | n | Valid | Correct | Accuracy | "
        "TRANSLATE | ROTATE | LIFT | FC_CLEAR | FC_BLOCKED |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['level']} | {row['geometry_condition']} {row['condition_name']} | "
            f"{row['direction_resolution']} | {row['n']} | "
            f"{row['post_validation_pass_n']} | {row['correct_n']} | {pct(row['accuracy'])} |"
            f" {pct(row['translate_accuracy'])} | {pct(row['rotate_accuracy'])} |"
            f" {pct(row['lift_and_relocate_accuracy'])} | {pct(row['fc_clear_accuracy'])} |"
            f" {pct(row['fc_blocked_accuracy'])} |"
        )
    lines += ["", "## L4 replay", ""]
    for key, value in replay_counts.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Paired comparisons against C0", "",
              "Exact two-sided McNemar p-values are uncorrected exploratory values. "
              "The five stochastic repeats within a scene are not independent, so these "
              "p-values must not be read as confirmatory scene-level inference.", "",
              "| Level | Condition | Res. | n | C0 | Condition | Delta | p |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(
            f"| {row['level']} | {row['geometry_condition']} {row['condition_name']} | "
            f"{row['direction_resolution']} | {row['paired_n']} | "
            f"{pct(row['baseline_accuracy'])} | {pct(row['condition_accuracy'])} | "
            f"{100.0 * row['accuracy_delta']:+.2f} pp | {row['mcnemar_exact_two_sided_p']:.6g} |"
        )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Hash every immutable/result artifact except this self-referential manifest.
    hashes = {}
    for path in sorted(EXPERIMENT.rglob("*")):
        if path.is_file() and path != EXPERIMENT / "audit/final_artifact_hashes.json":
            hashes[str(path.relative_to(EXPERIMENT))] = {
                "sha256": common.sha256_path(path), "bytes": path.stat().st_size,
            }
    common.write_json(EXPERIMENT / "audit/final_artifact_hashes.json", {
        "generated_at": utc_now(), "file_count": len(hashes), "files": hashes,
    })
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
