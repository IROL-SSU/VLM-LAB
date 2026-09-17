#!/usr/bin/env python3
"""Analyze the 60-scene B0-N0 sparse-ID TEXT-vs-forced-JSON experiment."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/ssu/ShelfScene")
EXPERIMENT = ROOT / "experiments/b0_n0_sparse_ids_text_vs_json_60scenes_20260826"
MODES = ("TEXT", "JSON_FORCED")
TEXT_PATTERN = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_response(mode: str, raw: str) -> tuple[bool, list[int] | None, str | None]:
    if mode == "TEXT":
        if not TEXT_PATTERN.fullmatch(raw):
            return False, None, "text_format_invalid"
        return True, [int(value.strip()) for value in raw.split(",")], None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return False, None, "json_parse_error"
    if not isinstance(value, dict) or set(value) != {"visible_ids"}:
        return False, None, "json_shape_invalid"
    ids = value["visible_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in ids):
        return False, None, "json_items_invalid"
    return True, ids, None


def exact_binomial_two_sided(a: int, b: int) -> float:
    total = a + b
    if total == 0:
        return 1.0
    tail = min(a, b)
    probability = sum(math.comb(total, value) for value in range(tail + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def pct(numerator: int | float, denominator: int | float) -> float | None:
    return round(100.0 * numerator / denominator, 3) if denominator else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = len(rows)
    parsed = sum(row["format_valid"] for row in rows)
    exact = sum(row["id_set_exact"] for row in rows)
    tp = sum(row["id_true_positive_count"] for row in rows)
    fp = sum(row["id_false_positive_count"] for row in rows)
    fn = sum(row["id_false_negative_count"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "calls": calls,
        "format_valid_count": parsed,
        "format_valid_pct": pct(parsed, calls),
        "id_set_exact_count": exact,
        "id_set_exact_pct": pct(exact, calls),
        "id_micro_precision_pct": round(100 * precision, 3),
        "id_micro_recall_pct": round(100 * recall, 3),
        "id_micro_f1_pct": round(100 * f1, 3),
        "missing_id_count": fn,
        "extra_id_count": fp,
        "duplicate_output_calls": sum(row["has_duplicate_prediction"] for row in rows),
        "left_to_right_exact_diagnostic_pct": pct(sum(row["left_to_right_exact_diagnostic"] for row in rows), calls),
        "numerically_sorted_output_pct": pct(sum(row["numerically_sorted_output"] for row in rows), calls),
        "parse_failures": calls - parsed,
        "length_finishes": sum(row["finish_reason"] == "length" for row in rows),
        "mean_output_tokens": round(sum(row["output_tokens"] for row in rows) / calls, 3),
        "mean_wall_seconds": round(sum(row["wall_seconds"] for row in rows) / calls, 4),
    }


def main() -> None:
    config = read_json(EXPERIMENT / "config/experiment_config.json")
    frozen = read_json(EXPERIMENT / "config/frozen_scenes.json")
    frozen_by_scene = {row["scene_id"]: row for row in frozen}
    raw_rows = load_jsonl(EXPERIMENT / "logs/runs.jsonl")
    expected = int(config["scheduled_main_calls"])
    if len(raw_rows) != expected:
        raise ValueError(f"expected {expected} records, found {len(raw_rows)}")
    run_ids = [row["run_id"] for row in raw_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run IDs")

    evaluated: list[dict[str, Any]] = []
    for row in raw_rows:
        valid, prediction, parse_error = parse_response(row["mode"], row["raw_response"])
        truth = [int(value) for value in row["ground_truth"]["visible_ids_sorted"]]
        left_to_right = [int(value) for value in row["ground_truth"]["ids_left_to_right"]]
        predicted = prediction or []
        gt_counter = Counter(truth)
        pred_counter = Counter(predicted)
        tp = sum((gt_counter & pred_counter).values())
        fp = sum((pred_counter - gt_counter).values())
        fn = sum((gt_counter - pred_counter).values())
        missing = sorted((gt_counter - pred_counter).elements())
        extra = sorted((pred_counter - gt_counter).elements())
        duplicates = sorted(value for value, count in pred_counter.items() if count > 1)
        set_exact = valid and pred_counter == gt_counter
        if not valid:
            error_type = "parse_failure"
        elif set_exact:
            error_type = "correct"
        elif duplicates:
            error_type = "duplicate_output"
        elif missing and extra:
            error_type = "missing_and_extra"
        elif missing:
            error_type = "missing_only"
        elif extra:
            error_type = "extra_only"
        else:
            error_type = "other_content_error"
        evaluated.append(
            {
                **row,
                "format_valid": valid,
                "parse_error": parse_error,
                "prediction_ids": prediction,
                "ground_truth_ids_sorted": truth,
                "ground_truth_ids_left_to_right": left_to_right,
                "id_set_exact": set_exact,
                "id_true_positive_count": tp,
                "id_false_positive_count": fp,
                "id_false_negative_count": fn,
                "missing_ids": missing,
                "extra_ids": extra,
                "duplicate_ids": duplicates,
                "has_duplicate_prediction": bool(duplicates),
                "left_to_right_exact_diagnostic": valid and predicted == left_to_right,
                "numerically_sorted_output": valid and predicted == sorted(predicted),
                "error_type": error_type,
            }
        )

    (EXPERIMENT / "results/evaluated_records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in evaluated),
        encoding="utf-8",
    )

    mode_summary = [
        {"mode": mode, **summarize([row for row in evaluated if row["mode"] == mode])}
        for mode in MODES
    ]
    write_csv(EXPERIMENT / "results/mode_summary.csv", mode_summary)

    level_summary: list[dict[str, Any]] = []
    for scope in ("Overall", "L3", "L4", "L5"):
        scoped = evaluated if scope == "Overall" else [row for row in evaluated if row["level"] == scope]
        for mode in MODES:
            level_summary.append({"scope": scope, "mode": mode, **summarize([row for row in scoped if row["mode"] == mode])})
    write_csv(EXPERIMENT / "results/level_mode_summary.csv", level_summary)

    factor_summary: list[dict[str, Any]] = []
    for level in ("L3", "L4", "L5"):
        factor = "object_count" if level == "L3" else "occlusion_band"
        values = sorted({str(row["factors"][factor]) for row in evaluated if row["level"] == level})
        for value in values:
            for mode in MODES:
                rows = [
                    row for row in evaluated
                    if row["level"] == level and row["mode"] == mode and str(row["factors"][factor]) == value
                ]
                factor_summary.append({"level": level, "factor": factor, "value": value, "mode": mode, **summarize(rows)})
    write_csv(EXPERIMENT / "results/factor_summary.csv", factor_summary)

    scene_summary: list[dict[str, Any]] = []
    for scene in frozen:
        for mode in MODES:
            rows = [row for row in evaluated if row["scene_id"] == scene["scene_id"] and row["mode"] == mode]
            scene_summary.append(
                {
                    "scene_id": scene["scene_id"],
                    "level": scene["level"],
                    "mode": mode,
                    "id_set_exact_out_of_5": sum(row["id_set_exact"] for row in rows),
                    "parse_success_out_of_5": sum(row["format_valid"] for row in rows),
                    "missing_ids_total": sum(row["id_false_negative_count"] for row in rows),
                    "extra_ids_total": sum(row["id_false_positive_count"] for row in rows),
                    "ground_truth_ids": json.dumps(scene["ground_truth"]["visible_ids_sorted"]),
                    "image": scene["image"],
                }
            )
    write_csv(EXPERIMENT / "results/scene_summary.csv", scene_summary)

    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in evaluated:
        paired[(row["scene_id"], int(row["seed"]))][row["mode"]] = row
    both = text_only = json_only = neither = same_set = 0
    for key, pair in paired.items():
        if set(pair) != set(MODES):
            raise ValueError(f"incomplete pair: {key}")
        text_ok = pair["TEXT"]["id_set_exact"]
        json_ok = pair["JSON_FORCED"]["id_set_exact"]
        if text_ok and json_ok:
            both += 1
        elif text_ok:
            text_only += 1
        elif json_ok:
            json_only += 1
        else:
            neither += 1
        text_prediction = pair["TEXT"]["prediction_ids"] or []
        json_prediction = pair["JSON_FORCED"]["prediction_ids"] or []
        same_set += Counter(text_prediction) == Counter(json_prediction)
    paired_summary = {
        "paired_scene_seed_trials": len(paired),
        "both_correct": both,
        "text_only_correct": text_only,
        "json_only_correct": json_only,
        "neither_correct": neither,
        "same_predicted_id_multiset_count": same_set,
        "same_predicted_id_multiset_pct": pct(same_set, len(paired)),
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(text_only, json_only),
    }
    write_json(EXPERIMENT / "results/paired_comparison.json", paired_summary)

    error_breakdown = {
        mode: dict(Counter(row["error_type"] for row in evaluated if row["mode"] == mode))
        for mode in MODES
    }
    single_substitutions = Counter()
    for row in evaluated:
        if len(row["missing_ids"]) == 1 and len(row["extra_ids"]) == 1:
            single_substitutions[(row["mode"], row["missing_ids"][0], row["extra_ids"][0])] += 1
    substitution_rows = [
        {"mode": mode, "ground_truth_id": truth, "predicted_id": predicted, "count": count}
        for (mode, truth, predicted), count in single_substitutions.most_common()
    ]
    write_csv(EXPERIMENT / "results/single_id_substitutions.csv", substitution_rows)

    audit = {
        "expected_records": expected,
        "observed_records": len(evaluated),
        "unique_run_ids": len(set(run_ids)),
        "batch_size_values": sorted({row["runtime"]["batch_size"] for row in evaluated}),
        "finish_reason_counts": dict(Counter(row["finish_reason"] for row in evaluated)),
        "parse_failures": sum(not row["format_valid"] for row in evaluated),
        "json_schema_failures": sum(row["mode"] == "JSON_FORCED" and not row["format_valid"] for row in evaluated),
        "length_finishes": sum(row["finish_reason"] == "length" for row in evaluated),
        "error_breakdown": error_breakdown,
        "paired": paired_summary,
    }
    write_json(EXPERIMENT / "audit/final_audit.json", audit)

    mode_lookup = {row["mode"]: row for row in mode_summary}
    level_lookup = {(row["scope"], row["mode"]): row for row in level_summary}
    failing_scene_rows = sorted(
        (row for row in scene_summary if row["id_set_exact_out_of_5"] < 5),
        key=lambda row: (row["id_set_exact_out_of_5"], row["level"], row["scene_id"], row["mode"]),
    )
    failing_lines = "\n".join(
        f"| {row['level']} | `{row['scene_id']}` | {row['mode']} | {row['id_set_exact_out_of_5']}/5 | {row['missing_ids_total']} | {row['extra_ids_total']} |"
        for row in failing_scene_rows
    ) or "| — | No failing scenes | — | — | — | — |"

    report = f"""# B0-N0 sparse-ID recognition: TEXT vs forced JSON

## Protocol

- Model: `{config['model_id']}`
- Scenes: 60 (L3 20, L4 20, L5 20)
- Repeats: 5 paired seeds per scene and output mode
- Calls: 600, batch size 1
- IDs: scene-specific unique two-digit sparse IDs; matched occlusion scenes share the same IDs
- Primary metric: order-independent exact ID multiset match
- Generation: temperature 0.7, top-p 0.9, max tokens 1024, max model length 8192
- JSON: strict structured output with an unbounded integer array

## Main results

| Mode | ID-set exact | Micro precision | Micro recall | Parse success |
|---|---:|---:|---:|---:|
| TEXT | {mode_lookup['TEXT']['id_set_exact_pct']:.3f}% ({mode_lookup['TEXT']['id_set_exact_count']}/300) | {mode_lookup['TEXT']['id_micro_precision_pct']:.3f}% | {mode_lookup['TEXT']['id_micro_recall_pct']:.3f}% | {mode_lookup['TEXT']['format_valid_pct']:.3f}% |
| JSON forced | {mode_lookup['JSON_FORCED']['id_set_exact_pct']:.3f}% ({mode_lookup['JSON_FORCED']['id_set_exact_count']}/300) | {mode_lookup['JSON_FORCED']['id_micro_precision_pct']:.3f}% | {mode_lookup['JSON_FORCED']['id_micro_recall_pct']:.3f}% | {mode_lookup['JSON_FORCED']['format_valid_pct']:.3f}% |

## ID-set exact by level

| Level | TEXT | JSON forced |
|---|---:|---:|
| L3 | {level_lookup[('L3','TEXT')]['id_set_exact_pct']:.3f}% | {level_lookup[('L3','JSON_FORCED')]['id_set_exact_pct']:.3f}% |
| L4 | {level_lookup[('L4','TEXT')]['id_set_exact_pct']:.3f}% | {level_lookup[('L4','JSON_FORCED')]['id_set_exact_pct']:.3f}% |
| L5 | {level_lookup[('L5','TEXT')]['id_set_exact_pct']:.3f}% | {level_lookup[('L5','JSON_FORCED')]['id_set_exact_pct']:.3f}% |

## Paired comparison

- Both correct: {both}/300
- TEXT only correct: {text_only}/300
- JSON only correct: {json_only}/300
- Neither correct: {neither}/300
- Same predicted ID multiset: {paired_summary['same_predicted_id_multiset_pct']:.3f}%
- Exact McNemar p: {paired_summary['mcnemar_exact_two_sided_p']:.6g}

## Error breakdown

- TEXT: {json.dumps(error_breakdown['TEXT'], ensure_ascii=False)}
- JSON forced: {json.dumps(error_breakdown['JSON_FORCED'], ensure_ascii=False)}
- Parse failures: {audit['parse_failures']}; length finishes: {audit['length_finishes']}

## Failing scene-mode groups

| Level | Scene | Mode | Exact | Missing IDs | Extra IDs |
|---|---|---|---:|---:|---:|
{failing_lines}

## Actual user prompts

Common:

```text
{(EXPERIMENT / 'prompts/common_prompt_en.txt').read_text(encoding='utf-8').strip()}
```

TEXT suffix:

```text
{(EXPERIMENT / 'prompts/text_suffix_en.txt').read_text(encoding='utf-8').strip()}
```

JSON suffix:

```text
{(EXPERIMENT / 'prompts/json_suffix_en.txt').read_text(encoding='utf-8').strip()}
```
"""
    (EXPERIMENT / "reports/final_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({"mode_summary": mode_summary, "paired": paired_summary, "errors": error_breakdown}, indent=2))


if __name__ == "__main__":
    main()
