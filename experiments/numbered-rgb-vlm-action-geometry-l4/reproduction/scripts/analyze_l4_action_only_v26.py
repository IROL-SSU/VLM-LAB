#!/usr/bin/env python3
"""Score action-only L4 labels against the scene's advisory action."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l4_action_only_v26"
OUT = EXP / "results"
ACTIONS = ("RETRIEVE", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")
FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rate(rows: list[dict]) -> dict:
    correct = sum(row["advisory_match"] for row in rows)
    return {"total": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None}


def main() -> None:
    runs = read_jsonl(EXP / "logs/runs.jsonl")
    if len(runs) != 1750 or len({r["run_id"] for r in runs}) != 1750:
        raise RuntimeError(f"Expected exactly 1,750 unique runs, got {len(runs)}")
    assert all(row["action_only"] for row in runs)
    records = []
    for run in runs:
        predicted = run["parsed_response"]["action"] if isinstance(run["parsed_response"], dict) else None
        gt = run["scoring"]["advisory_action"]
        records.append({
            "run_id": run["run_id"],
            "scene_id": run["scene_id"],
            "scene_family": run["scene_family"],
            "geometry_condition": run["geometry_condition"],
            "direction_resolution": run["direction_resolution"],
            "seed": run["seed"],
            "target_id": run["target_object_id"],
            "l3_reason": run["selected_blocking_reason"],
            "l3_blocker_id": run["selected_blocker_id"],
            "l3_joint_correct": run["l3_source_joint_correct"],
            "l3_blocker_correct": run["l3_source_blocker_correct"],
            "advisory_action": gt,
            "predicted_action": predicted,
            "advisory_match": predicted == gt,
            "post_validation_pass": run["post_validation_pass"],
            "raw_response": run["raw_response"],
        })
    records.sort(key=lambda row: (row["geometry_condition"], row["direction_resolution"], row["scene_id"], row["seed"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    by_family = {family: rate([row for row in records if row["scene_family"] == family]) for family in FAMILIES}
    by_cell = {
        f"C{c}_{d}": rate([row for row in records if row["geometry_condition"] == f"C{c}" and row["direction_resolution"] == d])
        for c in range(7) for d in ("D4", "D8")
    }
    by_condition = {
        f"C{c}": rate([row for row in records if row["geometry_condition"] == f"C{c}"])
        for c in range(7)
    }
    by_l3 = {
        name: rate([row for row in records if row["l3_joint_correct"] is value])
        for name, value in (("joint_correct", True), ("joint_wrong", False))
    }
    by_l3_blocked = {
        name: rate([
            row for row in records
            if row["l3_joint_correct"] is value and row["l3_reason"] != "NONE"
        ])
        for name, value in (("joint_correct", True), ("joint_wrong", False))
    }
    by_resolution = {d: rate([row for row in records if row["direction_resolution"] == d]) for d in ("D4", "D8")}
    confusion = {
        gt: {pred: sum(row["advisory_action"] == gt and row["predicted_action"] == pred for row in records) for pred in ACTIONS}
        for gt in ACTIONS
    }
    source = ROOT / "experiments/vlm_action_geometry_single_info_l4_blocker_conditioned_v25/logs/runs.jsonl"
    v25 = {r["run_id"]: r for r in read_jsonl(source)}
    paired = Counter()
    for row in records:
        old = v25[row["run_id"].replace("v26__", "v25__")]
        old_action = old["parsed_response"]["action"]
        old_correct = old_action == row["advisory_action"]
        new_correct = row["advisory_match"]
        paired[(old_correct, new_correct)] += 1
    summary = {
        "scope": "v26 action-only, same 1,750 v25 input/L3 cells",
        "scoring_note": "advisory action-label agreement; not official physical action_valid",
        "overall": rate(records),
        "parse_count": sum(isinstance(r["parsed_response"], dict) for r in runs),
        "schema_valid_count": sum(r["post_validation_pass"] for r in runs),
        "output_distribution": dict(Counter(row["predicted_action"] for row in records)),
        "by_family": by_family,
        "by_condition_resolution": by_cell,
        "by_condition": by_condition,
        "by_resolution": by_resolution,
        "by_l3_joint_correctness": by_l3,
        "by_l3_joint_correctness_non_none_only": by_l3_blocked,
        "confusion_gt_rows_prediction_columns": confusion,
        "paired_v25_action_advisory_match": {
            "v25_correct_v26_correct": paired[(True, True)],
            "v25_correct_v26_wrong": paired[(True, False)],
            "v25_wrong_v26_correct": paired[(False, True)],
            "v25_wrong_v26_wrong": paired[(False, False)],
            "v25_advisory_correct": paired[(True, True)] + paired[(True, False)],
            "v26_advisory_correct": paired[(True, True)] + paired[(False, True)],
        },
    }
    (OUT / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    colors = {"D4": "#4E79A7", "D8": "#F28E2B"}
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.34
    for shift, resolution in ((-0.5, "D4"), (0.5, "D8")):
        values = [100 * by_cell[f"C{i}_{resolution}"]["accuracy"] for i in range(7)]
        bars = ax.bar(x + shift * width, values, width, label=resolution, color=colors[resolution])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6,
                    f"{value:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"C{i}" for i in range(7)])
    ax.set_ylim(0, 66)
    ax.set_ylabel("Advisory action-label agreement (%)")
    ax.set_title("v26 Action-Only: Agreement by Geometry Condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "condition_action_agreement.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    matrix = np.array([[confusion[gt][pred] for pred in ACTIONS] for gt in ACTIONS])
    fig, ax = plt.subplots(figsize=(8.8, 7))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=700)
    labels = ["RETRIEVE", "TRANSLATE", "ROTATE", "LIFT & RELOCATE"]
    ax.set_xticks(range(4), labels, rotation=20, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Model output")
    ax.set_ylabel("Scene advisory action")
    ax.set_title("v26 Action-Only Confusion Matrix (n=1,750)")
    for i in range(4):
        for j in range(4):
            value = matrix[i, j]
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value >= 400 else "#243447", fontweight="bold", fontsize=14)
    fig.colorbar(image, ax=ax, label="Runs")
    fig.savefig(OUT / "action_confusion_matrix.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
