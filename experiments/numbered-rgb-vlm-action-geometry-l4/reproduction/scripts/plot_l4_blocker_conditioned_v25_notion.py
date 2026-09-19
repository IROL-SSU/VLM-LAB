#!/usr/bin/env python3
"""Create Notion-ready figures for the L3-conditioned L4 v25 experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l4_blocker_conditioned_v25"
OUT = EXP / "results/notion_visualizations"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "red": "#E15759",
    "green": "#59A14F",
    "teal": "#76B7B2",
    "yellow": "#EDC948",
    "purple": "#B07AA1",
    "gray": "#BAB0AC",
    "dark": "#243447",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finish(fig, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}%") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.6, ax.get_ylim()[1] * 0.012),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["dark"],
        )


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 17,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def make_case_panel(runs: dict, replay: dict, run_id: str, filename: str, title: str) -> None:
    run = runs[run_id]
    result = replay[run_id]
    artifact = read_json(EXP / result["cache_artifact"])
    rgb = Image.open(EXP / run["numbered_rgb"]).convert("RGB")
    mask = Image.open(EXP / artifact["post_target_mask"]).convert("L")
    rgb.thumbnail((900, 620), Image.Resampling.LANCZOS)
    mask.thumbnail((900, 620), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (1900, 920), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    try:
        title_font = ImageFont.truetype(font_bold, 42)
        body_font = ImageFont.truetype(font_path, 27)
        small_font = ImageFont.truetype(font_path, 23)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((55, 35), title, fill="#243447", font=title_font)
    left_x, right_x, top_y = 45, 970, 130
    canvas.paste(rgb, (left_x + (860 - rgb.width) // 2, top_y))
    canvas.paste(mask.convert("RGB"), (right_x + (860 - mask.width) // 2, top_y))
    draw.text((left_x, 735), "Input: numbered RGB", fill="#4E79A7", font=body_font)
    draw.text((right_x, 735), "After action: target semantic mask", fill="#4E79A7", font=body_font)
    l3 = run["l3_source_response"]
    action = run["parsed_response"]
    summary = (
        f"L3 = {l3}    |    L4 = {action}\n"
        f"post_visibility={result['post_visibility']}, gap={result['post_minimum_surface_gap_cm']} cm, "
        f"collision={result['collision']}, success={result['action_valid']}"
    )
    draw.multiline_text((55, 790), summary, fill="#243447", font=small_font, spacing=8)
    canvas.save(OUT / filename, quality=95)


def main() -> None:
    configure_plotting()
    results = read_json(EXP / "results/results.json")
    runs_list = read_jsonl(EXP / "logs/runs.jsonl")
    replay_list = read_jsonl(EXP / "logs/replay.jsonl")
    runs = {row["run_id"]: row for row in runs_list}
    replay = {row["run_id"]: row for row in replay_list}

    # 1. Corrected two-stage pipeline.
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Corrected Two-Stage Evaluation Pipeline", pad=18)

    def box(x, y, w, h, title, subtitle, color):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.14",
            linewidth=2.2, edgecolor=color, facecolor="white"
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=16, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=10, color=COLORS["dark"])

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=2.2, color=COLORS["dark"]))

    box(0.2, 2.1, 2.25, 1.7, "Input", "Numbered RGB\n+ C0-C6 geometry", COLORS["blue"])
    box(3.05, 2.1, 2.55, 1.7, "Frozen L3 v19", "reason + one blocker ID\nexact scene/condition/seed", COLORS["purple"])
    box(6.2, 2.1, 2.55, 1.7, "L4 decision", "RETRIEVE target, or\naction on selected blocker", COLORS["orange"])
    box(9.35, 2.1, 2.0, 1.7, "Validation", "JSON + ID rule\n+ label consistency", COLORS["yellow"])
    box(11.95, 2.1, 1.85, 1.7, "Isaac Sim", "collision, boundary,\nvisibility, clearance", COLORS["green"])
    arrow(2.45, 2.95, 3.05, 2.95)
    arrow(5.60, 2.95, 6.20, 2.95)
    arrow(8.75, 2.95, 9.35, 2.95)
    arrow(11.35, 2.95, 11.95, 2.95)
    ax.text(7, 0.85, "25 scenes × 14 cells × 5 seeds = 1,750 L4 calls",
            ha="center", fontsize=15, fontweight="bold", color=COLORS["dark"])
    finish(fig, "01_pipeline.png")

    # 2. v25-only evaluation funnel.
    stages = ["Scheduled", "JSON parsed", "Pre-check passed", "Replay success"]
    current = [1750, 1750, 1572, 350]
    y = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    bars = ax.barh(y, current, height=0.58,
                   color=[COLORS["blue"], COLORS["teal"], COLORS["yellow"], COLORS["green"]])
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xlim(0, 1900)
    ax.set_xlabel("Runs")
    ax.set_title("v25 Evaluation Funnel")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, current):
        ax.text(value + 18, bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({100*value/1750:.1f}%)", va="center",
                fontsize=10, fontweight="bold", color=COLORS["dark"])
    finish(fig, "02_v25_evaluation_funnel.png")

    # 3. v25 condition accuracy by resolution.
    metrics = results["condition_metrics"]
    lookup = {(m["geometry_condition"], m["direction_resolution"]): 100 * m["accuracy"] for m in metrics}
    conditions = [f"C{i}" for i in range(7)]
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    w = 0.32
    d4 = [lookup[(c, "D4")] for c in conditions]
    d8 = [lookup[(c, "D8")] for c in conditions]
    b1 = ax.bar(x - w / 2, d4, w, color=COLORS["blue"], label="D4")
    b2 = ax.bar(x + w / 2, d8, w, color=COLORS["orange"], label="D8")
    ax.set_xticks(x, ["C0\nImage", "C1\nR num", "C2\nR qual", "C3\nF num", "C4\nF qual", "C5\nM num", "C6\nM qual"])
    ax.set_ylim(0, 38)
    ax.set_ylabel("Final action success (%)")
    ax.set_title("v25 Final Success by Geometry Condition and Direction Resolution")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    label_bars(ax, b1)
    label_bars(ax, b2)
    finish(fig, "03_v25_condition_accuracy.png")

    # 4. Condition × family heatmaps, split by direction resolution.
    families = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
    family_labels = ["FC clear", "FC blocked", "Translate", "Rotate", "Lift & relocate"]
    for resolution_name, suffix in (("D4", "d4"), ("D8", "d8")):
        matrix = np.zeros((len(families), 7))
        for fi, family in enumerate(families):
            for ci, condition in enumerate(conditions):
                group = [
                    replay[row["run_id"]]["action_valid"]
                    for row in runs_list
                    if row["scene_family"] == family
                    and row["geometry_condition"] == condition
                    and row["direction_resolution"] == resolution_name
                ]
                matrix[fi, ci] = 100 * sum(bool(v) for v in group) / len(group)
        fig, ax = plt.subplots(figsize=(11.8, 6.4))
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(7), conditions)
        ax.set_yticks(range(len(families)), family_labels)
        ax.set_title(
            f"v25 {resolution_name} Final Success: Scene Family × Geometry Condition\n"
            "Each cell: 5 scenes × 5 seeds = 25 runs"
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                        color="white" if matrix[i, j] >= 55 else COLORS["dark"], fontweight="bold")
        fig.colorbar(im, ax=ax, label="Success (%)", shrink=0.82)
        finish(fig, f"04_condition_family_heatmap_{suffix}.png")

    # 5. v25 family result.
    fam = {m["scene_family"]: 100 * m["accuracy"] for m in results["scene_family_metrics"]}
    labels = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT"]
    keys = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    a = ax.bar(x, [fam[k] for k in keys], 0.62,
               color=[COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["teal"]])
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, 62)
    ax.set_ylabel("Final success (%)")
    ax.set_title("v25 Final Success by Scene Family")
    ax.grid(axis="y", alpha=0.2)
    label_bars(ax, a)
    finish(fig, "05_v25_family_accuracy.png")

    # 6. Action collapse.
    counts = Counter(row["parsed_response"]["action"] for row in runs_list)
    actions = ["RETRIEVE", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
    values = [counts.get(action, 0) for action in actions]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.bar(actions, values, color=[COLORS["blue"], COLORS["red"], COLORS["gray"], COLORS["gray"]])
    ax.set_ylim(0, 1750)
    ax.set_ylabel("Model outputs")
    ax.set_title("Action Distribution — Every Non-NONE Case Became TRANSLATE")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 25, f"{value:,}\n({100*value/1750:.1f}%)",
                ha="center", fontweight="bold", color=COLORS["dark"])
    finish(fig, "06_action_distribution.png")

    # 7. Effect of frozen L3 correctness.
    fields = [
        ("Reason", "l3_source_reason_correct"),
        ("Blocker ID", "l3_source_blocker_correct"),
        ("Joint", "l3_source_joint_correct"),
    ]
    correct_rates, wrong_rates = [], []
    for _, field in fields:
        yes = [replay[row["run_id"]]["action_valid"] for row in runs_list if row[field]]
        no = [replay[row["run_id"]]["action_valid"] for row in runs_list if not row[field]]
        correct_rates.append(100 * sum(bool(v) for v in yes) / len(yes))
        wrong_rates.append(100 * sum(bool(v) for v in no) / len(no))
    x = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    w = 0.36
    a = ax.bar(x - w / 2, correct_rates, w, color=COLORS["green"], label="L3 correct")
    b = ax.bar(x + w / 2, wrong_rates, w, color=COLORS["red"], label="L3 wrong")
    ax.set_xticks(x, [name for name, _ in fields])
    ax.set_ylim(0, 45)
    ax.set_ylabel("Final action success (%)")
    ax.set_title("Upstream L3 Correctness Strongly Controls Downstream Success")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    label_bars(ax, a)
    label_bars(ax, b)
    finish(fig, "07_l3_correctness_effect.png")

    # 8. Resolution and physical failure indicators.
    resolution = {}
    for name in ("D4", "D8"):
        group = [replay[row["run_id"]]["action_valid"] for row in runs_list if row["direction_resolution"] == name]
        resolution[name] = 100 * sum(bool(v) for v in group) / len(group)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
    bars = axes[0].bar(["D4", "D8"], [resolution["D4"], resolution["D8"]],
                       color=[COLORS["blue"], COLORS["orange"]], width=0.6)
    axes[0].set_ylim(0, 25)
    axes[0].set_ylabel("Final success (%)")
    axes[0].set_title("Direction Resolution")
    axes[0].grid(axis="y", alpha=0.2)
    label_bars(axes[0], bars)

    physical = [row for row in replay_list if row["simulator_executed"]]
    failure_values = [
        sum(bool(row["collision"]) for row in physical),
        sum(bool(row["boundary_violation"]) for row in physical),
        sum(row["post_direct_graspable"] is False for row in physical),
    ]
    bars = axes[1].bar(["Collision", "Boundary", "Not graspable\nafter action"], failure_values,
                       color=[COLORS["red"], COLORS["orange"], COLORS["purple"]])
    axes[1].set_ylim(0, 1250)
    axes[1].set_ylabel("Executed runs")
    axes[1].set_title("Physical Failure Indicators (overlap allowed)")
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, failure_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 18, f"{value:,}", ha="center", fontweight="bold")
    finish(fig, "08_resolution_and_failures.png")

    # 9-11. Concrete scene+mask panels.
    make_case_panel(
        runs, replay,
        "v25__scene_fc_blocked_v02__L4__C2_D4__r03_s28103",
        "09_case_translate_success.png",
        "Success case: correct blocker + 3 cm LEFT translation",
    )
    make_case_panel(
        runs, replay,
        "v25__scene_translate_v02__L4__C2_D8__r05_s28105",
        "10_case_translate_failure.png",
        "Failure case: correct blocker, but target becomes NOT_VISIBLE",
    )
    make_case_panel(
        runs, replay,
        "v25__scene_fc_clear_v01__L4__C2_D8__r01_s28101",
        "11_case_retrieve_success.png",
        "Success case: L3 NONE gates RETRIEVE of the target",
    )

    summary = {
        "experiment": "l4_blocker_conditioned_from_l3_v19_v25",
        "scheduled": 1750,
        "parsed": 1750,
        "rule_valid": 1572,
        "stage_rule_adherent": 1741,
        "replay_success": 350,
        "actions": counts,
        "figures_for_v25_page": [
            "01_pipeline.png",
            "02_v25_evaluation_funnel.png",
            "03_v25_condition_accuracy.png",
            "04_condition_family_heatmap_d4.png",
            "04_condition_family_heatmap_d8.png",
            "05_v25_family_accuracy.png",
            "06_action_distribution.png",
            "07_l3_correctness_effect.png",
            "08_resolution_and_failures.png",
            "09_case_translate_success.png",
            "10_case_translate_failure.png",
            "11_case_retrieve_success.png",
        ],
    }
    (OUT / "notion_asset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
