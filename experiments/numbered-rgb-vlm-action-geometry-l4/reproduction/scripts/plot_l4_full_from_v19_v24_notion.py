#!/usr/bin/env python3
"""Create Notion-ready visual summaries for the L4 full-matrix experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l4_full_from_v19_v24"
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


def add_bar_labels(ax, bars, *, fmt="{:.0f}", suffix=""):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.8, ax.get_ylim()[1] * 0.012),
            fmt.format(value) + suffix,
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=COLORS["dark"],
        )


def main() -> None:
    results = read_json(EXP / "results/results.json")
    runs = read_jsonl(EXP / "logs/runs.jsonl")
    replay = read_jsonl(EXP / "logs/replay.jsonl")
    metrics = results["condition_metrics"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 17,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # 1. Why L4 has 1,750 calls rather than the L3 v19 total of 1,500.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(2)
    base = np.array([250, 250])
    directional = np.array([1250, 1250])
    extra = np.array([0, 250])
    ax.bar(x, base, color=COLORS["gray"], label="C0/C1 base: 2 cells × 125")
    ax.bar(x, directional, bottom=base, color=COLORS["blue"], label="C2–C6: 10 cells × 125")
    ax.bar(x, extra, bottom=base + directional, color=COLORS["orange"], label="L4 extra: C0/C1 second resolution")
    ax.set_xticks(x, ["L3 v19\n12 cells", "Native L4\n14 cells"])
    ax.set_ylabel("Model calls")
    ax.set_ylim(0, 1950)
    ax.set_title("Why the Full L4 Matrix Has 1,750 Calls")
    ax.text(0, 1535, "1,500", ha="center", fontsize=18, fontweight="bold", color=COLORS["dark"])
    ax.text(1, 1785, "1,750", ha="center", fontsize=18, fontweight="bold", color=COLORS["dark"])
    ax.annotate("+250 calls\nC0/C1 split into D4 and D8", xy=(1, 1625), xytext=(0.45, 1820),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
                color=COLORS["red"], fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "01_why_1750_calls.png")

    # 2. End-to-end funnel.
    stages = ["Scheduled", "JSON parsed", "Rule-valid", "Replay success"]
    values = [1750, 1750, 461, 131]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["yellow"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 1900)
    ax.set_xlabel("Runs")
    ax.set_title("L4 Evaluation Funnel")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 25, bar.get_y() + bar.get_height() / 2,
                f"{value:,}  ({100*value/1750:.1f}%)", va="center",
                fontsize=12, fontweight="bold", color=COLORS["dark"])
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "02_evaluation_funnel.png")

    # 3. Accuracy heatmap by condition and resolution.
    names = ["C0 Image", "C1 R num", "C2 R qual", "C3 F num", "C4 F qual", "C5 M num", "C6 M qual"]
    lookup = {(m["geometry_condition"], m["direction_resolution"]): 100*m["accuracy"] for m in metrics}
    matrix = np.array([[lookup[(f"C{i}", r)] for r in ("D4", "D8")] for i in range(7)])
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=16, aspect="auto")
    ax.set_xticks([0, 1], ["D4", "D8"])
    ax.set_yticks(range(7), names)
    ax.set_title("Final Action Success by Condition (%)")
    for i in range(7):
        for j in range(2):
            color = "white" if matrix[i, j] >= 10 else COLORS["dark"]
            ax.text(j, i, f"{matrix[i,j]:.1f}%\n{int(round(matrix[i,j]*1.25))}/125",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, label="Success (%)", shrink=0.84)
    finish(fig, "03_condition_accuracy_heatmap.png")

    # 4. Rule-valid versus final-success rate for all 14 cells.
    labels = [f"{m['geometry_condition']}_{m['direction_resolution']}" for m in metrics]
    valid = np.array([100*m["post_validation_pass_n"]/m["n"] for m in metrics])
    success = np.array([100*m["accuracy"] for m in metrics])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6.2))
    w = 0.38
    b1 = ax.bar(x-w/2, valid, width=w, color=COLORS["yellow"], label="Rule-valid")
    b2 = ax.bar(x+w/2, success, width=w, color=COLORS["green"], label="Replay success")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Output Validity vs Final Action Success")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for bars in (b1, b2):
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    finish(fig, "04_validity_vs_success.png")

    # 5. Scene-family accuracy.
    fam_lookup = {row["scene_family"]: 100*row["accuracy"] for row in results["scene_family_metrics"]}
    families = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
    vals = [fam_lookup[name] for name in families]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = ax.bar(families, vals, color=[COLORS["green"]] + [COLORS["red"]]*4)
    ax.set_ylim(0, 45)
    ax.set_ylabel("Final success (%)")
    ax.set_title("Final Success by Scene Family")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    add_bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    finish(fig, "05_family_accuracy.png")

    # 6. Action distribution.
    counts = Counter(row["parsed_response"]["action"] for row in runs)
    action_names = ["TRANSLATE", "RETRIEVE", "ROTATE", "LIFT_AND_RELOCATE"]
    action_values = [counts.get(name, 0) for name in action_names]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(action_names, action_values,
                  color=[COLORS["red"], COLORS["blue"], COLORS["gray"], COLORS["gray"]])
    ax.set_ylim(0, 1450)
    ax.set_ylabel("Outputs")
    ax.set_title("Model Action Distribution — Only Two Actions Were Used")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.2)
    add_bar_labels(ax, bars)
    finish(fig, "06_action_distribution.png")

    # 7. Aggregated D4/D8 comparison.
    resolution_totals = {}
    resolutions = ("D4", "D8")
    for resolution in resolutions:
        group = [m for m in metrics if m["direction_resolution"] == resolution]
        resolution_totals[resolution] = sum(m["correct_n"] for m in group)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    vals = [100*resolution_totals[r]/875 for r in resolutions]
    bars = ax.bar(resolutions, vals, color=[COLORS["blue"], COLORS["orange"]], width=0.56)
    ax.set_ylim(0, 12)
    ax.set_ylabel("Final success (%)")
    ax.set_title("Aggregated Resolution Effect")
    ax.grid(axis="y", alpha=0.2)
    add_bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    for index, resolution in enumerate(resolutions):
        ax.text(index, vals[index]/2, f"{resolution_totals[resolution]}/875",
                ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    finish(fig, "07_d4_vs_d8.png")

    # 8. Failure mechanism flow.
    action_valid = sum(bool(row["action_valid"]) for row in replay)
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Dominant Failure Flow", pad=20)

    def box(x, y, w, h, title, subtitle, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                               linewidth=2, edgecolor=color, facecolor="white")
        ax.add_patch(patch)
        ax.text(x+w/2, y+h*0.62, title, ha="center", va="center",
                fontsize=17, fontweight="bold", color=color)
        ax.text(x+w/2, y+h*0.28, subtitle, ha="center", va="center",
                fontsize=10.5, color=COLORS["dark"])

    def arrow(x1, y1, x2, y2, color):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=2.3, color=color))

    box(0.3, 2.55, 2.4, 1.55, "1,750", "All parsed JSON", COLORS["blue"])
    box(4.0, 4.45, 3.2, 1.55, "1,289 TRANSLATE", "Target itself selected\n→ rule-invalid", COLORS["red"])
    box(4.0, 1.05, 3.2, 1.55, "461 RETRIEVE", "Only outputs sent to replay", COLORS["yellow"])
    box(9.2, 3.35, 3.1, 1.55, "330 failed", "Target was not directly graspable", COLORS["red"])
    box(9.2, 0.75, 3.1, 1.55, f"{action_valid} succeeded", "All successes from FC_CLEAR", COLORS["green"])
    arrow(2.7, 3.5, 4.0, 5.2, COLORS["red"])
    arrow(2.7, 3.2, 4.0, 1.8, COLORS["yellow"])
    arrow(7.2, 1.85, 9.2, 4.1, COLORS["red"])
    arrow(7.2, 1.65, 9.2, 1.5, COLORS["green"])
    ax.text(3.15, 5.0, "73.7%", color=COLORS["red"], fontweight="bold", fontsize=11)
    ax.text(3.15, 1.45, "26.3%", color=COLORS["dark"], fontweight="bold", fontsize=11)
    finish(fig, "08_failure_flow.png")

    print(json.dumps({"output_dir": str(OUT), "files": sorted(p.name for p in OUT.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
