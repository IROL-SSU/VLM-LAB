#!/usr/bin/env python3
"""Create action-only v26 Notion visuals without cross-version comparisons."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l4_action_only_v26"
OUT = EXP / "results/notion_visualizations"
OUT.mkdir(parents=True, exist_ok=True)
ACTIONS = ["RETRIEVE", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
COLORS = {
    "blue": "#4E79A7", "orange": "#F28E2B", "red": "#E15759",
    "green": "#59A14F", "teal": "#76B7B2", "yellow": "#EDC948",
    "purple": "#B07AA1", "dark": "#243447",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save(fig, filename: str):
    fig.savefig(OUT / filename, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def labels(ax, bars, fmt="{:.1f}%"):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(0.5, ax.get_ylim()[1] * 0.012),
                fmt.format(value), ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=COLORS["dark"])


def make_case_panel(run: dict, filename: str):
    gt = read_json(EXP / run["ground_truth"])
    rgb = Image.open(EXP / run["numbered_rgb"]).convert("RGB")
    rgb.thumbnail((1150, 760), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (2000, 980), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    try:
        title_font = ImageFont.truetype(bold_path, 36)
        head_font = ImageFont.truetype(bold_path, 28)
        body_font = ImageFont.truetype(font_path, 24)
        mono_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 23)
    except OSError:
        title_font = head_font = body_font = mono_font = ImageFont.load_default()
    advisory = gt["L4"]["manifest_focus_action_advisory_only"]
    predicted = run["parsed_response"]["action"]
    match = advisory == predicted
    banner = COLORS["green"] if match else COLORS["red"]
    draw.rectangle((0, 0, 2000, 95), fill=banner)
    draw.text((48, 22), f"{run['scene_id']} · C2 D8 · seed {run['seed']}", fill="white", font=title_font)
    canvas.paste(rgb, (30 + (1150 - rgb.width) // 2, 115))
    draw.text((55, 875), "Input: numbered RGB", fill=COLORS["dark"], font=head_font)
    x = 1220
    draw.text((x, 140), "Frozen L3 output", fill=COLORS["blue"], font=head_font)
    draw.multiline_text((x, 185), json.dumps(run["l3_source_response"], indent=2),
                        fill=COLORS["dark"], font=mono_font, spacing=4)
    draw.text((x, 320), "Scene GT advisory", fill=COLORS["blue"], font=head_font)
    draw.text((x, 365), f"{advisory}", fill=COLORS["dark"], font=mono_font)
    draw.text((x, 430), "Model JSON", fill=COLORS["orange"], font=head_font)
    draw.multiline_text((x, 475), json.dumps(run["parsed_response"], indent=2),
                        fill=COLORS["dark"], font=mono_font, spacing=4)
    draw.text((x, 615), "Advisory label agreement", fill=COLORS["dark"], font=head_font)
    draw.text((x, 665), "MATCH" if match else "MISMATCH", fill=banner, font=title_font)
    draw.text((x, 745), "No simulator replay: parameters omitted", fill=COLORS["dark"], font=body_font)
    canvas.save(OUT / filename, quality=95)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 16})
    runs = read_jsonl(EXP / "logs/runs.jsonl")
    result = read_json(EXP / "results/results.json")
    assert len(runs) == 1750

    # 1. Action-only evaluation topology.
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("v26 Action-Only Evaluation Pipeline", pad=15)
    boxes = [
        (0.2, 2.0, 2.4, "Observation", "Numbered RGB\n+ C0-C6 geometry", COLORS["blue"]),
        (3.0, 2.0, 2.5, "Frozen L3", "blocking_reason\n+ blocker_id", COLORS["purple"]),
        (5.9, 2.0, 2.3, "L4 choice", '{"action": "..."}\nonly one key', COLORS["orange"]),
        (8.6, 2.0, 2.2, "Format check", "1,750 / 1,750\nvalid JSON", COLORS["teal"]),
        (11.2, 2.0, 2.6, "Advisory match", "878 / 1,750\n50.2%", COLORS["green"]),
    ]
    for x, y, w, title, subtitle, color in boxes:
        patch = FancyBboxPatch((x, y), w, 1.6, boxstyle="round,pad=0.05,rounding_size=0.14",
                               edgecolor=color, facecolor="white", linewidth=2.2)
        ax.add_patch(patch)
        ax.text(x+w/2, y+1.12, title, ha="center", va="center", fontsize=14, fontweight="bold", color=color)
        ax.text(x+w/2, y+0.5, subtitle, ha="center", va="center", fontsize=10, color=COLORS["dark"])
    for a,b in ((2.6,3.0),(5.5,5.9),(8.2,8.6),(10.8,11.2)):
        ax.add_patch(FancyArrowPatch((a,2.8),(b,2.8),arrowstyle="-|>",mutation_scale=16,
                                     linewidth=2,color=COLORS["dark"]))
    ax.text(7, 0.85, "25 scenes × 7 conditions × D4/D8 × 5 seeds = 1,750 calls; no physical replay",
            ha="center", fontsize=13, fontweight="bold", color=COLORS["dark"])
    save(fig, "01_pipeline.png")

    # 2. Output action distribution.
    counts = result["output_distribution"]
    fig, ax = plt.subplots(figsize=(10.7,5.8))
    action_colors = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["green"]]
    bars = ax.bar(["RETRIEVE", "TRANSLATE", "ROTATE", "LIFT & RELOCATE"],
                  [counts.get(action,0) for action in ACTIONS], color=action_colors)
    ax.set_ylim(0, 1250)
    ax.set_ylabel("Model outputs")
    ax.set_title("v26 Action Output Distribution")
    ax.grid(axis="y", alpha=0.2)
    for bar, action in zip(bars, ACTIONS):
        value = counts.get(action,0)
        ax.text(bar.get_x()+bar.get_width()/2,value+18,f"{value:,}\n({100*value/1750:.1f}%)",
                ha="center",fontweight="bold",color=COLORS["dark"])
    save(fig,"02_action_distribution.png")

    # 3. Scene-family advisory agreement.
    fig, ax = plt.subplots(figsize=(11,5.8))
    family = result["by_family"]
    values = [100*family[name]["accuracy"] for name in FAMILIES]
    bars = ax.bar(["FC clear","FC blocked","Translate","Rotate","Lift & relocate"],
                  values,color=[COLORS["blue"],COLORS["teal"],COLORS["orange"],COLORS["red"],COLORS["green"]])
    ax.set_ylim(0,105)
    ax.set_ylabel("Advisory action-label agreement (%)")
    ax.set_title("v26 Agreement by Scene Family (350 runs each)")
    ax.grid(axis="y",alpha=0.2)
    labels(ax,bars)
    save(fig,"03_family_agreement.png")

    # 4. L3 correctness effect, including the blocked-only subset.
    l3_all = result["by_l3_joint_correctness"]
    l3_blocked = result["by_l3_joint_correctness_non_none_only"]
    fig,ax=plt.subplots(figsize=(9.8,5.6))
    x=np.arange(2); w=0.35
    a=ax.bar(x-w/2,[100*l3_all[k]["accuracy"] for k in ("joint_correct","joint_wrong")],w,
             label="All runs",color=COLORS["blue"])
    b=ax.bar(x+w/2,[100*l3_blocked[k]["accuracy"] for k in ("joint_correct","joint_wrong")],w,
             label="Non-NONE only",color=COLORS["orange"])
    ax.set_xticks(x,["L3 reason + ID correct","L3 joint wrong"])
    ax.set_ylim(0,90)
    ax.set_ylabel("Advisory action-label agreement (%)")
    ax.set_title("v26 Agreement Conditional on Frozen L3 Correctness")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=0.2)
    labels(ax,a); labels(ax,b)
    save(fig,"04_l3_correctness.png")

    # 5-6. Split family × geometry heatmaps by D4/D8.
    for resolution,suffix in (("D4","d4"),("D8","d8")):
        matrix=np.zeros((5,7))
        for fi,family_name in enumerate(FAMILIES):
            for ci in range(7):
                group=[r for r in runs if r["scene_family"]==family_name
                       and r["geometry_condition"]==f"C{ci}"
                       and r["direction_resolution"]==resolution]
                assert len(group)==25
                matrix[fi,ci]=100*sum(r["scoring"]["action_matches_advisory"] for r in group)/25
        fig,ax=plt.subplots(figsize=(11.7,6.3))
        im=ax.imshow(matrix,cmap="YlGnBu",vmin=0,vmax=100,aspect="auto")
        ax.set_xticks(range(7),[f"C{i}" for i in range(7)])
        ax.set_yticks(range(5),["FC clear","FC blocked","Translate","Rotate","Lift & relocate"])
        ax.set_title(f"v26 {resolution} Agreement: Family × Geometry Condition\nEach cell = 5 scenes × 5 seeds = 25 calls")
        for i in range(5):
            for j in range(7):
                ax.text(j,i,f"{matrix[i,j]:.0f}%",ha="center",va="center",
                        color="white" if matrix[i,j]>=60 else COLORS["dark"],fontweight="bold")
        fig.colorbar(im,ax=ax,label="Advisory action-label agreement (%)",shrink=0.82)
        save(fig,f"05_family_condition_{suffix}.png")

    # 7. Frozen reason → model output.
    reasons=["NONE","CLEARANCE_OVERLAP","OCCLUSION","BOTH"]
    matrix=np.array([[sum(r["selected_blocking_reason"]==reason and r["parsed_response"]["action"]==action for r in runs)
                      for action in ACTIONS] for reason in reasons])
    fig,ax=plt.subplots(figsize=(9.6,6.4))
    im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=1050,aspect="auto")
    ax.set_xticks(range(4),["RETRIEVE","TRANSLATE","ROTATE","LIFT & RELOCATE"],rotation=15,ha="right")
    ax.set_yticks(range(4),reasons)
    ax.set_xlabel("Model output action")
    ax.set_ylabel("Frozen L3 blocking reason")
    ax.set_title("v26 Output Action by L3 Reason")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",
                    color="white" if matrix[i,j]>=700 else COLORS["dark"],fontweight="bold",fontsize=12)
    fig.colorbar(im,ax=ax,label="Runs",shrink=0.8)
    save(fig,"07_l3_reason_action_matrix.png")

    # 8. C2 D8 confusion.
    subset=[r for r in runs if r["geometry_condition"]=="C2" and r["direction_resolution"]=="D8"]
    assert len(subset)==125
    matrix=np.array([[sum(r["scoring"]["advisory_action"]==gt and r["parsed_response"]["action"]==pred for r in subset)
                      for pred in ACTIONS] for gt in ACTIONS])
    fig,ax=plt.subplots(figsize=(8.6,6.7))
    im=ax.imshow(matrix,cmap="YlGnBu",vmin=0,vmax=50)
    action_labels=["RETRIEVE","TRANSLATE","ROTATE","LIFT & RELOCATE"]
    ax.set_xticks(range(4),action_labels,rotation=20,ha="right")
    ax.set_yticks(range(4),action_labels)
    ax.set_xlabel("Model output")
    ax.set_ylabel("Scene advisory action")
    ax.set_title("v26 C2 D8 Action Confusion (n=125)")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",
                    color="white" if matrix[i,j]>=30 else COLORS["dark"],fontweight="bold",fontsize=14)
    fig.colorbar(im,ax=ax,label="Runs",shrink=0.8)
    save(fig,"08_c2_d8_confusion.png")

    # 9. Number of distinct actions across five seeds for each scene-condition-resolution.
    groups=defaultdict(set)
    for row in runs:
        groups[(row["scene_id"],row["geometry_condition"],row["direction_resolution"])].add(row["parsed_response"]["action"])
    diversity=Counter(len(actions) for actions in groups.values())
    assert len(groups)==350
    fig,ax=plt.subplots(figsize=(8.6,5.4))
    bars=ax.bar([str(i) for i in range(1,5)],[diversity.get(i,0) for i in range(1,5)],
                color=[COLORS["blue"],COLORS["orange"],COLORS["red"],COLORS["purple"]])
    ax.set_xlabel("Distinct actions across 5 seeds")
    ax.set_ylabel("Scene × condition × resolution cells")
    ax.set_ylim(0,350)
    ax.set_title("v26 Seed-Level Action Diversity Across 350 Cells")
    ax.grid(axis="y",alpha=0.2)
    labels(ax,bars,fmt="{:.0f}")
    save(fig,"09_seed_action_diversity.png")

    # 10-14. Same C2 D8 observation format, contrasting action cases.
    cases=[
        ("scene_fc_clear_v01",28102,"10_case_retrieve_match.png"),
        ("scene_fc_blocked_v01",28102,"11_case_translate_match.png"),
        ("scene_lift_and_relocate_v02",28102,"12_case_lift_match.png"),
        ("scene_translate_v02",28102,"13_case_translate_lift_mismatch.png"),
        ("scene_rotate_v02",28105,"14_case_rotate_lift_mismatch.png"),
    ]
    for scene_id,seed,filename in cases:
        row=next(r for r in subset if r["scene_id"]==scene_id and r["seed"]==seed)
        make_case_panel(row,filename)

    shutil.copy2(EXP / "results/action_confusion_matrix.png",OUT / "15_full_confusion_matrix.png")
    shutil.copy2(EXP / "results/condition_action_agreement.png",OUT / "16_condition_agreement.png")
    v26_only={key:value for key,value in result.items() if key!="paired_v25_action_advisory_match"}
    v26_only["scope"] = "v26 action-only: 25 scenes x 7 conditions x 2 resolutions x 5 seeds"
    v26_only["l3_reason_action_counts"]={reason:{action:int(matrix_value) for action,matrix_value in zip(ACTIONS,row)}
                                          for reason,row in zip(reasons,np.array([[sum(r["selected_blocking_reason"]==reason and r["parsed_response"]["action"]==action for r in runs) for action in ACTIONS] for reason in reasons]))}
    v26_only["seed_action_diversity"]={str(k):v for k,v in sorted(diversity.items())}
    (OUT / "v26_only_summary.json").write_text(json.dumps(v26_only,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":sorted(p.name for p in OUT.glob("*.png")),
                      "seed_action_diversity":diversity,"scope":"v26 only"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
