"""Render a focused explanatory figure from completed v18 results; no inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/vlm_action_geometry_single_info_l2_approach_access_geometry_v18/results"
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_name = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
plt.rcParams.update({"font.family": font_name, "font.size": 11, "axes.unicode_minus": False})
summary = json.loads((RESULTS / "summary.json").read_text())
conditions = {r["condition_key"]: r for r in summary["condition_results"]}
families = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
labels = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT"]
focus = [("C0", "RGB만 제공"), ("C2_D8", "RGB + R 정성형 D8"), ("C4_D8", "RGB + F 정성형 D8")]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
for ax, (key, title) in zip(axes, focus):
    accessible = [conditions[key]["by_family"][f]["ACCESSIBLE"] for f in families]
    blocked = [conditions[key]["by_family"][f]["BLOCKED"] for f in families]
    assert all(a + b == 25 for a, b in zip(accessible, blocked))
    ax.barh(range(5), accessible, color="#168C82", height=.56)
    ax.barh(range(5), blocked, left=accessible, color="#D7DFE8", height=.56)
    for y, (a, b) in enumerate(zip(accessible, blocked)):
        if a >= 3:
            ax.text(a / 2, y, f"{a}", ha="center", va="center", color="white", weight="bold", fontsize=13)
        elif a:
            ax.text(a / 2, y -.34, f"{a}", ha="center", va="center", color="#087468", weight="bold", fontsize=12)
        if b:
            ax.text(a + b / 2, y, f"{b}", ha="center", va="center", color="#354457", fontsize=13)
    ax.set_title(f"{key}\n{title}", fontsize=14, weight="bold", pad=14)
    ax.set_xlim(0, 25)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("응답 횟수 / 25")
    ax.set_yticks(range(5), labels)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle("같은 접근 질문에, 입력 정보만 다르게 주었을 때", fontsize=21, weight="bold", y=.98)
fig.legend(handles=[Patch(color="#168C82", label="ACCESSIBLE · 접근 가능 응답"),
                    Patch(color="#D7DFE8", label="BLOCKED · 접근 차단 응답")],
           loc="lower center", bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=12)
fig.text(.5, .025, "각 막대 = 같은 family의 5장면 × 5개 seed. 응답 분포이며 정확도나 실제 회수 성공률이 아님.", ha="center", fontsize=10, color="#526073")
fig.subplots_adjust(left=.12, right=.98, top=.76, bottom=.23, wspace=.15)
for extension in ("png", "pdf"):
    path = RESULTS / "plots" / f"05_explained_c0_r_f.{extension}"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(path)
plt.close(fig)
