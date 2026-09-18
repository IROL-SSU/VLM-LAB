#!/usr/bin/env python3
"""Report BLOCKED response counts, never silently treating family names as GT."""
import csv
import json
from collections import Counter

import run_l2_front_obstruction_geometry_v20 as run

FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")


def decision(row):
    return row["parsed_response"]["decision"]


def key(row):
    return row["scene_id"], row["condition_key"], row["seed"]


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plots(out, keys, families, scenes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False})
    directory = out / "plots"
    directory.mkdir(exist_ok=True)
    files = []
    lookup = {(r["condition_key"], r["scene_family"]): r for r in families}
    labels = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT"]

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(f"plots/{name}.png")

    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    data = [[lookup[(k, f)]["BLOCKED"] for k in keys] for f in FAMILIES]
    mesh = ax.imshow(data, cmap="Oranges", vmin=0, vmax=25, aspect="auto")
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right")
    ax.set_yticks(range(5), labels)
    ax.set_title("v20: 앞쪽 물체가 꺼내기를 조금이라도 방해하는가?\n각 칸: 방해 있다고 답한 횟수 (BLOCKED /25) · 정답 개수 아님", loc="left", pad=16)
    for i, line in enumerate(data):
        for j, value in enumerate(line):
            ax.text(j, i, f"{value}/25", ha="center", va="center", color="white" if value >= 17 else "#172033")
    fig.colorbar(mesh, ax=ax, shrink=.85, label="BLOCKED 응답 횟수")
    save(fig, "01_blocked_by_condition_family")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True, layout="constrained")
    for ax, k, name in zip(axes, ("C0", "C2_D8", "C4_D8"), ("RGB only", "R 정성형 D8", "F 정성형 D8")):
        group = [lookup[(k, f)] for f in FAMILIES]
        for offset, field, color, label in ((-.2, "v19_BLOCKED", "#9AA8B9", "v19 접근 방해"),
                                          (.2, "BLOCKED", "#D66B21", "v20 앞쪽 회수 방해")):
            bars = ax.bar([i + offset for i in range(5)], [r[field] for r in group], .38, color=color, label=label)
            ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xticks(range(5), labels, rotation=32, ha="right")
        ax.set_title(f"{k} · {name}", weight="bold")
        ax.set_ylim(0, 31)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("BLOCKED 응답 횟수 /25")
    axes[1].legend(loc="upper center", fontsize=9)
    fig.suptitle("질문 변경 전후의 BLOCKED 응답 분포\n질문 의미가 다르므로 정확도 비교가 아닙니다", fontsize=16)
    save(fig, "02_v19_v20_blocked_comparison")

    ids = sorted({r["scene_id"] for r in scenes})
    lookup_scene = {(r["scene_id"], r["condition_key"]): r for r in scenes}
    data = [[lookup_scene[(s, k)]["BLOCKED"] for k in keys] for s in ids]
    fig, ax = plt.subplots(figsize=(13, 12), layout="constrained")
    mesh = ax.imshow(data, cmap="Oranges", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right")
    ax.set_yticks(range(len(ids)), [s.replace("scene_", "") for s in ids])
    ax.set_title("v20 장면별 BLOCKED 응답\n각 칸: 5개 seed 중 방해 있다고 답한 횟수 /5", loc="left", pad=16)
    for i, line in enumerate(data):
        for j, value in enumerate(line):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value >= 3 else "#172033")
    fig.colorbar(mesh, ax=ax, shrink=.6, label="BLOCKED /5")
    save(fig, "03_scene_blocked_matrix")
    return files


def main():
    config, schedule = run.prepare()
    records = run.engine.rows(run.DEST / "logs/runs.jsonl")
    previous = run.engine.rows(run.SOURCE / "logs/runs.jsonl")
    run.validate(config, schedule, records)
    assert len(previous) == 1500
    assert [key(r) for r in records] == [key(r) for r in previous]
    prior = {key(r): r for r in previous}
    assert len(prior) == 1500
    for row in records:
        old = prior[key(row)]
        for field in ("image_sha256", "id_mapping_sha256", "geometry_sha256", "geometry_json_raw",
                      "target_object_id", "scene_family", "model_id", "model_path", "generation"):
            assert row[field] == old[field], (row["run_id"], field)
        for field in ("python", "vllm", "transformers"):
            assert row["runtime"][field] == old["runtime"][field]
    keys = [c["condition"] + ("_" + c["resolution"] if c["resolution"] else "") for c in config["conditions"]]
    names = {k: c["name"] for k, c in zip(keys, config["conditions"])}
    conditions, families, scenes = [], [], []
    for k in keys:
        group = [r for r in records if r["condition_key"] == k]
        assert len(group) == 125
        tally = Counter(decision(r) for r in group)
        unanimous = 0
        for scene in sorted({r["scene_id"] for r in group}):
            local = [r for r in group if r["scene_id"] == scene]
            assert len(local) == 5 and {r["seed"] for r in local} == set(config["seeds"])
            counts = Counter(decision(r) for r in local)
            same = max(counts.values()) == 5
            unanimous += same
            scenes.append({"scene_id": scene, "scene_family": local[0]["scene_family"], "condition_key": k,
                           "target_object_id": local[0]["target_object_id"], "BLOCKED": counts["BLOCKED"],
                           "CLEAR": counts["CLEAR"], "unanimous": same,
                           "v19_BLOCKED": sum(decision(prior[key(r)]) == "BLOCKED" for r in local)})
        family_map = {}
        for family in FAMILIES:
            local = [r for r in group if r["scene_family"] == family]
            assert len(local) == 25
            counts = Counter(decision(r) for r in local)
            value = {"condition_key": k, "condition_name": names[k], "scene_family": family,
                     "n": 25, "BLOCKED": counts["BLOCKED"], "CLEAR": counts["CLEAR"],
                     "v19_BLOCKED": sum(decision(prior[key(r)]) == "BLOCKED" for r in local)}
            families.append(value)
            family_map[family] = value
        conditions.append({"condition_key": k, "condition_name": names[k], "n": 125,
                           "BLOCKED": tally["BLOCKED"], "CLEAR": tally["CLEAR"],
                           "unanimous_scenes": unanimous, "by_family": family_map})
    paired = [{"scene_id": r["scene_id"], "scene_family": r["scene_family"], "condition_key": r["condition_key"],
               "seed": r["seed"], "v19_approach_decision": decision(prior[key(r)]),
               "v20_front_obstruction_decision": decision(r)} for r in records]
    attempts = run.engine.rows(run.DEST / "logs/attempts.jsonl")
    summary = {
        "experiment_version": run.VERSION, "completed_calls": len(records),
        "model_id": config["model_id"], "schema_valid_calls": 1500,
        "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in attempts),
        "all_first_attempt": all(r["attempt"] == 1 for r in records),
        "total_generation_seconds": round(sum(r["latency_seconds"] for r in records), 3),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
        "overall": dict(Counter(decision(r) for r in records)),
        "overall_by_family": {f: dict(Counter(decision(r) for r in records if r["scene_family"] == f)) for f in FAMILIES},
        "accuracy": None, "front_obstruction_ground_truth_available": False,
        "interpretation": "BLOCKED response frequency, not correct counts. CLEAR is not a guarantee of grasp/removal feasibility.",
        "condition_results": conditions, "finished_at": run.engine.now(),
    }
    out = run.DEST / "results"
    out.mkdir(exist_ok=True)
    for name, data in (("family_results", families), ("scene_results", scenes), ("paired_vs_v19", paired)):
        run.engine.write_json(out / f"{name}.json", data)
        write_csv(out / f"{name}.csv", data)
    write_csv(out / "condition_results.csv", conditions, [k for k in conditions[0] if k != "by_family"])
    summary["plots"] = plots(out, keys, families, scenes)
    run.engine.write_json(out / "summary.json", summary)
    run.engine.write_json(out / "integrity_report.json", {
        "status": "pass", "unique_complete_scene_condition_seed_combinations": 1500,
        "same_v19_images_geometry_model_settings_runtime_seeds_order": True,
        "all_logged_prompts_match_schedule_and_agreed_question": True,
        "schema_valid_raw_json_calls": 1500,
        "logs_sha256": run.engine.digest(run.DEST / "logs/runs.jsonl"),
        "v19_logs_sha256": run.engine.digest(run.SOURCE / "logs/runs.jsonl"),
    })
    lines = ["# L2 앞쪽 물체의 회수 방해 × Geometry 단일정보 — v20 결과", "",
             "25장면 × 12조건 × 5 seed = 1,500회. v19와 이미지·Geometry·모델·생성 설정·실행 환경·seed·순서를 유지했다. 질문/system/출력 label을 변경했다.", "",
             "BLOCKED = 앞쪽 물체가 타깃을 꺼내는 것을 조금이라도 방해한다고 응답. CLEAR = 그런 방해가 없다고 응답.", "",
             "표의 숫자는 BLOCKED 응답 횟수이며 정답 개수가 아니다. 앞쪽 회수 방해 GT는 별도로 없으며, family 이름을 정답으로 취급하지 않는다. CLEAR도 옆쪽 간섭·파지 공간 등을 포함한 전체 회수 가능성을 보증하지 않는다.", "",
             f"완료/JSON 유효: 1500/1500. 인프라 오류 {summary['infrastructure_errors']}, 모두 첫 시도: {summary['all_first_attempt']}. 생성 시간 합계 {summary['total_generation_seconds']}초.", "",
             f"전체 응답: BLOCKED {summary['overall'].get('BLOCKED', 0)}회, CLEAR {summary['overall'].get('CLEAR', 0)}회.", "",
             "## 실제 프롬프트", "", "System:", "```text", config["system_prompt"], "```", "",
             "User:", "```text", config["user_prompt_template"], "```", "",
             "## 전체 조건: BLOCKED 응답 횟수", "",
             "각 family는 5장면 × 5 seed = 25회, 조건별 전체는 125회. LIFT는 LIFT_AND_RELOCATE의 줄임말.", "",
             "| 조건 | 정보 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT | BLOCKED /125 | CLEAR /125 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in conditions:
        lines.append(f"| {r['condition_key']} | {r['condition_name']} | " + " | ".join(str(r['by_family'][f]['BLOCKED']) for f in FAMILIES) + f" | {r['BLOCKED']} | {r['CLEAR']} |")
    lines += ["", "## v19 → v20 BLOCKED 응답 횟수 (/25)", "",
              "v19는 접근 방해, v20는 앞쪽 물체의 회수 방해를 질문한다. 이름이 같은 BLOCKED라도 의미가 달라 정확도 향상 비교가 아니다.", "",
              "| 조건 | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * 5]
    for r in conditions:
        lines.append("| " + r["condition_key"] + " | " + " | ".join(f"{r['by_family'][f]['v19_BLOCKED']} → {r['by_family'][f]['BLOCKED']}" for f in FAMILIES) + " |")
    lines += ["", "## 장면별 BLOCKED 응답 (/5)", "", "| 장면 | 타깃 | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
    lookup = {(r["scene_id"], r["condition_key"]): r for r in scenes}
    for scene in sorted({r["scene_id"] for r in scenes}):
        lines.append(f"| {scene} | {lookup[(scene, 'C0')]['target_object_id']} | " + " | ".join(str(lookup[(scene, k)]['BLOCKED']) for k in keys) + " |")
    lines += ["", "## 그림", ""]
    for file in summary["plots"]:
        lines += [f"![v20 결과]({file})", ""]
    lines += ["## 재현", "", "```bash", ".qwen3-vl/venv/bin/python scripts/run_l2_front_obstruction_geometry_v20.py",
              "uv run --no-project --python 3.13 --with matplotlib python scripts/analyze_l2_front_obstruction_geometry_v20.py", "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed": len(records), "overall": summary["overall"], "by_family": summary["overall_by_family"],
                      "report": str(out / "report.md"), "conditions": [{k: v for k, v in r.items() if k != "by_family"} for r in conditions]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
