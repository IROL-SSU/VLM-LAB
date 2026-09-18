#!/usr/bin/env python3
"""Compare a fresh Notion v16 prompt replay with the original v16 responses."""
import csv
import json
from collections import Counter

import run_l2_v16_prompt_replay_v21 as run

FAMILIES = ("FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE")
LABELS = ("RETRIEVE_NOW", "REARRANGE_FIRST")


def decision(row):
    return row["parsed_response"]["decision"]


def key(row):
    return row["scene_id"], row["condition_key"], row["seed"]


def tally(rows):
    counts = Counter(decision(r) for r in rows)
    return {label: counts[label] for label in LABELS}


def design(row):
    return "RETRIEVE_NOW" if row["scene_family"] == "FC_CLEAR" else "REARRANGE_FIRST"


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plots(out, keys, families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "font.size": 10, "axes.unicode_minus": False})
    directory = out / "plots"
    directory.mkdir(exist_ok=True)
    lookup = {(r["condition_key"], r["scene_family"]): r for r in families}
    files = []
    for delta in (False, True):
        values = [[lookup[(k, f)]["REARRANGE_FIRST"] - (lookup[(k, f)]["v16_REARRANGE_FIRST"] if delta else 0) for k in keys] for f in FAMILIES]
        fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
        mesh = ax.imshow(values, cmap="RdBu_r" if delta else "Oranges", vmin=-25 if delta else 0, vmax=25, aspect="auto")
        ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right")
        ax.set_yticks(range(5), ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT"])
        title = "원본 v16 → 재실행 v21: 먼저 재배치 응답 횟수 차이" if delta else "노션 v16 프롬프트 재실행: 먼저 재배치 응답"
        ax.set_title(title + "\n각 유형·조건 25회 기준 · 정확도나 정답 개수가 아닙니다", loc="left", pad=16)
        for i, line in enumerate(values):
            for j, value in enumerate(line):
                ax.text(j, i, f"{value:+d}" if delta else f"{value}/25", ha="center", va="center",
                        color="white" if abs(value) >= 17 else "#172033")
        fig.colorbar(mesh, ax=ax, shrink=.85, label="REARRANGE_FIRST 횟수 차이" if delta else "REARRANGE_FIRST /25")
        name = "02_delta_vs_original_v16" if delta else "01_rearrange_by_condition_family"
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(f"plots/{name}.png")
    return files


def main():
    config, schedule = run.prepare()
    records = run.engine.rows(run.DEST / "logs/runs.jsonl")
    original = run.engine.rows(run.SOURCE / "logs/runs.jsonl")
    run.validate(config, schedule, records)
    assert len(original) == 1500 and [key(r) for r in records] == [key(r) for r in original]
    old = {key(r): r for r in original}
    assert len(old) == 1500
    for row in records:
        previous = old[key(row)]
        for field in ("image_sha256", "id_mapping_sha256", "geometry_sha256", "geometry_json_raw", "target_object_id",
                      "scene_family", "model_id", "model_path", "generation", "system_prompt", "user_prompt", "prompt_sha256", "schema", "schema_sha256"):
            assert row[field] == previous[field], (row["run_id"], field)
        for field in ("python", "vllm", "transformers"):
            assert row["runtime"][field] == previous["runtime"][field]
    keys = [c["condition"] + ("_" + c["resolution"] if c["resolution"] else "") for c in config["conditions"]]
    names = {k: c["name"] for k, c in zip(keys, config["conditions"])}
    conditions, families, scenes = [], [], []
    for k in keys:
        group = [r for r in records if r["condition_key"] == k]
        assert len(group) == 125
        unanimous = 0
        for scene in sorted({r["scene_id"] for r in group}):
            local = [r for r in group if r["scene_id"] == scene]
            assert len(local) == 5 and {r["seed"] for r in local} == set(config["seeds"])
            counts = tally(local)
            same = max(counts.values()) == 5
            unanimous += same
            scenes.append({"scene_id": scene, "scene_family": local[0]["scene_family"], "condition_key": k,
                           "target_object_id": local[0]["target_object_id"], **counts, "unanimous": same,
                           "v16_REARRANGE_FIRST": sum(decision(old[key(r)]) == "REARRANGE_FIRST" for r in local),
                           "changed_vs_v16": sum(decision(r) != decision(old[key(r)]) for r in local)})
        family_map = {}
        for family in FAMILIES:
            local = [r for r in group if r["scene_family"] == family]
            assert len(local) == 25
            value = {"condition_key": k, "condition_name": names[k], "scene_family": family,
                     "n": 25, **tally(local), "v16_REARRANGE_FIRST": sum(decision(old[key(r)]) == "REARRANGE_FIRST" for r in local),
                     "changed_vs_v16": sum(decision(r) != decision(old[key(r)]) for r in local)}
            families.append(value)
            family_map[family] = value
        matches = sum(decision(r) == design(r) for r in group)
        old_matches = sum(decision(old[key(r)]) == design(r) for r in group)
        conditions.append({"condition_key": k, "condition_name": names[k], "n": 125, **tally(group),
                           "unanimous_scenes": unanimous, "scene_design_agreement_count": matches,
                           "scene_design_agreement_rate": matches / 125,
                           "v16_scene_design_agreement_count": old_matches,
                           "changed_vs_v16": sum(decision(r) != decision(old[key(r)]) for r in group), "by_family": family_map})
    paired = [{"scene_id": r["scene_id"], "scene_family": r["scene_family"], "condition_key": r["condition_key"],
               "seed": r["seed"], "v16_decision": decision(old[key(r)]), "v21_decision": decision(r),
               "changed": decision(old[key(r)]) != decision(r)} for r in records]
    attempts = run.engine.rows(run.DEST / "logs/attempts.jsonl")
    summary = {"experiment_version": run.VERSION, "completed_calls": 1500, "schema_valid_calls": 1500,
               "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in attempts),
               "all_first_attempt": all(r["attempt"] == 1 for r in records),
               "total_generation_seconds": round(sum(r["latency_seconds"] for r in records), 3),
               "overall": tally(records), "changed_vs_v16": sum(r["changed"] for r in paired),
               "accuracy": None, "retrieval_execution_gt_available": False,
               "scene_design_reference": "Original Notion reference: FC_CLEAR => RETRIEVE_NOW; other families => REARRANGE_FIRST. Not execution GT; not the later preference to retrieve FC_BLOCKED.",
               "condition_results": conditions, "finished_at": run.engine.now()}
    out = run.DEST / "results"
    out.mkdir(exist_ok=True)
    for name, data in (("family_results", families), ("scene_results", scenes), ("paired_vs_v16", paired)):
        run.engine.write_json(out / f"{name}.json", data)
        write_csv(out / f"{name}.csv", data)
    write_csv(out / "condition_results.csv", conditions, [k for k in conditions[0] if k != "by_family"])
    summary["plots"] = plots(out, keys, families)
    run.engine.write_json(out / "summary.json", summary)
    run.engine.write_json(out / "integrity_report.json", {
        "status": "pass", "unique_complete_scene_condition_seed_combinations": 1500,
        "same_v16_inputs_prompts_schema_model_settings_runtime_seeds_order": True,
        "notion_prompt_verified_against_local_v16_config": True,
        "schema_valid_raw_json_calls": 1500, "logs_sha256": run.engine.digest(run.DEST / "logs/runs.jsonl"),
        "original_v16_logs_sha256": run.engine.digest(run.SOURCE / "logs/runs.jsonl"),
    })
    lines = ["# 노션 v16 직접 회수 프롬프트 재실행 — v21", "",
             f"원문: [{run.engine.read(run.DEST / 'config/notion_prompt_source.json')['title']}]({config['notion_url']})", "",
             "노션 system/user/schema가 원본 v16 config와 정확히 일치함을 확인했다. 이미지·Geometry·모델·생성 설정·실행 환경·seed·순서·프롬프트·schema를 동일하게 유지하여 새 추론 1,500회를 실행했다. 이전 응답은 재사용하지 않았다.", "",
             f"완료/JSON 유효 1500/1500, 오류 {summary['infrastructure_errors']}, 모두 첫 시도: {summary['all_first_attempt']}. 생성 시간 합계 {summary['total_generation_seconds']}초.", "",
             f"전체: RETRIEVE_NOW {summary['overall']['RETRIEVE_NOW']}, REARRANGE_FIRST {summary['overall']['REARRANGE_FIRST']}. 원본 v16과 개별 응답이 다른 호출 {summary['changed_vs_v16']}/1500.", "",
             "## 전체 조건 응답: 바로 회수 / 먼저 재배치", "",
             "각 칸 합계는 25회. 정답 개수가 아니라 두 응답의 횟수다.", "",
             "| 조건 | 정보 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT | 전체 회수 / 재배치 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in conditions:
        lines.append(f"| {r['condition_key']} | {r['condition_name']} | " + " | ".join(f"{r['by_family'][f]['RETRIEVE_NOW']} / {r['by_family'][f]['REARRANGE_FIRST']}" for f in FAMILIES) + f" | {r['RETRIEVE_NOW']} / {r['REARRANGE_FIRST']} |")
    lines += ["", "## 원본 노션 기준 장면 설계 일치율", "",
              "FC_CLEAR는 RETRIEVE_NOW, 나머지 네 family는 REARRANGE_FIRST를 기대하는 원본 노션의 참고 지표다. 실제 회수 성공률·물리적 정확도가 아니며, 이후 논의의 FC_BLOCKED를 회수 쪽으로 유도하려는 목표와도 다르다.", "",
              "| 조건 | 원본 v16 | 재실행 v21 | 개별 응답 변경 /125 | 5회 만장일치 장면 /25 |", "|---|---:|---:|---:|---:|"]
    for r in conditions:
        lines.append(f"| {r['condition_key']} | {r['v16_scene_design_agreement_count']}/125 ({r['v16_scene_design_agreement_count']/125:.1%}) | {r['scene_design_agreement_count']}/125 ({r['scene_design_agreement_rate']:.1%}) | {r['changed_vs_v16']} | {r['unanimous_scenes']} |")
    lines += ["", "## 실제 사용 프롬프트", "", "System:", "```text", config["system_prompt"], "```", "",
              "User:", "```text", config["user_prompt_template"], "```", "", "## 장면별 회수 / 재배치 횟수", "",
              "| 장면 | 타깃 | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
    lookup = {(r["scene_id"], r["condition_key"]): r for r in scenes}
    for scene in sorted({r["scene_id"] for r in scenes}):
        lines.append(f"| {scene} | {lookup[(scene, 'C0')]['target_object_id']} | " + " | ".join(f"{lookup[(scene,k)]['RETRIEVE_NOW']} / {lookup[(scene,k)]['REARRANGE_FIRST']}" for k in keys) + " |")
    lines += ["", "## 그림", ""]
    for path in summary["plots"]:
        lines += [f"![v21 결과]({path})", ""]
    lines += ["## 재현", "", "```bash", ".qwen3-vl/venv/bin/python scripts/run_l2_v16_prompt_replay_v21.py",
              "uv run --no-project --python 3.13 --with matplotlib python scripts/analyze_l2_v16_prompt_replay_v21.py", "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": summary["overall"], "changed_vs_v16": summary["changed_vs_v16"],
                      "conditions": [{k: v for k,v in r.items() if k != "by_family"} for r in conditions],
                      "report": str(out / "report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
