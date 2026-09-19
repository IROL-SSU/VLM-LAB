# Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L4

| 항목 | 내용 |
|---|---|
| Status | **L4 v24–v26 snapshot complete; overall program ongoing** |
| Main model | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| Generation runtime | vLLM offline, batch size 1 |
| Scene set | 25 scenes, 5 scene families |
| Repeats | 장면–조건별 seed 5개 (`28101`–`28105`) |
| Archived inference calls | **5,250** (study별 1,750) |
| Archived replay records | **3,500** (v24·v25 각각 1,750) |
| Source project | `/home/ssu/ShelfScene` |

L4는 Numbered RGB에서 지정된 target에 대해 **다음 한 동작**을 선택하는 단계입니다. [L2 direct retrieval](../numbered-rgb-vlm-action-geometry-l2/)과 [L3 blocking reason·blocker 식별](../numbered-rgb-vlm-action-geometry-l3/)에 이어, 동작 전체 출력(v24), L3 blocker 조건부 동작(v25), action label만 출력(v26)을 비교합니다. 기초 시각 능력에 해당하는 L1은 별도 재실험하지 않고 선행 [N0·N1·N2·N3 평가](../numbered-rgb-vlm-basics-n0-n3/)를 전제로 합니다.

이 아카이브는 canonical Numbered RGB 입력으로 수행한 L4 기록을 대상으로 합니다. 세 study의 25개 scene 이미지 hash는 L3 v19 source와 일치합니다.

## 1. 공통 실험 매트릭스

- Scene family: `FC_CLEAR`, `FC_BLOCKED`, `TRANSLATE`, `ROTATE`, `LIFT_AND_RELOCATE` (각 5 scenes)
- Geometry 정보: C0 Numbered RGB only, C1 `R` numeric, C2 `R` qualitative, C3 `F` numeric, C4 `F` qualitative, C5 `M` numeric, C6 `M` qualitative
- C0–C6 각각 D4/D8 셀로 기록: **14 cells × 25 scenes × 5 seeds = 1,750 calls/study**
- C0/C1은 D4/D8 비교 셀을 유지한 것으로, 방향별 추가 geometry가 제공되는 조건으로 해석하지 않습니다.
- 각 study에서 이미지, geometry, seed를 고정하고 L4 과제·입력 단계·출력 schema를 변경했습니다.

## 2. 수행한 L4 study

| Study | L4 입력과 출력 | 평가 방식 | 주요 결과 |
|---|---|---|---|
| [v24 full action](studies/full-action-v24/) | L3 v19의 scene 입력을 사용하되 L3 예측은 주지 않음. `RETRIEVE`, `TRANSLATE`, `ROTATE`, `LIFT_AND_RELOCATE` 중 한 동작과 필요한 object ID·방향·거리/각도 등의 파라미터 출력 | 출력 검증 후 Isaac Sim direct replay | replay-defined 유효 동작 **131/1,750 (7.5%)** |
| [v25 blocker-conditioned](studies/blocker-conditioned-v25/) | 같은 scene–condition–seed의 frozen L3 v19 `blocking_reason`·`blocker_id`를 제공. `NONE`이면 target `RETRIEVE`, 아니면 선택된 blocker에 대한 동작·파라미터 출력 | 출력 검증 후 Isaac Sim direct replay | replay-defined 유효 동작 **350/1,750 (20.0%)** |
| [v26 action-only](studies/action-only-v26/) | v25와 같은 frozen L3 입력. `{"action":"..."}` 한 필드만 출력 | Scene manifest의 *advisory* action label과 일치율 비교; replay 없음 | advisory label 일치 **878/1,750 (50.2%)** |

v24와 v25 모두 inference와 replay **기록** 1,750건을 완료했고 parse failure는 0건입니다. 다만 출력 규칙을 통과하지 못한 경우 시뮬레이터 동작을 실행하지 않았습니다.

| Study | Semantic post-validation failure | Simulator-executed runs | Replay-defined action-valid runs | 주요 출력 분포 |
|---|---:|---:|---:|---|
| v24 | 1,289 | 461 | 131 | `RETRIEVE` 461, `TRANSLATE` 1,289 |
| v25 | 178 | 1,572 | 350 | `RETRIEVE` 181, `TRANSLATE` 1,569 |

v25의 action-valid 수치가 v24보다 높지만, 두 study 사이에는 L3 blocker 제공과 prompt 제약 변경이 함께 있으므로 차이를 단일 요소의 인과효과로 해석하지 않습니다. Replay-defined action validity는 본 실험의 Isaac Sim 장면·충돌·경계·graspability 기준이며 실제 로봇 동작 성공률이 아닙니다.

## 3. v25 C2-D8 실패 분석

v25의 C2-D8 셀은 **41/125 (32.8%)**가 replay-defined 유효 동작이었습니다. [후속 실패 분석](studies/blocker-conditioned-v25/results/c2_d8_failure_analysis/summary.json)은 나머지 84건의 주된 실패 원인을 분류했습니다.

| 주된 실패 원인 | 건수 |
|---|---:|
| Action 종류 불일치 | 49 |
| 방향 불일치 | 9 |
| 거리 불일치 | 9 |
| 권고 action·parameter와 일치했으나 물리 replay 실패 | 15 |
| 출력 사전 검증 오류 | 2 |

이 분류는 실패 사례를 설명하기 위한 분석입니다. Scene manifest의 action과 parameter는 `advisory_only`이며, v25의 공식 유효성은 direct Isaac Sim replay로 판단합니다.

## 4. v26 action-only 결과

v26은 JSON parsing과 action-only schema를 **1,750/1,750** 통과했습니다. 출력 분포는 `TRANSLATE` 1,110, `LIFT_AND_RELOCATE` 459, `RETRIEVE` 181, `ROTATE` 0건입니다. Scene advisory action과의 일치율은 전체 **50.2%**, C2-D8 **68/125 (54.4%)**였습니다.

- Frozen L3 reason과 blocker가 모두 맞았던 셀: v26 action 일치 **456/590 (77.3%)**
- Frozen L3 joint 오답 셀: v26 action 일치 **422/1,160 (36.4%)**
- `ROTATE`가 advisory인 350건에서 `ROTATE` 출력은 0건
- v25와 같은 **action-label 기준**으로 비교하면 v25 879/1,750, v26 878/1,750. 이는 v25의 replay 유효성과 다른 지표입니다.

v26은 object ID·동작 방향·거리·각도·목적지를 출력하지 않아 실행 가능한 전체 동작을 평가하지 않습니다. [run-level 결과](studies/action-only-v26/results/run_outcomes.csv), [집계](studies/action-only-v26/results/results.json), [confusion matrix](studies/action-only-v26/results/action_confusion_matrix.png)를 보존했습니다.

## 5. 보존 범위와 재현

```text
numbered-rgb-vlm-action-geometry-l4/
  README.md
  SOURCES.md
  studies/
    full-action-v24/
    blocker-conditioned-v25/
    action-only-v26/
  reproduction/scripts/
  audit/archive_manifest.csv
```

- **모든 추론 원시 로그:** v24·v25·v26 각각 `runs.jsonl` 1,750건과 `attempts.jsonl` 1,750건
- **모든 replay 원시 로그:** v24·v25 각각 `replay.jsonl` 1,750건과 `replay_attempts.jsonl` 24/227건
- 각 study의 config/run table, prompt/schema, audit, 집계 결과, failure analysis·figure, v24·v25 simulator replay cache와 post-mask
- [원본과 포함·제외 범위](SOURCES.md), [실행·분석 코드](reproduction/README.md), [파일별 hash manifest](audit/archive_manifest.csv)

전체 입력 이미지와 geometry collection은 중복 용량 때문에 Git snapshot에 포함하지 않았습니다. 원본 scene/run 경로와 SHA-256은 config·raw log·source audit에 남아 있습니다.

## 6. 해석 경계

- v24·v25의 replay-defined action validity와 v26의 advisory action-label agreement는 **서로 다른 지표**이며 한 표의 성공률로 직접 비교할 수 없습니다.
- v24·v25에서 replay 기록 1,750건은 시뮬레이터에서 동작을 실행한 횟수와 다릅니다. 검증 실패 실행은 기록되지만 동작은 실행되지 않습니다.
- Scene manifest의 action·parameter는 유일한 물리 정답이 아닌 권고값입니다. 다른 동작도 성공할 수 있습니다.
- L3 예측이 L4 입력으로 들어가는 v25·v26에서는 상류 L3 오류가 L4 결과에 영향을 줄 수 있습니다.
- 5개 seed는 sampling 안정성을 보여주지만 독립적인 scene 125개를 뜻하지 않습니다.
