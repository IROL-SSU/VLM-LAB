# Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L2

| 항목 | 내용 |
|---|---|
| Status | **L2 snapshot complete; overall program ongoing** |
| Main model | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| Runtime | vLLM offline, batch size 1 |
| Main repeats | 장면–조건별 고정 seed 5개 (`28101`–`28105`) |
| Main scene set | 25 scenes, 5 scene families |
| Archived generation calls | **21,210** |
| Source project | `/home/ssu/ShelfScene` |

Numbered RGB로 지정된 target에 대해 VLM이 어느 수준의 manipulation 판단을 내릴 수 있는지, 그리고 한 번에 한 종류의 geometry 정보를 추가했을 때 판단이 어떻게 달라지는지를 평가하는 실험 프로그램입니다. 이 snapshot은 그중 **L2 direct retrieval** 단계의 유효한 실행 기록 전체를 보존합니다.

기초 시각 능력에 해당하는 L1은 별도 재실험하지 않습니다. ID 인식, ID–물체 의미 연결, 자연어 target grounding, 공간관계 추론은 선행 실험인 [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](../numbered-rgb-vlm-basics-n0-n3/)에서 다뤘으며, 본 프로그램은 그 결과를 전제로 L2부터 시작합니다.

```text
N0–N3: Numbered RGB 기초 능력 검증
                    ↓
L2: target을 지금 직접 회수할 수 있는가?
                    ↓
L3·L4: 후속 action-level 판단으로 확장
```

이 문서와 아카이브는 canonical Numbered RGB 입력으로 수행한 유효 설계와 결과만을 대상으로 합니다.

## 1. L2 연구 질문

L2의 최종 과제는 다음 이진 판단입니다.

> 현재 장면에서 다른 물체를 제자리에 둔 채 접촉을 피하면서 target을 잡아 선반 개구부 밖으로 꺼낼 수 있는가?

출력은 blocker 식별이나 행동 제안 없이 하나의 decision만 반환합니다.

```json
{"decision":"RETRIEVE_NOW"}
```

- `RETRIEVE_NOW`: 다른 물체의 재배치 없이 target을 회수할 수 있음
- `REARRANGE_FIRST`: target 회수 전에 다른 물체를 이동해야 함

부분적으로 가려져 있거나 다른 물체와 가깝다는 사실만으로 차단이라고 판단하지 않도록 prompt에 명시했습니다.

## 2. 공통 실험 설계

### Scene set

- 총 25 scenes
- scene family 5종, family별 5 scenes
  - `FC_CLEAR`
  - `FC_BLOCKED`
  - `TRANSLATE`
  - `ROTATE`
  - `LIFT_AND_RELOCATE`
- 각 scene–condition에 seed 5개를 고정 적용
- main comparison은 25 scenes × 12 conditions × 5 seeds = **1,500 calls**

### Geometry 단일정보 조건

모든 조건은 동일한 Numbered RGB와 target ID를 사용합니다. 한 조건에는 한 geometry representation family만 제공합니다.

| 조건 | 추가 정보 | 표현 |
|---|---|---|
| C0 | 없음 | Numbered RGB only |
| C1 | `R` | numeric |
| C2-D4 / C2-D8 | `R` | qualitative, 4/8 directions |
| C3-D4 / C3-D8 | `F` | numeric, 4/8 directions |
| C4-D4 / C4-D8 | `F` | qualitative, 4/8 directions |
| C5-D4 / C5-D8 | `M` | numeric, 4/8 directions |
| C6-D4 / C6-D8 | `M` | qualitative, 4/8 directions |

각 실행에는 가능한 범위에서 scene/run ID, condition, seed, model/runtime, generation setting, image·prompt·schema hash, raw response, token 수와 latency를 저장했습니다.

## 3. L2 실험 흐름

L2는 판단 의미를 분리해 확인한 뒤 direct retrieval 과제로 수렴시키고, main comparison과 재현성·내부표현 분석으로 이어졌습니다.

| 구간 | 목적 | 아카이브 |
|---|---|---|
| v2–v10 | visibility, occlusion, access, interference, blocker 등 판단 predicate 분리 검증 | `formal-predicate-v2`부터 `primary-grasp-blocker-v10` |
| v11–v14 | 단일 scene·단일 image 기반 obstruction 진단 | `single-scene-instance-obstruction-v11`부터 `simple-obstruction-score-v14` |
| v15 | direct retrieval prompt의 single-image 및 25-scene pilot | `direct-retrieval-*-v15` |
| v16 | 12개 geometry 조건 main comparison | `direct-retrieval-geometry-v16` |
| v18–v20 | approach, straight access, front obstruction 의미 분해 | `*-geometry-v18`부터 `*-geometry-v20` |
| v21 | v16 prompt replay | `v16-prompt-replay-v21` |
| v22–v23 | C2-D8/C2-D4 decision readout latent 분석 | `c2d8-decision-latent-v22`, `c2d4-decision-latent-v23` |

`plain-rgb-named-object-obstruction-v12`는 Numbered RGB main comparison과 분리된 plain-RGB 진단입니다. L2 과제 정의를 점검한 실험 흐름의 일부이므로 함께 보존하되 main geometry 결과로 합산하지 않습니다.

## 4. Main result — v16

[`studies/direct-retrieval-geometry-v16`](studies/direct-retrieval-geometry-v16/)은 25 scenes × 12 conditions × 5 seeds의 **1,500 calls**를 완료했습니다.

- schema-valid: 1,500/1,500
- infrastructure error: 0
- C0 scene-design agreement: 82.4%
- C2-D4 scene-design agreement: 87.2%
- C2-D8 scene-design agreement: 93.6%
- v21 exact replay: v16 대비 decision 변경 0/1,500

여기서 scene-design reference는 `FC_CLEAR → RETRIEVE_NOW`, 나머지 family → `REARRANGE_FIRST`인 설계 의도입니다. **실제 robot retrieval execution ground truth는 없으므로 위 agreement를 물리적 성공률이나 accuracy로 해석하지 않습니다.**

세부 condition별 결과와 scene별 5-seed 분포는 다음 파일에 있습니다.

- [`results/summary.json`](studies/direct-retrieval-geometry-v16/results/summary.json)
- [`results/condition_results.csv`](studies/direct-retrieval-geometry-v16/results/condition_results.csv)
- [`results/scene_results.json`](studies/direct-retrieval-geometry-v16/results/scene_results.json)
- [`results/paired_changes_vs_c0.json`](studies/direct-retrieval-geometry-v16/results/paired_changes_vs_c0.json)

## 5. Decision latent follow-up — v22·v23

v16의 C2-D8과 C2-D4 조건에서 scene별 5개 historical response를 묶고, 첫 분기 token 직전의 decoder state를 scene 단위로 분석했습니다. 새 generation call을 추가한 실험이 아니라 v16 artifact에서 파생한 25-scene 분석입니다.

| 분석 | Scene | Raw cosine, mid | Raw cosine, final | Branch logit ↔ historical majority | Branch logit ↔ scene design |
|---|---:|---:|---:|---:|---:|
| C2-D8, v22 | 25 | 0.991748 | 0.985997 | 21/25 | 20/25 |
| C2-D4, v23 | 25 | 0.992142 | 0.983769 | 19/25 | 16/25 |

이 값은 descriptive analysis입니다. 25개 scene만 사용했고 scene family, object identity, image layout, prompt 길이와 geometry payload가 함께 변하므로 인과적인 reasoning mechanism의 증거로 해석하지 않습니다. 또한 latent state를 추출한 Transformers BF16 경로와 historical vLLM constrained sampling 사이의 수치적 동일성을 가정하지 않습니다.

## 6. 보존한 실행 로그

이 snapshot에는 **21,210개의 `runs.jsonl` record와 21,210개의 대응 `attempts.jsonl` record**가 있습니다.

| 실험 | runs | attempts |
|---|---:|---:|
| formal predicate v2 | 1,500 | 1,500 |
| grasp zone v3 | 1,500 | 1,500 |
| visibility only v4 | 1,500 | 1,500 |
| occlusion boolean v5 | 1,500 | 1,500 |
| target access v6 | 1,500 | 1,500 |
| potential interference v7 | 1,500 | 1,500 |
| instance interference v8 | 1,500 | 1,500 |
| instance occlusion LTR v9 | 1,500 | 1,500 |
| primary grasp blocker v10 | 1,500 | 1,500 |
| single-scene instance obstruction v11 | 5 | 5 |
| plain-RGB named-object obstruction v12 | 25 | 25 |
| single-image obstruction score v13 | 25 | 25 |
| simple obstruction score v14 | 25 | 25 |
| direct retrieval single-image v15 | 5 | 5 |
| direct retrieval 25-scenes v15 | 125 | 125 |
| direct retrieval geometry v16 | 1,500 | 1,500 |
| approach access geometry v18 | 1,500 | 1,500 |
| straight access geometry v19 | 1,500 | 1,500 |
| front obstruction geometry v20 | 1,500 | 1,500 |
| v16 prompt replay v21 | 1,500 | 1,500 |
| **Total** | **21,210** | **21,210** |

v22와 v23은 각각 25개의 derived scene record를 포함하지만 새 generation call은 없으므로 위 합계에는 더하지 않았습니다.

## 7. 디렉터리 구성과 재현

```text
numbered-rgb-vlm-action-geometry-l2/
  README.md
  SOURCES.md
  studies/                 # L2 study별 config, prompt, schema, raw log, result
  reproduction/scripts/    # 실행·분석 코드 snapshot
  audit/archive_manifest.csv
```

- 실행·분석 코드 목록과 사용 경계는 [`reproduction/README.md`](reproduction/README.md)에 있습니다.
- 원본 실험 경로와 포함·제외 범위는 [`SOURCES.md`](SOURCES.md)에 있습니다.
- 파일별 SHA-256, byte size, JSONL row count는 [`audit/archive_manifest.csv`](audit/archive_manifest.csv)에 있습니다.

전체 입력 이미지, geometry payload collection, Isaac Sim asset와 checkpoint는 중복 용량 때문에 포함하지 않았습니다. raw log의 scene/image 경로와 SHA-256, 각 study의 config·manifest를 사용해 원본 입력을 추적할 수 있습니다.

## 8. 해석 경계

- L2는 VLM의 이진 판단을 측정하며 실제 robot execution 성공을 직접 검증하지 않습니다.
- geometry 조건 간 차이는 정보 representation에 대한 반응이며 물리적 최적성의 직접 증거가 아닙니다.
- 5개 seed 결과는 sampling 안정성을 보여주지만 scene 다양성을 대신하지 않습니다.
- L2의 결론을 blocker identity나 multi-step action sequence 능력으로 확장하지 않습니다.
