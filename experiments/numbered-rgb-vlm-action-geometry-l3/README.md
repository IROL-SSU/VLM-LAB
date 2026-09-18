# Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L3

| 항목 | 내용 |
|---|---|
| Status | **L3 snapshot complete; overall program ongoing** |
| Main model | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| Generation runtime | vLLM offline, batch size 1 |
| Main repeats | 장면–조건별 고정 seed 5개 (`28101`–`28105`) |
| Main scene set | 25 scenes, 5 scene families |
| Archived generation calls | **4,500** |
| Latent follow-up | C2-D4, 25 scene vectors, cosine similarity |
| Source project | `/home/ssu/ShelfScene` |

L3는 Numbered RGB에서 지정된 target의 직접 회수를 막는 원인과 blocker ID를 식별하는 단계입니다. 동일한 장면 집합과 12개 geometry 단일정보 조건에서 세 가지 blocker 출력 정의를 비교하고, 최종 primary-blocker 과제의 C2-D4 reason readout을 latent space에서 추가 분석했습니다.

기초 시각 능력에 해당하는 L1은 별도 재실험하지 않습니다. ID 인식, ID–물체 의미 연결, 자연어 target grounding, 공간관계 추론은 선행 실험인 [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](../numbered-rgb-vlm-basics-n0-n3/)에서 다뤘습니다. 본 프로그램은 이를 전제로 [L2 direct retrieval](../numbered-rgb-vlm-action-geometry-l2/)부터 시작합니다.

```text
N0–N3: Numbered RGB 기초 능력 검증
                    ↓
L2: target을 지금 직접 회수할 수 있는가?
                    ↓
L3: 회수를 막는 원인과 blocker는 무엇인가?
```

이 문서와 아카이브는 canonical Numbered RGB 입력으로 수행한 유효 설계와 결과만을 대상으로 합니다.

## 1. L3 연구 질문

L3는 두 정보를 함께 예측합니다.

1. target 회수를 막는 이유: `OCCLUSION`, `CLEARANCE_OVERLAP`, `BOTH`, `NONE`
2. 장면에서 그 차단을 일으키는 visible object ID

세 generation study는 blocker를 반환하는 범위를 다르게 정의합니다.

| Study | Blocker 출력 정의 | 출력 예시 |
|---|---|---|
| v17 single causal blocker | 정답 집합에 속하는 visible blocker 하나; `NONE`이면 `null` | `{"blocking_reason":"OCCLUSION","blocker_id":17}` |
| v18 all visible blockers | 차단에 기여하는 visible blocker ID의 완전한 오름차순 목록 | `{"blocking_reason":"BOTH","blocker_ids":[17,42]}` |
| v19 primary blocker | 제거했을 때 graspability가 가장 크게 개선되어 먼저 이동해야 하는 blocker 하나 | `{"blocking_reason":"CLEARANCE_OVERLAP","blocker_id":42}` |

Reason과 blocker가 모두 맞아야 joint correct로 계산합니다.

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
- study별 25 scenes × 12 conditions × 5 seeds = **1,500 calls**

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

각 실행에는 scene/run ID, condition, seed, model/runtime, generation setting, image·prompt·schema hash, raw response, token 수와 latency가 저장되어 있습니다.

## 3. Generation 결과

| Study | Calls | Reason correct | Blocker metric | Joint correct |
|---|---:|---:|---:|---:|
| v17 single causal blocker | 1,500 | 657 (43.8%) | 1,109 (73.9%) | 596 (39.7%) |
| v18 all visible blockers | 1,500 | 699 (46.6%) | exact set 778 (51.9%) | 440 (29.3%) |
| v19 primary blocker | 1,500 | 514 (34.3%) | 1,123 (74.9%) | 512 (34.1%) |

- v17과 v19는 parse error, semantic validation failure, infrastructure error가 모두 0입니다.
- v18은 1,500개 응답이 모두 parse되었고, reason과 blocker list의 일관성 규칙에서 266개 semantic validation failure가 기록되었습니다. Blocker-set micro F1은 63.3%입니다.
- v18 scene GT는 empty set 4개와 singleton set 21개로 구성되어 있어, 여러 blocker를 동시에 빠짐없이 회수하는 능력을 직접 검증하는 dataset은 아닙니다.
- v19의 blocked scene마다 visible valid blocker가 하나이므로 여러 후보 사이의 primary ranking 능력은 이 scene set에서 분리해 검증되지 않습니다.

세부 결과는 각 study의 `results/`에 있습니다.

- [v17 scene results](studies/single-causal-blocker-v17/results/scene_results.json)
- [v18 summary](studies/all-visible-blockers-v18/results/summary.json)
- [v19 summary](studies/primary-blocker-v19/results/summary.json)
- [v19 detailed analysis](studies/primary-blocker-v19/results/v19_detailed_analysis.json)

## 4. v19 핵심 관찰

v19에서 C2-D8의 joint correct는 72/125 (57.6%)로 12개 조건 중 가장 높았습니다. C2-D4는 60/125 (48.0%)였으며, 두 조건의 blocker correct는 모두 116/125 (92.8%)였습니다.

전체적으로 blocker ID 식별(74.9%)이 reason label 분류(34.3%)보다 강했습니다. Reason confusion에서는 `CLEARANCE_OVERLAP`으로의 편향이 크게 나타났습니다.

| Ground truth reason | Calls | Correct | `CLEARANCE_OVERLAP` 예측 |
|---|---:|---:|---:|
| `OCCLUSION` | 420 | 26 | 332 |
| `BOTH` | 540 | 81 | 443 |
| `CLEARANCE_OVERLAP` | 300 | 266 | 266 |
| `NONE` | 240 | 141 | 92 |

따라서 v19 joint score의 주된 병목은 blocker ID보다 reason label입니다. 300개 scene–condition group 중 5개 seed의 exact output이 모두 같은 group은 190개, blocker ID가 모두 같은 group은 226개였습니다.

## 5. C2-D4 reason latent cosine 분석

[`studies/c2-d4-reason-latent-v1`](studies/c2-d4-reason-latent-v1/)은 v19의 C2-D4 조건을 대상으로 한 후속 분석입니다. 새 answer generation을 수행하지 않고, 25개 scene 각각에서 고정 assistant prefix `{"blocking_reason": "`의 마지막 token, 즉 첫 reason value token 직전 hidden state를 추출했습니다.

- 48개 decoder block 출력과 final RMSNorm 출력을 저장
- primary comparison: block 24 vs final norm
- 독립 분석 단위: seed가 아닌 scene 25개
- historical v19 C2-D4: reason 60/125, blocker 116/125

| Hypothesis group | Block 24 within | Block 24 between | Gap | Final within | Final between | Gap |
|---|---:|---:|---:|---:|---:|---:|
| `FC_CLEAR` family | 0.999321 | 0.998419 | 0.000902 | 0.990699 | 0.985283 | 0.005416 |
| Semantic `NONE` scenes C1–C4 | 0.999363 | 0.998448 | 0.000915 | 0.997376 | 0.984311 | 0.013065 |

Semantic `NONE`의 within-minus-between gap은 block 24에서 final norm으로 가며 0.012151 증가했습니다. 동일 family nearest-neighbor 비율은 64%에서 72%로 증가했고, layerwise separation은 후반부에서 가장 크게 나타났습니다(`FC_CLEAR`: block 40에서 0.01272, semantic `NONE`: block 42에서 0.03402).

이는 late-layer representation에서 clear/`NONE` 관련 분리가 더 뚜렷해졌다는 descriptive result입니다. 표본이 25 scenes이고 scene family, object identity, layout, prompt와 geometry payload가 함께 변하므로 인과적인 reasoning mechanism의 증거로 해석하지 않습니다. 또한 latent extraction의 Transformers BF16 경로와 historical generation의 vLLM 경로가 수치적으로 동일하다고 가정하지 않습니다.

아카이브에는 원본 [`hidden_states.npz`](studies/c2-d4-reason-latent-v1/hidden_states.npz), cosine matrix, pairwise CSV, 분석 summary, PNG/SVG figure를 함께 보존했습니다.

## 6. 보존한 실행 로그

이 snapshot에는 **4,500개의 `runs.jsonl` record와 4,500개의 대응 `attempts.jsonl` record**가 있습니다.

| Study | runs | attempts |
|---|---:|---:|
| single causal blocker v17 | 1,500 | 1,500 |
| all visible blockers v18 | 1,500 | 1,500 |
| primary blocker v19 | 1,500 | 1,500 |
| **Total** | **4,500** | **4,500** |

Latent follow-up은 25개의 derived scene record를 포함하지만 새 generation call은 없으므로 위 합계에는 더하지 않았습니다.

## 7. 디렉터리 구성과 재현

```text
numbered-rgb-vlm-action-geometry-l3/
  README.md
  SOURCES.md
  studies/                 # L3 config, prompt, schema, raw log, result
  reproduction/scripts/    # 실행·분석 코드 snapshot
  audit/archive_manifest.csv
```

- 실행·분석 코드와 재현 범위는 [`reproduction/README.md`](reproduction/README.md)에 있습니다.
- 원본 실험 경로와 포함·제외 범위는 [`SOURCES.md`](SOURCES.md)에 있습니다.
- 파일별 SHA-256, byte size, JSONL row count는 [`audit/archive_manifest.csv`](audit/archive_manifest.csv)에 있습니다.

전체 입력 이미지, geometry payload collection, Isaac Sim asset와 model checkpoint는 중복 용량 때문에 포함하지 않았습니다. raw log의 scene/image 경로와 hash, 각 study config와 manifest로 원본 입력을 추적할 수 있습니다.

## 8. 해석 경계

- Reason과 blocker GT는 scene geometry와 실험 규칙에서 정의한 label이며, 실제 robot manipulation execution 성공을 직접 측정한 값이 아닙니다.
- Geometry 조건 간 차이는 정보 representation에 대한 모델 반응이며 물리적 최적성이나 인과적 효과의 직접 증거가 아닙니다.
- 5개 seed는 sampling 안정성을 보여주지만 scene 다양성을 대신하지 않습니다.
- v18 결과를 multi-blocker recall 능력으로, v19 결과를 multi-candidate ranking 능력으로 일반화하지 않습니다.
- Latent cosine 분석은 representation의 기술적 비교이며 모델 내부 reasoning을 단독으로 입증하지 않습니다.
