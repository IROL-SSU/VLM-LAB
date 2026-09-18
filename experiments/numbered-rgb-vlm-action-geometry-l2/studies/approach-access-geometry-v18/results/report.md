# L2 접근 경로 판단 × Geometry 단일정보 — v18 결과

25장면 × 12조건 × 5 seed = 1,500회. 직전 합의한 짧은 프롬프트로 선반 입구에서 타깃까지의 접근 경로만 질문했습니다. v16과 동일한 이미지·Geometry·모델·샘플링 설정·seed 조합을 사용하고 system/user 프롬프트 및 출력 라벨을 변경했습니다.

접근 가능성 정답(GT)이나 실제 로봇 실행 결과는 없습니다. `FC_BLOCKED` 등은 기존 장면 family 이름이며 접근 차단 정답으로 사용하지 않았습니다. 아래 수치는 모델의 응답 비율입니다. 특히 v16의 `RETRIEVE_NOW`와 v18의 `ACCESSIBLE`은 질문의 의미가 달라, 대응시킨 비율 차이는 정확도·회수 성공률 개선을 뜻하지 않습니다.

## 주요 관찰

- FC_BLOCKED 전체 응답: v16 RETRIEVE_NOW 186/300 (62.0%) → v18 ACCESSIBLE 274/300 (91.3%), +29.3%p. 12조건을 합산한 기술 통계이며 독립 장면 수가 늘어난 것은 아닙니다.
- FC_BLOCKED / C0: 19/25 → 25/25 (+24.0%p). 같은 장면·seed의 v16 REARRANGE_FIRST→v18 ACCESSIBLE 6회, v16 RETRIEVE_NOW→v18 BLOCKED 0회.
- FC_BLOCKED / C2_D8: 7/25 → 13/25 (+24.0%p). 같은 장면·seed의 v16 REARRANGE_FIRST→v18 ACCESSIBLE 7회, v16 RETRIEVE_NOW→v18 BLOCKED 1회.
- FC_BLOCKED 외 변화가 큰 조합: C4_D8 / TRANSLATE 2/25→20/25 (+72.0%p); C6_D8 / TRANSLATE 2/25→19/25 (+68.0%p); C4_D8 / ROTATE 0/25→16/25 (+64.0%p); C6_D4 / ROTATE 0/25→15/25 (+60.0%p); C3_D8 / LIFT_AND_RELOCATE 1/25→14/25 (+52.0%p).

## 실험 및 무결성

- 모델: `Qwen/Qwen3-VL-30B-A3B-Instruct`; seed: 28101, 28102, 28103, 28104, 28105.
- 완료 1500/1,500; JSON schema 유효 1500/1,500; 인프라 오류 0회.
- 생성 시간 합계: 113.0초. 모델 적재 및 분석 시간은 제외합니다.
- 1,500개의 run ID 및 장면·조건·seed 조합이 누락·중복 없이 일치하며, 모든 조건은 125회, family×조건은 25회입니다.
- 이미지/Geometry SHA256, target ID, model ID, sampling 설정을 v16 로그와 대조했습니다. 응답 비율 분모에는 전체 실행 수를 사용하고 잘못된 응답은 별도 집계합니다.

## 프롬프트

System:

```text
Assess whether the target is accessible from the shelf opening.
Numbered badges identify objects.
Return only a JSON object. Do not include explanations.
```

User:

```text
Target object ID: {target_object_id}

Geometry information:
{geometry_json}

Is there a clear approach path from the shelf opening to the target
without moving or contacting other objects?

Choose:
- ACCESSIBLE: there is a clear approach path.
- BLOCKED: other objects block the approach.

Return exactly one JSON object with the field "decision".
```

C0의 Geometry는 `null`이며, 나머지 조건은 각 조건의 기존 단일정보 Geometry JSON입니다.

## 조건별 응답

| 조건 | 정보 | ACCESSIBLE /125 | 비율 | BLOCKED | INVALID | 5 seed 전원 일치 /25 | C0 대비 B→A | C0 대비 A→B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | RGB only | 50 | 40.0% | 75 | 0 | 25 | 0 | 0 |
| C1 | R Numeric | 56 | 44.8% | 69 | 0 | 18 | 11 | 5 |
| C2_D4 | R Qualitative D4 | 53 | 42.4% | 72 | 0 | 23 | 12 | 9 |
| C2_D8 | R Qualitative D8 | 45 | 36.0% | 80 | 0 | 21 | 7 | 12 |
| C3_D4 | F Numeric D4 | 80 | 64.0% | 45 | 0 | 19 | 30 | 0 |
| C3_D8 | F Numeric D8 | 81 | 64.8% | 44 | 0 | 19 | 31 | 0 |
| C4_D4 | F Qualitative D4 | 71 | 56.8% | 54 | 0 | 20 | 21 | 0 |
| C4_D8 | F Qualitative D8 | 97 | 77.6% | 28 | 0 | 22 | 47 | 0 |
| C5_D4 | M Numeric D4 | 72 | 57.6% | 53 | 0 | 20 | 22 | 0 |
| C5_D8 | M Numeric D8 | 88 | 70.4% | 37 | 0 | 18 | 38 | 0 |
| C6_D4 | M Qualitative D4 | 81 | 64.8% | 44 | 0 | 19 | 31 | 0 |
| C6_D8 | M Qualitative D8 | 81 | 64.8% | 44 | 0 | 21 | 31 | 0 |

B→A는 같은 v18의 C0 BLOCKED→해당 조건 ACCESSIBLE, A→B는 반대 방향입니다. seed 전원 일치는 응답의 안정성을 나타내며 정답 여부를 나타내지 않습니다.

## Family별 ACCESSIBLE 횟수 /25

| 조건 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT_AND_RELOCATE |
|---|---:|---:|---:|---:|---:|
| C0 | 25 | 25 | 0 | 0 | 0 |
| C1 | 25 | 20 | 1 | 6 | 4 |
| C2_D4 | 25 | 16 | 0 | 12 | 0 |
| C2_D8 | 25 | 13 | 1 | 6 | 0 |
| C3_D4 | 25 | 25 | 12 | 11 | 7 |
| C3_D8 | 25 | 25 | 12 | 5 | 14 |
| C4_D4 | 25 | 25 | 12 | 6 | 3 |
| C4_D8 | 25 | 25 | 20 | 16 | 11 |
| C5_D4 | 25 | 25 | 15 | 5 | 2 |
| C5_D8 | 25 | 25 | 11 | 17 | 10 |
| C6_D4 | 25 | 25 | 11 | 15 | 5 |
| C6_D8 | 25 | 25 | 19 | 9 | 3 |

## v16→v18 응답 비율 비교

각 칸은 v16 RETRIEVE_NOW → v18 ACCESSIBLE 횟수(/25), 괄호는 비율 차이(%p)입니다. 서로 다른 질문의 응답 분포를 비교합니다.

| 조건 | FC_CLEAR | FC_BLOCKED | TRANSLATE | ROTATE | LIFT_AND_RELOCATE |
|---|---:|---:|---:|---:|---:|
| C0 | 25→25 (+0) | 19→25 (+24) | 2→0 (-8) | 1→0 (-4) | 0→0 (+0) |
| C1 | 20→25 (+20) | 13→20 (+28) | 5→1 (-16) | 1→6 (+20) | 0→4 (+16) |
| C2_D4 | 25→25 (+0) | 16→16 (+0) | 0→0 (+0) | 0→12 (+48) | 0→0 (+0) |
| C2_D8 | 24→25 (+4) | 7→13 (+24) | 0→1 (+4) | 0→6 (+24) | 0→0 (+0) |
| C3_D4 | 22→25 (+12) | 22→25 (+12) | 4→12 (+32) | 4→11 (+28) | 5→7 (+8) |
| C3_D8 | 25→25 (+0) | 21→25 (+16) | 5→12 (+28) | 4→5 (+4) | 1→14 (+52) |
| C4_D4 | 16→25 (+36) | 13→25 (+48) | 2→12 (+40) | 2→6 (+16) | 0→3 (+12) |
| C4_D8 | 17→25 (+32) | 14→25 (+44) | 2→20 (+72) | 0→16 (+64) | 0→11 (+44) |
| C5_D4 | 18→25 (+28) | 8→25 (+68) | 4→15 (+44) | 6→5 (-4) | 2→2 (+0) |
| C5_D8 | 23→25 (+8) | 19→25 (+24) | 3→11 (+32) | 4→17 (+52) | 0→10 (+40) |
| C6_D4 | 19→25 (+24) | 13→25 (+48) | 4→11 (+28) | 0→15 (+60) | 0→5 (+20) |
| C6_D8 | 22→25 (+12) | 21→25 (+16) | 2→19 (+68) | 0→9 (+36) | 0→3 (+12) |

## 장면별 ACCESSIBLE 횟수 /5

| 장면 | target | C0 | C1 | C2_D4 | C2_D8 | C3_D4 | C3_D8 | C4_D4 | C4_D8 | C5_D4 | C5_D8 | C6_D4 | C6_D8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_fc_blocked_v01 | 64 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_blocked_v02 | 53 | 5 | 5 | 1 | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_blocked_v03 | 12 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_blocked_v04 | 89 | 5 | 4 | 5 | 1 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_blocked_v05 | 59 | 5 | 1 | 0 | 0 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_clear_v01 | 53 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_clear_v02 | 29 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_clear_v03 | 95 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_clear_v04 | 61 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_fc_clear_v05 | 62 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_lift_and_relocate_v01 | 59 | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 1 | 1 | 4 | 1 | 2 |
| scene_lift_and_relocate_v02 | 70 | 0 | 4 | 0 | 0 | 5 | 5 | 1 | 5 | 1 | 5 | 4 | 1 |
| scene_lift_and_relocate_v03 | 12 | 0 | 0 | 0 | 0 | 1 | 5 | 2 | 5 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v04 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene_lift_and_relocate_v05 | 43 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| scene_rotate_v01 | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 4 | 1 | 4 | 4 | 0 |
| scene_rotate_v02 | 38 | 0 | 4 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 2 | 0 |
| scene_rotate_v03 | 83 | 0 | 1 | 5 | 1 | 5 | 1 | 0 | 5 | 2 | 5 | 5 | 5 |
| scene_rotate_v04 | 47 | 0 | 1 | 2 | 5 | 2 | 2 | 3 | 5 | 2 | 4 | 4 | 4 |
| scene_rotate_v05 | 44 | 0 | 0 | 0 | 0 | 2 | 2 | 3 | 2 | 0 | 0 | 0 | 0 |
| scene_translate_v01 | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| scene_translate_v02 | 45 | 0 | 0 | 0 | 0 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 5 |
| scene_translate_v03 | 18 | 0 | 1 | 0 | 0 | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| scene_translate_v04 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| scene_translate_v05 | 54 | 0 | 0 | 0 | 1 | 5 | 2 | 2 | 5 | 5 | 2 | 1 | 5 |

## 시각화

![조건·family별 ACCESSIBLE 비율. 각 칸은 25회이며 정확도를 뜻하지 않습니다.](plots/01_accessible_by_family_condition.png)

조건·family별 ACCESSIBLE 비율. 각 칸은 25회이며 정확도를 뜻하지 않습니다.

![FC_BLOCKED 장면별 ACCESSIBLE 횟수(/5). 한 장면 안에서 seed 간 응답이 달라지는지도 확인할 수 있습니다.](plots/02_fc_blocked_per_scene.png)

FC_BLOCKED 장면별 ACCESSIBLE 횟수(/5). 한 장면 안에서 seed 간 응답이 달라지는지도 확인할 수 있습니다.

![C0 및 C2 D8의 v16 RETRIEVE_NOW ↔ v18 ACCESSIBLE 비율 비교. 질문과 라벨의 의미가 달라 정확도 향상으로 해석하지 않습니다.](plots/03_v16_v18_c0_c2d8.png)

C0 및 C2 D8의 v16 RETRIEVE_NOW ↔ v18 ACCESSIBLE 비율 비교. 질문과 라벨의 의미가 달라 정확도 향상으로 해석하지 않습니다.

![모든 조건에서 v16→v18 응답 비율 변화(%p). FC_BLOCKED 외 family로 변화가 확산됐는지 확인합니다.](plots/04_v16_v18_family_delta.png)

모든 조건에서 v16→v18 응답 비율 변화(%p). FC_BLOCKED 외 family로 변화가 확산됐는지 확인합니다.


## 해석 범위

이 프롬프트에는 접근 주체의 부피, 접근 방향/궤적, 그리퍼 크기가 지정돼 있지 않습니다. 모델의 ACCESSIBLE 응답은 해당 이미지와 Geometry를 바탕으로 한 접근 경로 판단이며, 파지·타깃 제거 가능성이나 실제 무충돌 경로의 검증은 아닙니다.

v16과 비교할 때 질문을 회수에서 접근으로 바꾸면서 문장 길이, 판단 보조 문구, 출력 라벨도 함께 바뀌었습니다. 관찰된 차이를 특정 문구 하나의 효과로 분리할 수 없습니다. 5회 seed 반복은 같은 장면의 반복 측정이므로 독립적인 1,500장면 실험으로 해석하지 않습니다.

원자료: `raw_scene_seed_decisions.csv`, `scene_results.json`, `paired_vs_c0.csv`, `paired_vs_v16.csv`. 조건·family 요약: `condition_results.csv`, `family_results.csv`, `v16_v18_family_comparison.csv`. 각 행의 실제 seed와 결정을 원자료에서 확인할 수 있습니다.
