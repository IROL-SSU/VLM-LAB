# L2 instance occlusion left-to-right v9

## 실험 정의

보이는 번호 물체를 visible-mask 중심의 x좌표로 왼쪽부터 나열하고, target을 제외한 각 instance가 target을 시각적으로 가리는지만 Boolean으로 평가한다.

프롬프트에는 수치 임계값을 노출하지 않았다. GT는 후보 instance가 target-only silhouette의 5% 이상을 가리면 양성이다.

## 무결성

- 완료: 1500/1500
- 파싱 오류 / 후검증 실패 / 출력 불변식 실패: 0 / 0 / 0
- 인프라 오류: 0

## 전체 결과

| 평가 단위 | Accuracy | BA | Occlusion recall | Non-occlusion specificity | Precision | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|---:|
| 후보 instance | 85.6% | 57.9% | 16.1% | 99.7% | 92.4% | 145/755/4428/12 |
| 장면 내 하나라도 가림 | 49.3% | 57.6% | 16.1% | 99.2% | 96.7% | 145/755/595/5 |

- 전체 출력 완전일치: 737/1500 (49.1%)
- Occluder ID 집합 완전일치: 737/1500 (49.1%)
- 왼쪽→오른쪽 배열 및 target ID 정확도: 100.0% / 100.0%

## 조건별 결과

| 조건 | 후보 정확도 | BA | 가림 recall | 비가림 specificity | 전체 출력 완전일치 | 장면 Boolean 정확도 |
|---|---:|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 86.1% | 58.7% | 17.3% | 100.0% | 50.4% | 50.4% |
| C1  R Numeric | 84.3% | 53.3% | 6.7% | 100.0% | 44.0% | 44.0% |
| C2 D4 R Qualitative D4 | 84.7% | 54.7% | 9.3% | 100.0% | 45.6% | 45.6% |
| C2 D8 R Qualitative D8 | 85.2% | 56.0% | 12.0% | 100.0% | 47.2% | 47.2% |
| C3 D4 F Numeric D4 | 84.5% | 55.1% | 10.7% | 99.5% | 45.6% | 46.4% |
| C3 D8 F Numeric D8 | 86.3% | 59.3% | 18.7% | 100.0% | 51.2% | 51.2% |
| C4 D4 F Qualitative D4 | 85.2% | 56.5% | 13.3% | 99.7% | 47.2% | 48.0% |
| C4 D8 F Qualitative D8 | 85.6% | 57.3% | 14.7% | 100.0% | 48.8% | 48.8% |
| C5 D4 M Numeric D4 | 87.4% | 62.7% | 25.3% | 100.0% | 55.2% | 55.2% |
| C5 D8 M Numeric D8 | 85.6% | 59.5% | 20.0% | 98.9% | 48.8% | 48.8% |
| C6 D4 M Qualitative D4 | 86.5% | 60.5% | 21.3% | 99.7% | 52.0% | 52.0% |
| C6 D8 M Qualitative D8 | 86.3% | 61.5% | 24.0% | 98.9% | 53.6% | 54.4% |

## 장면 종류별 결과

| 장면 종류 | 후보 정확도 | BA | 가림 recall | 비가림 specificity | 전체 출력 완전일치 | 장면 Boolean 정확도 |
|---|---:|---:|---:|---:|---:|---:|
| TRANSLATE | 70.7% | 59.0% | 18.0% | 100.0% | 18.0% | 18.0% |
| ROTATE | 83.1% | 64.8% | 30.0% | 99.7% | 29.3% | 30.0% |
| LIFT_AND_RELOCATE | 79.0% | 50.0% | 0.3% | 99.6% | 0.0% | 0.3% |
| FC_CLEAR | 100.0% | N/A | N/A | 100.0% | 100.0% | 100.0% |
| FC_BLOCKED | 99.4% | N/A | N/A | 99.4% | 98.3% | 98.3% |

## 이전 formulation과 비교

### 후보 instance 단위

| 방식 | Accuracy | BA | Recall | Specificity | Precision | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|---:|
| v9 visual occlusion LTR | 85.6% | 57.9% | 16.1% | 99.7% | 92.4% | 145/755/4428/12 |
| v8 grasp interference | 61.8% | 63.3% | 65.6% | 61.0% | 25.4% | 590/310/2709/1731 |

### 장면 Boolean 단위

| 방식 | Accuracy | BA | Recall | Specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| v9 visual occlusion LTR | 49.3% | 57.6% | 16.1% | 99.2% | 145/755/595/5 |
| v8 grasp interference | 55.2% | 52.2% | 67.3% | 37.0% | 606/294/222/378 |
| v7 single Boolean | 52.9% | 60.5% | 22.3% | 98.7% | 201/699/592/8 |
| v5 direct occlusion Boolean | 63.7% | 69.8% | 39.6% | 100.0% | 356/544/600/0 |
| original L1 non-FULL | 70.9% | 75.7% | 51.4% | 100.0% | 463/437/600/0 |

## 대표 양성 instance

- 최저 검출: `scene_lift_and_relocate_v01`의 object 64는 GT overlap ratio 0.606이지만 60회 중 0회만 true였다.
- 최고 검출: `scene_rotate_v02`의 object 50는 GT overlap ratio 0.341이고 60회 중 60회 true였다.

오답 candidate 행 767개는 `candidate_instance_errors.csv`에 모두 기록했다.
