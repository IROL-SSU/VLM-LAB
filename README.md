# VLM-LAB

**접촉 센서 정보를 활용한 VLM 기반 parameterized sub-goal 생성 연구**

Clutter한 선반에서 target 물체를 회수하려면 주변 방해물을 이동하고, 작업 진행에 따라 다음 행동을 결정해야 합니다. 이 과정에서 로봇 팔과 그리퍼가 작업 부위를 가려 시각 관측만으로 접촉 상태와 진행 상황을 판단하기 어려워질 수 있습니다.

본 연구는 **시각·언어와 F/T·tactile 정보를 함께 활용하여, 가려짐 상황에서도 실행 가능한 sub-goal을 생성·수정하는 VLM을 개발**하는 것을 목표로 합니다. 로봇 상태(proprioception), 이전 관측과 현재 행동의 맥락을 함께 고려하는 표현 및 학습 방법을 탐색합니다. 기존 공개 모델을 기반으로 한 확장·학습 방식을 검토하며, 최종 구조와 학습 방법은 연구 중입니다.

## 연구 질문

- 공개 VLM은 물체 식별부터 행동 종류·대상·방향·거리 결정까지 어디에서 한계를 보이는가?
- 추가 정보의 종류와 수치형·범주형 표현에 따라 판단이 어떻게 달라지는가?
- 통계적·의미적 성격이 다른 Vision–Language와 F/T·tactile을 어디서, 어떻게 연결·정렬·융합할 것인가?
- 가려지기 전의 관측, 접촉 신호와 행동 이력을 이용해 sub-goal을 어떻게 유지·수정할 것인가?

모달리티 연결 연산뿐 아니라 표현 학습 목표, 시간적 대응, 센서의 유효성, 실제 의사결정 기여를 함께 살펴봅니다. Concatenation의 사용 여부만으로 융합의 유효성을 판단하지 않습니다.

## 세 가지 연구 축

| 경로 | 역할 | 현재 내용 |
|---|---|---|
| [experiments/](experiments/README.md) | 기존 VLM의 능력·한계와 추가 정보의 효과를 실험으로 파악 | Numbered RGB N0–N3, Action-Level × Geometry L2–L4 |
| [literature/](literature/README.md) | 모델 구조와 학습 목표를 설계하기 위한 문헌 조사 | Multimodal Context Fusion / Alignment 비교표와 논문별 상세 정리 |
| [research/](research/multimodal-online-subgoal-adaptation/) | 최종 시스템 목표와 단계별 검증 관계를 관리 | Multimodal online adaptation과 Grounded RGB-D TRANSLATE 중간 목표 |

실험은 모델이 어려워하는 판단과 필요한 정보를 찾고, 문헌 조사는 그 정보를 표현·연결·학습하는 방법의 근거를 제공합니다. 연구 로드맵은 두 결과를 최종 multimodal state estimation 및 online sub-goal adaptation의 설계·검증 단계로 연결합니다.

## 목표 출력과 현재 진행 상태

목표 출력은 행동 종류, 대상, 방향·거리 또는 회전각 등을 포함하는 sub-goal입니다. 예를 들어 `TRANSLATE(object=D, direction=LEFT, distance=0.12 m)`처럼 표현할 수 있습니다. 이는 개념 예시이며 확정된 공통 출력 schema는 아닙니다. 실제 실행에는 좌표계와 제약 정의가 필요합니다.

- **진행한 실험:** 공개 VLM의 N0–N3 기초 능력과 L2–L4 조작 판단 평가. 세부 상태·수치는 각 실험 문서에 기록합니다.
- **진행 중인 조사:** 이종 모달리티의 연결 위치, 정렬 목표, 융합 구조와 접촉 정보 활용을 비교합니다.
- **연구 목표:** 접촉 센서 입력을 이용한 sub-goal 생성·갱신 모델 개발과 가려짐 상황의 폐루프 평가. 현재 저장소가 이 목표를 달성한 구현을 제공하는 것은 아닙니다.

## 실험 목록

| 실험 | 상태 | 핵심 범위 |
|---|---|---|
| [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](experiments/numbered-rgb-vlm-basics-n0-n3/) | Complete | ID 인식, ID–물체 의미 연결, 자연어 target grounding, 공간관계 추론 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L2](experiments/numbered-rgb-vlm-action-geometry-l2/) | L2 snapshot complete | 직접 회수 가능성 판단과 geometry 단일정보 조건 비교 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L3](experiments/numbered-rgb-vlm-action-geometry-l3/) | L3 snapshot complete | 차단 원인·blocker 식별과 C2-D4 reason latent cosine 분석 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L4](experiments/numbered-rgb-vlm-action-geometry-l4/) | L4 snapshot complete | 다음 동작 선택, blocker 조건부 replay, action-only 비교 |

L2의 장면 설계 일치율, L3의 label 정확도, L4의 시뮬레이션 replay 유효성은 서로 다른 지표입니다. 이를 실기 성공률이나 전체 sub-goal 시퀀스 성공률로 통합하지 않습니다.

## 문헌 조사

[**Multimodal Context Fusion / Alignment**](literature/multimodal-context-fusion-alignment/README.md)

ForceVLA, Adaptive Vision–Torque Fusion, FuSe, ViTaS, TacFiLM, TA-VLA의 모달리티·융합 위치·정렬 학습·실험 근거·한계를 비교합니다. 논문 결과와 본 연구에 적용하기 위한 가설을 구분합니다.

## 기록 원칙

- 실험별 prompt, schema, 설정, 원시 응답, 채점 규칙과 결과를 함께 보존합니다.
- 완료된 실험은 snapshot으로 관리하며 후속 해석은 별도 revision으로 기록합니다.
- 논문별 원문과 노션 출처, 정리 기준일, 연구적 해석을 남깁니다.
- 원문 그림·표와 해설 도식을 구분합니다. PDF와 대용량 자산은 출처 링크·manifest를 우선 사용합니다.

[최종 연구 로드맵](research/multimodal-online-subgoal-adaptation/) · [실험 기록 안내](experiments/README.md) · [실험 템플릿](docs/experiment-template.md) · [문헌 기록 안내](literature/README.md)
