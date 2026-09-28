# Experiments — 공개 VLM 능력 평가

[연구 개요](../README.md) · [관련 문헌](../literature/README.md)

공개 VLM의 object grounding, 공간관계, 회수 가능성, blocker 식별, 다음 행동과 파라미터 결정을 단계별로 평가한다. 현재 주 모델은 `Qwen/Qwen3-VL-30B-A3B-Instruct`다.

| 실험 | 상태 | 핵심 범위 |
|---|---|---|
| [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](numbered-rgb-vlm-basics-n0-n3/) | Complete | ID 인식, ID–물체 의미 연결, 자연어 target grounding, 공간관계 추론 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L2](numbered-rgb-vlm-action-geometry-l2/) | L2 snapshot complete | 직접 회수 가능성 판단과 geometry 단일정보 조건 비교 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L3](numbered-rgb-vlm-action-geometry-l3/) | L3 snapshot complete | 차단 원인·blocker 식별과 C2-D4 reason latent cosine 분석 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L4](numbered-rgb-vlm-action-geometry-l4/) | L4 snapshot complete | 다음 동작 선택, blocker 조건부 replay, action-only 비교 |

## 평가 범위

- N0–N3: Numbered RGB의 ID·의미·target grounding·공간관계.
- L2: 직접 회수 여부. 장면 설계 의도와의 일치율이며 실기 성공률이 아니다.
- L3: 차단 원인과 blocker 식별. v19 데이터의 후보 구성만으로 다중 후보 순위화를 입증하지 않는다.
- L4: 다음 한 동작. replay 유효성과 advisory action-label 일치율을 구분한다.
- 현재 기록은 정적 장면 기반 capability 평가다. F/T·tactile 융합 모델의 학습이나 폐루프 sub-goal 시퀀스 실행을 완료했다는 의미가 아니다.

## 저장 규칙

새 실험은 `experiments/experiment-slug/`에 추가합니다. 완료된 실험 폴더는 가능한 한 immutable snapshot으로 유지하고, 해석 수정이나 후속 분석은 원시 로그를 덮어쓰지 않고 별도 파일과 revision note로 남깁니다.

각 실험 폴더의 권장 구성은 다음과 같습니다.

```text
experiments/experiment-slug/
  README.md          # 연구 질문, 설계, 결과, 한계
  SOURCES.md         # 원본 경로와 외부 문서
  assets/            # 대표 이미지와 작은 그림
  config/            # frozen config와 scene manifest
  prompts/           # 실제 prompt
  schemas/           # structured-output schema
  logs/              # raw model responses와 실행 메타데이터
  results/           # 평가 결과와 표
  audit/             # completeness/integrity 검사
  reproduction/      # 실행·분석 코드 snapshot
```

새 기록을 만들 때는 [실험 기록 템플릿](../docs/experiment-template.md)을 사용할 수 있습니다. 실행 전에 입력·prompt·schema·seed를 먼저 고정합니다.

## 대용량 파일 정책

- JSONL, CSV, Markdown 등 검토 가능한 텍스트 로그를 우선 커밋합니다.
- 장면 전체 이미지, USD, checkpoint처럼 크고 중복되는 파일은 기본적으로 제외합니다.
- 제외한 입력은 원본 경로, scene ID, SHA-256 또는 manifest를 남겨 추적합니다.
- 단일 파일이 GitHub 일반 파일 제한에 가까워지면 Git LFS 또는 별도 release artifact로 이동합니다.
