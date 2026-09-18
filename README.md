# VLM-LAB

VLM 기반 로봇 조작 연구의 실험 설계, 고정 설정, 원시 추론 로그, 평가 결과를 누적하는 연구 로그 저장소입니다.

이 저장소의 목표는 결과 수치만 남기는 것이 아니라 다음 정보를 함께 보존하여 실험을 다시 추적할 수 있게 하는 것입니다.

- 어떤 연구 질문과 가설로 실험했는가
- 어떤 장면, 입력, prompt, schema, model 설정을 사용했는가
- 각 호출에서 모델이 무엇을 출력했는가
- 어떤 규칙과 지표로 채점했는가
- 결과로 무엇을 주장할 수 있고 무엇을 주장할 수 없는가

## Experiment index

| 실험 | 상태 | 핵심 범위 |
|---|---|---|
| [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](experiments/numbered-rgb-vlm-basics-n0-n3/) | Complete | ID 인식, ID–물체 의미 연결, 자연어 target grounding, 공간관계 추론 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L2](experiments/numbered-rgb-vlm-action-geometry-l2/) | L2 snapshot complete | 직접 회수 가능성 판단과 geometry 단일정보 조건 비교 |
| [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 — L3](experiments/numbered-rgb-vlm-action-geometry-l3/) | L3 snapshot complete | 차단 원인·blocker 식별과 C2-D4 reason latent cosine 분석 |

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

새 기록을 만들 때는 [실험 기록 템플릿](docs/experiment-template.md)을 사용할 수 있습니다.

## 대용량 파일 정책

- JSONL, CSV, Markdown 등 검토 가능한 텍스트 로그를 우선 커밋합니다.
- 장면 전체 이미지, USD, checkpoint처럼 크고 중복되는 파일은 기본적으로 제외합니다.
- 제외한 입력은 원본 경로, scene ID, SHA-256 또는 manifest를 남겨 추적합니다.
- 단일 파일이 GitHub 일반 파일 제한에 가까워지면 Git LFS 또는 별도 release artifact로 이동합니다.
