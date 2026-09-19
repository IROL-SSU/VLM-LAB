# Sources and provenance

이 폴더는 `/home/ssu/ShelfScene`의 Numbered RGB 기반 Action-Level × Geometry 단일정보 프로그램에서 수행한 L4 v24–v26의 archival snapshot입니다. 원본 artifact는 변경하지 않았습니다.

## Documents

- [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 실험](https://app.notion.com/p/3dc952c9e273815db294c6997f868773)
- [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](https://app.notion.com/p/3c8952c9e273816dba3ff2fdebb84660)

## Snapshot mapping

| Archive path under `studies/` | Original experiment directory under `/home/ssu/ShelfScene/experiments/` | Inference | Replay records |
|---|---|---:|---:|
| `full-action-v24` | `vlm_action_geometry_single_info_l4_full_from_v19_v24` | 1,750 | 1,750 |
| `blocker-conditioned-v25` | `vlm_action_geometry_single_info_l4_blocker_conditioned_v25` | 1,750 | 1,750 |
| `action-only-v26` | `vlm_action_geometry_single_info_l4_action_only_v26` | 1,750 | 0 |

세 study의 frozen scene별 Numbered RGB SHA-256은 L3 v19 source와 일치합니다. v24는 L3 v19의 scene 입력을, v25와 v26은 동일 scene–condition–seed의 frozen L3 v19 출력을 후속 단계 입력으로 사용합니다.

## Included

- 각 study의 **모든** `logs/` 파일: raw inference/attempt, runtime/progress/invocation, v24·v25 replay/replay-attempt 기록
- 전체 `config/` 파일: experiment config, frozen scene manifest, run table
- 전체 `audit/` 파일: preflight/final audit, frozen/source hash와 검증에 사용한 mask
- 전체 `results/`의 JSON, CSV, Markdown, PNG/SVG 분석물 및 후속 failure analysis
- v24·v25의 `simulator_replay/cache/`와 `post_masks/` 산출물
- 실제 L4 prompt/schema와 source project의 실행·replay·집계·시각화 script snapshot

## Deliberately omitted

- 세 study에서 반복되는 25개 전체 Numbered RGB image collection, scene geometry, geometry payload collection과 ID mapping 원본
- model checkpoint, Isaac Sim USD scene, simulator 설치 파일과 runtime cache
- source 결과와 중복되는 ZIP bundle

이미지·geometry 입력의 relative path와 SHA-256은 각 run table, raw response record와 frozen scene manifest에 남아 있습니다. 원본 입력은 `/home/ssu/ShelfScene/experiments/`의 위 경로에서 추적할 수 있습니다. 이 Git snapshot만으로 generation이나 simulator replay를 다시 실행하려면 source image/geometry/scene asset, model checkpoint와 Isaac Sim runtime이 별도로 필요합니다. 저장된 원시 결과의 집계와 파일 무결성은 snapshot에서 검산할 수 있습니다.

## Integrity

[`audit/archive_manifest.csv`](audit/archive_manifest.csv)는 archive의 모든 파일에 대해 relative path, byte size, SHA-256과 JSONL row count를 기록합니다. manifest 자체는 자기참조 hash 문제를 피하기 위해 목록에서 제외합니다.
