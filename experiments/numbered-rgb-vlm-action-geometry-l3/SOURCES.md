# Sources and provenance

이 폴더는 `/home/ssu/ShelfScene`에서 수행한 Numbered RGB 기반 Action-Level × Geometry 단일정보 프로그램의 L3 archival snapshot입니다. 원본 artifact는 변경하지 않았으며, canonical L3 generation 로그와 C2-D4 reason latent 분석 artifact를 복사했습니다.

## Documents

- [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 실험](https://app.notion.com/p/3dc952c9e273815db294c6997f868773)
- [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](https://app.notion.com/p/3c8952c9e273816dba3ff2fdebb84660)

## Snapshot mapping

| Archive path under `studies/` | Original path under `/home/ssu/ShelfScene/experiments/` | Generation calls |
|---|---|---:|
| `single-causal-blocker-v17` | `vlm_action_geometry_single_info_l3_maskfix_v17` | 1,500 |
| `all-visible-blockers-v18` | `vlm_action_geometry_single_info_l3_all_blockers_v18` | 1,500 |
| `primary-blocker-v19` | `vlm_action_geometry_single_info_l3_primary_blocker_v19` | 1,500 |
| `c2-d4-reason-latent-v1` | `l3_v19_c2_d4_reason_latent_20260918` | 0; 25 derived scene records |

## Included

- v17–v19의 raw `runs.jsonl`과 `attempts.jsonl` 전부
- 실행 console/runtime/progress log
- 각 study의 L3-specific config, frozen scene manifest, prompt와 structured-output schema
- 검토 가능한 JSON, CSV, Markdown result
- latent extraction manifest, 25 scene records, raw hidden state NPZ, cosine matrix NPZ, pairwise cosine CSV와 artifact hash
- latent analysis report와 PNG/SVG figure
- 실행·분석 Python script snapshot

`hidden_states.npz`의 SHA-256은 `fef78c125e2db1017b180d71a48e9343b3346a4df6acf1bf51869285aa92fc43`이며, 같은 값이 latent study의 `artifact_hashes.json`에 기록되어 있습니다.

## Deliberately omitted

- 전체 Numbered RGB image collection
- 전체 geometry payload collection과 반복 생성 가능한 `config/run_table.jsonl`
- 각 L3 실행에 중복되는 상위 scheduler 전체 config와 L3 외 prompt/schema
- Isaac Sim USD scene, simulator asset, model checkpoint와 runtime cache
- source experiment별 중복 ZIP archive
- v17의 현재 canonical snapshot 범위를 벗어난 비교용 summary, report와 condition table

제외된 입력은 raw record의 source path와 hash, study config와 frozen scene manifest로 추적합니다. 이 Git snapshot만으로 model generation을 다시 실행하려면 동일한 source image/geometry collection과 model runtime이 별도로 필요합니다. 포함된 원시 응답의 재분석, 집계 검증과 latent 결과 검산에는 snapshot의 로그·artifact·reproduction script를 사용할 수 있습니다.

## Integrity

[`audit/archive_manifest.csv`](audit/archive_manifest.csv)는 snapshot 파일별 relative path, byte size, SHA-256과 JSONL row count를 기록합니다. manifest 자체는 자기참조 hash 문제를 피하기 위해 목록에서 제외합니다.
