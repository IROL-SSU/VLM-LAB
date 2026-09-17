# Sources and provenance

이 폴더는 `/home/ssu/ShelfScene`에서 수행된 완료 실험의 archival snapshot입니다. 원본 파일은 수정하지 않았으며, GitHub에는 재현과 검토에 필요한 텍스트 artifact와 대표 이미지 위주로 복사했습니다.

## Notion 문서

- [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](https://app.notion.com/p/3c8952c9e273816dba3ff2fdebb84660)
- [N0 Visible ID Recognition — Sparse-ID TEXT vs JSON 실험 결과](https://app.notion.com/p/3c8952c9e2738117a7bfe3688a19885d)
- [N1 ID–Object–Color Mapping — Explicit visible_ids Enumeration 실험 결과](https://app.notion.com/p/3c9952c9e273811fad98daa7c6fa379f)
- [N2 Natural-Language Target Grounding — Explicit Constraint Scaffold 실험 결과](https://app.notion.com/p/3cf952c9e27381a4909ec6dd4b193f28)
- [N2-Basic Target Grounding — Unified Coarse-Color Enumeration 실험 결과](https://app.notion.com/p/3d0952c9e27381df8b43e8d244f89390)
- [N3 Depth Reasoning — Scene 구성 및 생성 규칙](https://app.notion.com/p/3c9952c9e2738135842bdbc947f859df)
- [N3 Spatial-Relation Reasoning — 좌우·앞뒤·Depth 입력·카메라 시점 종합 실험 결과](https://app.notion.com/p/3cf952c9e27381ed866ad846af8b084f)

## Snapshot mapping

| Archive path | Original path under `/home/ssu/ShelfScene` | 보존 범위 |
|---|---|---|
| `n0/sparse-id-text-vs-json` | `experiments/b0_n0_sparse_ids_text_vs_json_60scenes_20260826` | config, prompts, raw logs, reports, evaluated tables, audit |
| `n1/explicit-visible-ids-ablation` | `experiments/b0_n1_explicit_visible_ids_ablation_60scenes_20260827` | config, prompts, raw logs, figures, reports, evaluated tables, audit |
| `n2/explicit-constraint-scaffold` | `experiments/b0_n2_explicit_constraint_scaffold_60scenes_20260828` | config, prompts, raw logs, reports, evaluated tables, audit |
| `n2/unified-coarse-color` | `experiments/b0_n2_basic_unified_coarse_60scenes_20260903` | config, prompts, raw logs, reports, evaluated tables, audit |
| `n3/cross-task-75-scenes` | `experiments/n3_initial75_cross_tasks_75scenes_5seeds_20260906` | config, prompts, 375 new raw calls, reports, evaluated tables, audit |
| `n3/occlusion-prompt-comparison` | `experiments/n3_occlusion_rgb_missing20_5methods_5seeds_20260907` | config, prompts, 500 raw calls, reports, evaluated tables, audit |
| `n3/depth-input-ablation` | `experiments/n3_depth_input_ablation_3conditions_full40_20260907` | source manifest, aggregate results, report, audit |
| `n3/tilted-camera-input-ablation` | `experiments/n3_tilted24_height90_input_ablation_3conditions_20260907` | source manifest, 7,200 scored records, aggregate results, report, audit |
| `n3/camera-view-effect` | `experiments/n3_camera_view_effect_4_16_24_filtered_20260907` | comparison CSV와 figures |

## Deliberately omitted

- 전체 numbered RGB image collection
- Isaac Sim USD scenes
- model checkpoint와 runtime cache
- 하위 실험에 중복된 scene image
- `evaluated_records.jsonl`이 raw `logs/runs.jsonl`에서 재생성 가능하고 매우 큰 경우의 중복 사본

하위 config의 `frozen_scenes.json`, raw log의 `image_path`와 `image_sha256`, source experiment manifest를 이용해 원본 입력을 추적할 수 있습니다. 이 snapshot의 두 대표 이미지는 원본을 수정하지 않고 복사했습니다.

## Interpretation boundary

N3 종합 표 중 일부는 여러 source experiment에서 파생된 비교 결과입니다. aggregate 폴더만 보고 모든 source raw call이 이 Git snapshot에 포함됐다고 가정하면 안 됩니다. 포함된 call-level record의 정확한 범위는 상위 `README.md`의 로그 표를 기준으로 합니다.
