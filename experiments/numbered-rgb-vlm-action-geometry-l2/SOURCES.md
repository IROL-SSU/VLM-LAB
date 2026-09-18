# Sources and provenance

이 폴더는 `/home/ssu/ShelfScene`에서 수행한 Numbered RGB 기반 Action-Level × Geometry 단일정보 프로그램의 L2 archival snapshot입니다. 원본 artifact는 수정하지 않았으며, canonical L2 실행 로그와 검토·재현에 필요한 텍스트 파일을 복사했습니다.

## Documents

- [Numbered RGB 기반 VLM Action-Level × Geometry 단일정보 실험](https://app.notion.com/p/3dc952c9e273815db294c6997f868773)
- [Numbered RGB 기반 VLM 기초 능력 평가 — N0·N1·N2·N3](https://app.notion.com/p/3c8952c9e273816dba3ff2fdebb84660)
- [L2 Direct Retrieval × Geometry 단일정보 실험 — v16](https://app.notion.com/p/3de952c9e27381e5969cdef5d80d54d2)

## Snapshot mapping

| Archive path under `studies/` | Original experiment directory under `/home/ssu/ShelfScene/experiments/` | Runs |
|---|---|---:|
| `formal-predicate-v2` | `vlm_action_geometry_single_info_maskfix_validation` (`logs/l2_formal_v2`) | 1,500 |
| `grasp-zone-v3` | `vlm_action_geometry_single_info_l2_grasp_zone_v3` | 1,500 |
| `visibility-only-v4` | `vlm_action_geometry_single_info_l2_visibility_only_v4` | 1,500 |
| `occlusion-boolean-v5` | `vlm_action_geometry_single_info_l2_occlusion_boolean_v5` | 1,500 |
| `target-access-v6` | `vlm_action_geometry_single_info_l2_target_access_v6` | 1,500 |
| `potential-interference-v7` | `vlm_action_geometry_single_info_l2_potential_interference_v7` | 1,500 |
| `instance-interference-v8` | `vlm_action_geometry_single_info_l2_instance_interference_v8` | 1,500 |
| `instance-occlusion-ltr-v9` | `vlm_action_geometry_single_info_l2_instance_occlusion_ltr_v9` | 1,500 |
| `primary-grasp-blocker-v10` | `vlm_action_geometry_single_info_l2_primary_grasp_blocker_v10` | 1,500 |
| `single-scene-instance-obstruction-v11` | `vlm_action_geometry_single_info_l2_single_scene_instance_obstruction_v11` | 5 |
| `plain-rgb-named-object-obstruction-v12` | `vlm_action_geometry_single_info_l2_plain_rgb_named_object_obstruction_v12` | 25 |
| `single-image-obstruction-score-v13` | `vlm_action_geometry_single_info_l2_single_image_obstruction_score_v13` | 25 |
| `simple-obstruction-score-v14` | `vlm_action_geometry_single_info_l2_simple_obstruction_score_v14` | 25 |
| `direct-retrieval-single-image-v15` | `vlm_action_geometry_single_info_l2_direct_retrieval_v15` | 5 |
| `direct-retrieval-25-scenes-v15` | `vlm_action_geometry_single_info_l2_direct_retrieval_25scenes_v15` | 125 |
| `direct-retrieval-geometry-v16` | `vlm_action_geometry_single_info_l2_direct_retrieval_geometry_v16` | 1,500 |
| `approach-access-geometry-v18` | `vlm_action_geometry_single_info_l2_approach_access_geometry_v18` | 1,500 |
| `straight-access-geometry-v19` | `vlm_action_geometry_single_info_l2_straight_access_geometry_v19` | 1,500 |
| `front-obstruction-geometry-v20` | `vlm_action_geometry_single_info_l2_front_obstruction_geometry_v20` | 1,500 |
| `v16-prompt-replay-v21` | `vlm_action_geometry_single_info_l2_v16_prompt_replay_v21` | 1,500 |
| `c2d8-decision-latent-v22` | `vlm_action_geometry_single_info_l2_c2d8_decision_latent_v22` | 0; 25 derived scene records |
| `c2d4-decision-latent-v23` | `vlm_action_geometry_single_info_l2_c2d4_decision_latent_v23` | 0; 25 derived scene records |

## Included

- 모든 generation study의 raw `runs.jsonl`과 `attempts.jsonl`
- 실행 console/runtime log와 audit log
- L2-specific config, prompt, structured-output schema
- 검토 가능한 JSON, JSONL, CSV, Markdown result
- v22·v23의 extraction manifest, derived records, cosine table, analysis summary와 source artifact hash
- 실행·분석 Python script snapshot

## Deliberately omitted

- 전체 Numbered RGB/plain RGB image collection
- 전체 geometry payload collection과 반복 생성 가능한 `config/run_table.jsonl`
- 각 L2 실행에 중복되는 상위 scheduler 전체 config와 L2 외 prompt/schema
- Isaac Sim USD scene, simulator asset, model checkpoint와 runtime cache
- 대형 interactive HTML gallery와 중복 figure image
- v22·v23의 `latent_states.npz`; 원본 SHA-256은 각 study의 `artifact_hashes.json`과 `analysis_summary.json`에 보존

제외된 입력은 각 raw record의 source path와 hash, study config 및 source manifest로 추적합니다. 이 Git snapshot만으로 model generation을 다시 실행하려면 동일한 source image/geometry collection과 model runtime이 별도로 필요합니다. 포함된 원시 응답의 재분석과 집계 검증에는 snapshot의 로그와 reproduction script를 사용할 수 있습니다.

## Integrity

[`audit/archive_manifest.csv`](audit/archive_manifest.csv)는 snapshot 파일별 relative path, byte size, SHA-256과 JSONL row count를 기록합니다. manifest 자체는 자기참조 hash 문제를 피하기 위해 목록에서 제외합니다.
