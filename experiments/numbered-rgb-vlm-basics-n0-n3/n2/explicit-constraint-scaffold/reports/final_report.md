# Explicit request-constraint scaffold N2

## Protocol

- 60 numbered-RGB scenes × Q1/Q2/Q3 × five seeds = 900 calls
- Natural English request, confirmed visible IDs, forced JSON, batch size 1
- Controlled paired comparison with the prior semantic-scaffold run: identical scenes, requests, seeds, model, and decoding parameters

## Results

| Scope | Matching-ID exact | Request decomposition exact | Type-flag accuracy | Truth-table scene exact | Aggregation exact | Semantic scene joint exact | Absent FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| OVERALL | 85.111% | 100.000% | 99.594% | 93.889% | 100.000% | 93.667% | 18.000% |
| Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 0.000% |
| Q2_CATEGORY_COLOR_PRESENT | 73.333% | 100.000% | 100.000% | 93.333% | 100.000% | 100.000% | 0.000% |
| Q3_HARD_NEGATIVE_ABSENT | 82.000% | 100.000% | 98.783% | 88.333% | 100.000% | 81.000% | 18.000% |

## Results by level

| Level | Query | Matching-ID exact | Truth-table scene exact | Absent FPR |
|---|---|---:|---:|---:|
| L3 | Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 0.000% |
| L3 | Q2_CATEGORY_COLOR_PRESENT | 65.000% | 92.000% | 0.000% |
| L3 | Q3_HARD_NEGATIVE_ABSENT | 100.000% | 79.000% | 0.000% |
| L4 | Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 0.000% |
| L4 | Q2_CATEGORY_COLOR_PRESENT | 75.000% | 93.000% | 0.000% |
| L4 | Q3_HARD_NEGATIVE_ABSENT | 81.000% | 95.000% | 19.000% |
| L5 | Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 0.000% |
| L5 | Q2_CATEGORY_COLOR_PRESENT | 80.000% | 95.000% | 0.000% |
| L5 | Q3_HARD_NEGATIVE_ABSENT | 65.000% | 91.000% | 35.000% |

## Paired comparison

| Scope | Explicit − baseline | Baseline-only correct | Explicit-only correct | McNemar p |
|---|---:|---:|---:|---:|
| OVERALL | +12.778 pp | 12 | 127 | 2.10812e-25 |
| Q1_CATEGORY_PRESENT | +1.667 pp | 0 | 5 | 0.0625 |
| Q2_CATEGORY_COLOR_PRESENT | +0.667 pp | 11 | 13 | 0.83882 |
| Q3_HARD_NEGATIVE_ABSENT | +36.000 pp | 1 | 109 | 1.71023e-31 |

## Error attribution

| Query | Error type | Count | Percent of query calls |
|---|---|---:|---:|
| Q2_CATEGORY_COLOR_PRESENT | gt_target_set_disagreement_with_semantic_inventory_exact | 76 | 25.333% |
| Q2_CATEGORY_COLOR_PRESENT | truth_table_violation | 4 | 1.333% |
| Q3_HARD_NEGATIVE_ABSENT | gt_target_set_disagreement_with_semantic_inventory_exact | 16 | 5.333% |
| Q3_HARD_NEGATIVE_ABSENT | object_type_comparison_failure | 25 | 8.333% |
| Q3_HARD_NEGATIVE_ABSENT | target_failure_with_semantic_error | 9 | 3.000% |
| Q3_HARD_NEGATIVE_ABSENT | truth_table_violation | 4 | 1.333% |

## Evaluation caveat

The target-ID ground truth was generated from asset-level canonical color labels. Some rendered objects admit different but perceptually reasonable color descriptions. For example, the evaluator expects `beige can` while the model consistently describes the visible body as `brown`, and some `dark blue bottle` queries contain additional bottles that the model also describes as dark blue although they are excluded from the asset-derived target set. Consequently, Q2 absolute Matching-ID exact should be interpreted together with a human review of `scene_query_stability.csv`, rather than treating every target-set disagreement as an unambiguous reasoning failure.

## Audit

`{'expected': 900, 'observed': 900, 'unique_run_ids': 900, 'query_counts': {'Q1_CATEGORY_PRESENT': 300, 'Q2_CATEGORY_COLOR_PRESENT': 300, 'Q3_HARD_NEGATIVE_ABSENT': 300}, 'seed_counts': {'28101': 180, '28102': 180, '28103': 180, '28104': 180, '28105': 180}, 'batch_sizes': [1], 'finish_reasons': {'stop': 900}, 'parse_failures': 0, 'length_finishes': 0, 'outside_candidate_failures': 0}`
