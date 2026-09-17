# N2-Basic unified prompt and coarse-color grounding

## Protocol

- 60 numbered-RGB scenes × Q1/Q1.5/Q2/Q3 × five seeds = 1,200 calls
- One unified English prompt template for all four query types
- Confirmed visible IDs, forced JSON, batch size 1
- Nine basic colors: black, blue, brown, green, orange, pink, purple, white, yellow
- Fine shades are merged before query generation and canonical scoring; notably beige/tan -> brown, dark blue/navy -> blue, and dark purple/violet -> purple
- A separate perceptual score uses the previously adjudicated N1 asset palettes; it does not change the canonical target-ID set

## Results

| Scope | Calls | ID-set exact | Micro P | Micro R | Micro F1 | Request decomposition | Canonical basic color | Perceptual color | Semantic scene joint | Logic scene exact | Aggregation exact | Absent FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVERALL | 1200 | 86.667% | 94.184% | 95.030% | 94.605% | 100.000% | 83.880% | 100.000% | 32.167% | 99.583% | 100.000% | 8.333% |
| Q1_CATEGORY_PRESENT | 300 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 83.870% | 100.000% | 34.000% | 99.333% | 100.000% | 0.000% |
| Q1_5_COLOR_ONLY_PRESENT | 300 | 67.667% | 90.861% | 89.913% | 90.385% | 100.000% | 83.304% | 100.000% | 32.000% | 100.000% | 100.000% | 0.000% |
| Q2_CATEGORY_COLOR_PRESENT | 300 | 87.333% | 94.629% | 93.671% | 94.148% | 100.000% | 84.652% | 100.000% | 31.000% | 100.000% | 100.000% | 0.000% |
| Q3_HARD_NEGATIVE_ABSENT | 300 | 91.667% | N/A | N/A | N/A | 100.000% | 83.696% | 100.000% | 31.667% | 99.000% | 100.000% | 8.333% |

## Results by level

| Level | Query | ID-set exact | Micro F1 | Basic-color accuracy | Absent FPR |
|---|---|---:|---:|---:|---:|
| L3 | Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 91.667% | 0.000% |
| L3 | Q1_5_COLOR_ONLY_PRESENT | 90.000% | 97.872% | 90.833% | 0.000% |
| L3 | Q2_CATEGORY_COLOR_PRESENT | 100.000% | 100.000% | 90.833% | 0.000% |
| L3 | Q3_HARD_NEGATIVE_ABSENT | 100.000% | N/A | 91.667% | 0.000% |
| L4 | Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 80.750% | 0.000% |
| L4 | Q1_5_COLOR_ONLY_PRESENT | 55.000% | 82.840% | 80.250% | 0.000% |
| L4 | Q2_CATEGORY_COLOR_PRESENT | 77.000% | 84.577% | 81.625% | 0.000% |
| L4 | Q3_HARD_NEGATIVE_ABSENT | 98.000% | N/A | 79.500% | 2.000% |
| L5 | Q1_CATEGORY_PRESENT | 100.000% | 100.000% | 81.444% | 0.000% |
| L5 | Q1_5_COLOR_ONLY_PRESENT | 58.000% | 87.500% | 81.000% | 0.000% |
| L5 | Q2_CATEGORY_COLOR_PRESENT | 85.000% | 92.308% | 83.222% | 0.000% |
| L5 | Q3_HARD_NEGATIVE_ABSENT | 77.000% | N/A | 82.111% | 23.000% |

## Error attribution

| Query | Error type | Count | Percent of query calls |
|---|---|---:|---:|
| Q1_5_COLOR_ONLY_PRESENT | canonical_color_disagreement | 97 | 32.333% |
| Q2_CATEGORY_COLOR_PRESENT | canonical_color_disagreement | 38 | 12.667% |
| Q3_HARD_NEGATIVE_ABSENT | canonical_color_disagreement | 18 | 6.000% |
| Q3_HARD_NEGATIVE_ABSENT | object_recognition_failure | 7 | 2.333% |

## Interpretation

- The model returned the complete confirmed ID inventory in every call, parsed all request constraints, and kept `matching_ids` consistent with its own per-instance match flags.
- Canonical single-color agreement was 83.880%, while the previously adjudicated N1 perceptual palette accepted 100.000% of instance color labels.
- Of the 160 target-set failures, 153 were canonical-color disagreements whose predicted color remained perceptually acceptable; seven were object-recognition failures in Q3.
- Therefore the strict target-ID result mixes grounding ability with ambiguity in the single canonical color label. A clean follow-up should exclude any distractor whose perceptual palette also contains the query color, especially when constructing Q3 absent queries.

## Audit

`{'expected': 1200, 'observed': 1200, 'unique_run_ids': 1200, 'query_counts': {'Q1_CATEGORY_PRESENT': 300, 'Q1_5_COLOR_ONLY_PRESENT': 300, 'Q2_CATEGORY_COLOR_PRESENT': 300, 'Q3_HARD_NEGATIVE_ABSENT': 300}, 'seed_counts': {'28101': 240, '28102': 240, '28103': 240, '28104': 240, '28105': 240}, 'batch_sizes': [1], 'finish_reasons': {'stop': 1200}, 'parse_failures': 0, 'length_finishes': 0, 'outside_candidate_failures': 0, 'same_prompt_template_configured': True, 'rendered_prompt_hash_count': 240, 'system_prompt_hash_count': 1, 'basic_colors': ['black', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'white', 'yellow'], 'perceptual_basic_color_tokens': {'acafela': ['black', 'blue'], 'coldgrape': ['blue', 'purple'], 'top': ['blue', 'green'], 'cocopalm': ['pink', 'purple'], 'minutemad': ['brown', 'green', 'orange', 'yellow'], 'cantata': ['black', 'brown', 'white'], 'biracsikhye': ['black', 'brown', 'yellow'], 'letsbe': ['blue'], 'mug_7': ['brown', 'orange'], 'Mug_2': ['black'], 'Mug_4': ['yellow'], 'cup_7': ['purple'], 'cup_8': ['blue', 'purple'], 'Cup_2': ['white'], 'Cup_4': ['blue']}}`
