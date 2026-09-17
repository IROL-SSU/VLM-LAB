# N1 explicit visible-ID enumeration ablation

## Protocol

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Input: 60 frozen numbered-RGB scenes (L3/L4/L5), five paired seeds
- Design: temperature (0.3/0.5/0.7) × explicit visible-ID enumeration (off/on) × output (TEXT/forced JSON)
- Calls: 3,600; batch size 1
- No object/color vocabulary and no array-length or ID-value constraint
- Primary metric: full-output joint exact. For enumeration conditions this requires the visible-ID list, cross-field ID consistency, and every instance's object/color to be correct.

## Condition results

| Temp. | Enumeration | Output | Instance ID-set | Visible-ID set | Cross-field | Object | Color strict | Color perceptual | Instance scene joint | Full-output joint |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | Off | TEXT | 98.667% | — | — | 99.435% | 95.000% | 99.826% | 96.000% | 96.000% |
| 0.3 | Off | JSON | 100.000% | — | — | 100.000% | 93.783% | 100.000% | 100.000% | 100.000% |
| 0.3 | On | TEXT | 100.000% | 100.000% | 100.000% | 99.522% | 95.870% | 100.000% | 96.333% | 96.333% |
| 0.3 | On | JSON | 100.000% | 100.000% | 100.000% | 99.957% | 94.000% | 100.000% | 99.667% | 99.667% |
| 0.5 | Off | TEXT | 99.333% | — | — | 99.478% | 95.043% | 99.913% | 96.333% | 96.333% |
| 0.5 | Off | JSON | 100.000% | — | — | 100.000% | 93.522% | 99.957% | 99.667% | 99.667% |
| 0.5 | On | TEXT | 100.000% | 100.000% | 100.000% | 99.609% | 95.565% | 100.000% | 97.000% | 97.000% |
| 0.5 | On | JSON | 100.000% | 100.000% | 100.000% | 99.957% | 93.391% | 100.000% | 99.667% | 99.667% |
| 0.7 | Off | TEXT | 98.333% | — | — | 99.304% | 94.913% | 99.783% | 95.000% | 95.000% |
| 0.7 | Off | JSON | 100.000% | — | — | 100.000% | 93.391% | 99.957% | 99.667% | 99.667% |
| 0.7 | On | TEXT | 100.000% | 100.000% | 100.000% | 99.522% | 94.913% | 100.000% | 96.333% | 96.333% |
| 0.7 | On | JSON | 100.000% | 100.000% | 100.000% | 99.826% | 93.000% | 100.000% | 98.667% | 98.667% |

Best condition: `T03_DIRECT_JSON` at 100.000% full-output joint exact.

## Main effects

| Factor | Level | Full-output joint | Instance scene joint |
|---|---|---:|---:|
| temperature | 0.3 | 98.000% | 98.000% |
| temperature | 0.5 | 98.167% | 98.167% |
| temperature | 0.7 | 97.417% | 97.417% |
| enumeration | False | 97.778% | 97.778% |
| enumeration | True | 97.944% | 97.944% |
| output_format | TEXT | 96.167% | 96.167% |
| output_format | JSON_FORCED | 99.556% | 99.556% |

## Paired enumeration contrasts

| Contrast | Enumeration minus direct | Direct-only correct | Enumeration-only correct | McNemar p |
|---|---:|---:|---:|---:|
| enumeration at T=0.3, TEXT | +0.333 pp | 5 | 6 | 1 |
| enumeration at T=0.3, JSON | -0.333 pp | 1 | 0 | 1 |
| enumeration at T=0.5, TEXT | +0.667 pp | 5 | 7 | 0.774414 |
| enumeration at T=0.5, JSON | +0.000 pp | 1 | 1 | 1 |
| enumeration at T=0.7, TEXT | +1.333 pp | 7 | 11 | 0.480682 |
| enumeration at T=0.7, JSON | -1.000 pp | 4 | 1 | 0.375 |

## Interpretation

1. Explicit visible-ID enumeration completely removed ID-coverage errors. The direct conditions produced 11 missing IDs and one extra ID across 1,800 calls (99.389% ID-set exact), whereas enumeration produced no missing, extra, or duplicate IDs (100.000%). The explicit `visible_ids` field itself and its cross-field consistency with `instances[].id` were also 100.000% across all 1,800 enumeration calls.
2. Enumeration had only a small effect on the primary full-output score: 97.778% without it and 97.944% with it (+0.167 pp). In the fully paired comparison, direct-only correct occurred 23 times and enumeration-only correct 26 times (McNemar p=0.77545). Thus the experiment supports a strong ID-coverage benefit, but not a statistically detectable overall scene-joint improvement.
3. The small overall gain occurs because enumeration fixes omissions but does not solve semantic ambiguity. All 37 incorrect object instances under enumeration involved a small set of ambiguous labels, dominated by the handle-occluded `mug_7` being called `container` or `can`. In the enumeration-JSON conditions, every one of the six failed calls was a `mug_7` object-name error; the visible-ID list and color were still correct.
4. Forced JSON was the largest main effect: 99.556% versus 96.167% full-output joint exact for TEXT (+3.389 pp). In the paired comparison JSON-only correct occurred 66 times and TEXT-only correct five times (McNemar p=1.19e-14). JSON reached 100% instance ID-set exact and 99.957% object accuracy.
5. TEXT retained the higher strict single-hue color score (95.217% versus 93.514% for JSON), but their perceptual color scores were nearly identical (99.920% versus 99.986%). The strict gap is therefore mainly a canonical color-word choice effect rather than a substantial perceptual failure.
6. Temperature had a much smaller effect than output format. Marginal full-output scores were 98.000% at 0.3, 98.167% at 0.5, and 97.417% at 0.7. The 0.5-to-0.7 paired contrast was -0.750 pp (nominal McNemar p=0.02246); other temperature contrasts were not significant. These exploratory pairwise tests are not corrected for multiple comparisons.
7. By level, enumeration changed full-output joint exact from 99.167% to 99.500% on L3, 98.000% to 97.333% on L4, and 96.167% to 97.000% on L5. It helped ID coverage most in L4/L5, while the remaining semantic errors concentrated in particular occluded assets.

## Recommended operating condition

- For the intended N0-to-N1 pipeline, use explicit visible-ID enumeration with forced JSON and temperature 0.3. It achieved 299/300 full-output exact calls (99.667%), with 100% visible-ID accuracy, 100% cross-field consistency, and one remaining `mug_7 → container` error.
- If only the final N1 answer matters and an explicit N0 inventory is not required, temperature 0.3 direct JSON was the best observed condition at 300/300. The paired difference from enumeration JSON was one call and was not significant.

## Audit

- Records: 3600/3600
- Parse failures: 0
- Length finishes: 0
- Batch-size values: [1]
- A 163-call schema-order validation run was archived and excluded before the final run because it generated `instances` before `visible_ids`. The reported 3,600 calls all use the corrected `visible_ids → instances` order.

## Files

- `results/condition_summary.csv`
- `results/level_condition_summary.csv`
- `results/main_effect_summary.csv`
- `results/paired_comparisons.csv`
- `results/failures.csv`
- `results/evaluated_records.jsonl`
- `figures/main_effect_full_output_joint_exact.png`
