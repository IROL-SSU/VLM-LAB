# L2 finger-placement-zone v3 rerun

## Integrity

- Completed: 1500/1500
- Parse errors: 0
- Post-validation failures: 0
- Infrastructure errors: 0

## GT change

- Positive scenes: 6/25
- Labels changed from formal v2: 2
  - scene_fc_blocked_v02: False -> True
  - scene_fc_clear_v05: False -> True

## Overall comparison (both rescored against v3 GT)

| Responses | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| Formal v2 | 78.6% | 60.5% | 25.8% | 95.3% | 93/267/1086/54 |
| Grasp-zone v3 | 55.3% | 70.3% | 99.2% | 41.5% | 357/3/473/667 |

## Grasp-zone v3 condition metrics

| Condition | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP | Predicted true |
|---|---:|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 81.6% | 84.5% | 90.0% | 78.9% | 27/3/75/20 | 47 |
| C1  R Numeric | 65.6% | 77.4% | 100.0% | 54.7% | 30/0/52/43 | 73 |
| C2 D4 R Qualitative D4 | 51.2% | 67.9% | 100.0% | 35.8% | 30/0/34/61 | 91 |
| C2 D8 R Qualitative D8 | 47.2% | 65.3% | 100.0% | 30.5% | 30/0/29/66 | 96 |
| C3 D4 F Numeric D4 | 63.2% | 75.8% | 100.0% | 51.6% | 30/0/49/46 | 76 |
| C3 D8 F Numeric D8 | 60.0% | 73.7% | 100.0% | 47.4% | 30/0/45/50 | 80 |
| C4 D4 F Qualitative D4 | 53.6% | 69.5% | 100.0% | 38.9% | 30/0/37/58 | 88 |
| C4 D8 F Qualitative D8 | 42.4% | 62.1% | 100.0% | 24.2% | 30/0/23/72 | 102 |
| C5 D4 M Numeric D4 | 56.8% | 71.6% | 100.0% | 43.2% | 30/0/41/54 | 84 |
| C5 D8 M Numeric D8 | 46.4% | 64.7% | 100.0% | 29.5% | 30/0/28/67 | 97 |
| C6 D4 M Qualitative D4 | 50.4% | 67.4% | 100.0% | 34.7% | 30/0/33/62 | 92 |
| C6 D8 M Qualitative D8 | 45.6% | 64.2% | 100.0% | 28.4% | 30/0/27/68 | 98 |

## Per-condition change from formal v2

Both versions below are scored against the same v3 GT.

| Condition | Balanced accuracy v2→v3 | True recall v2→v3 | Specificity v2→v3 |
|---|---:|---:|---:|
| C0  Numbered RGB only | 50.0% → 84.5% | 0.0% → 90.0% | 100.0% → 78.9% |
| C1  R Numeric | 55.0% → 77.4% | 10.0% → 100.0% | 100.0% → 54.7% |
| C2 D4 R Qualitative D4 | 51.7% → 67.9% | 3.3% → 100.0% | 100.0% → 35.8% |
| C2 D8 R Qualitative D8 | 49.5% → 65.3% | 0.0% → 100.0% | 98.9% → 30.5% |
| C3 D4 F Numeric D4 | 70.4% → 75.8% | 53.3% → 100.0% | 87.4% → 51.6% |
| C3 D8 F Numeric D8 | 73.7% → 73.7% | 60.0% → 100.0% | 87.4% → 47.4% |
| C4 D4 F Qualitative D4 | 50.0% → 69.5% | 0.0% → 100.0% | 100.0% → 38.9% |
| C4 D8 F Qualitative D8 | 54.5% → 62.1% | 10.0% → 100.0% | 98.9% → 24.2% |
| C5 D4 M Numeric D4 | 76.5% → 71.6% | 66.7% → 100.0% | 86.3% → 43.2% |
| C5 D8 M Numeric D8 | 73.6% → 64.7% | 56.7% → 100.0% | 90.5% → 29.5% |
| C6 D4 M Qualitative D4 | 62.3% → 67.4% | 26.7% → 100.0% | 97.9% → 34.7% |
| C6 D8 M Qualitative D8 | 59.6% → 64.2% | 23.3% → 100.0% | 95.8% → 28.4% |

## Accuracy by scene family

| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 100.0% | 100.0% | 100.0% | 100.0% | 8.0% |
| C1  R Numeric | 64.0% | 72.0% | 72.0% | 100.0% | 20.0% |
| C2 D4 R Qualitative D4 | 32.0% | 48.0% | 44.0% | 100.0% | 32.0% |
| C2 D8 R Qualitative D8 | 32.0% | 40.0% | 36.0% | 100.0% | 28.0% |
| C3 D4 F Numeric D4 | 40.0% | 72.0% | 84.0% | 100.0% | 20.0% |
| C3 D8 F Numeric D8 | 32.0% | 56.0% | 92.0% | 100.0% | 20.0% |
| C4 D4 F Qualitative D4 | 32.0% | 52.0% | 64.0% | 100.0% | 20.0% |
| C4 D8 F Qualitative D8 | 16.0% | 40.0% | 36.0% | 100.0% | 20.0% |
| C5 D4 M Numeric D4 | 68.0% | 56.0% | 40.0% | 100.0% | 20.0% |
| C5 D8 M Numeric D8 | 52.0% | 36.0% | 24.0% | 100.0% | 20.0% |
| C6 D4 M Qualitative D4 | 32.0% | 44.0% | 56.0% | 100.0% | 20.0% |
| C6 D8 M Qualitative D8 | 12.0% | 44.0% | 52.0% | 100.0% | 20.0% |

## Byte-identical input subset

- Runs with identical image, geometry, schema, and target ID: 1125
- These isolate the prompt/decision-rule effect; C1/C2 are excluded where R geometry changed.

## Diagnostics

- All three false negatives are C0 responses for scene_fc_blocked_v02 (seeds 28102, 28103, and 28104).
- C1 R Numeric returned true in all 20 runs on the four GT-negative FC_BLOCKED scenes, even though each payload contains a zero distance to one finger-placement zone.
- C2 also returned true in 17/20 D4 runs and 18/20 D8 runs on those four scenes despite an authoritative OCCUPIED side status.
- scene_fc_blocked_v02 is the sole manifest/Isaac design mismatch: its actual minimum mesh gap is 2.352752 cm, slightly larger than the required 2.2 cm side clearance, so the Isaac geometry oracle labels it true.

## Interpretation

- Replacing the universal 5 cm rule removed the near-always-false behavior: true recall rose from 25.8% to 99.2% when both versions are scored on v3 GT.
- The change overcorrected toward true. Specificity fell from 95.3% to 41.5%, so raw accuracy fell even though balanced accuracy rose from 60.5% to 70.3%.
- C0 Numbered RGB only is the strongest v3 condition by balanced accuracy (84.5%). C1 R Numeric is second (77.4%).
- The main remaining error is failure to preserve the independent FULL-visibility test and, for R inputs, failure to obey explicit occupied-zone evidence. The revised clearance concept is physically better grounded, but this prompt does not yet make the two-test conjunction reliable.
