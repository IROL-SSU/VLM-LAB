# L2 visibility-only v4

## Definition

`direct_graspable = true` if and only if Isaac pixel-exact visibility is FULL. Object proximity and finger clearance are ignored.

## Integrity

- Completed: 1500/1500
- Parse errors: 0
- Post-validation failures: 0
- Infrastructure errors: 0

## Overall on the same v4 GT

| Output formulation | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| Visibility-only L2 v4 | 44.0% | 56.2% | 100.0% | 12.5% | 540/0/120/840 |
| Grasp-zone L2 v3 responses | 67.3% | 74.4% | 99.4% | 49.3% | 537/3/473/487 |
| Original L1, FULL→true | 66.9% | 74.1% | 100.0% | 48.2% | 540/0/463/497 |

## Visibility-only v4 by condition

| Condition | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 48.0% | 59.4% | 100.0% | 18.8% | 45/0/15/65 |
| C1  R Numeric | 48.0% | 59.4% | 100.0% | 18.8% | 45/0/15/65 |
| C2 D4 R Qualitative D4 | 50.4% | 61.3% | 100.0% | 22.5% | 45/0/18/62 |
| C2 D8 R Qualitative D8 | 52.8% | 63.1% | 100.0% | 26.2% | 45/0/21/59 |
| C3 D4 F Numeric D4 | 39.2% | 52.5% | 100.0% | 5.0% | 45/0/4/76 |
| C3 D8 F Numeric D8 | 39.2% | 52.5% | 100.0% | 5.0% | 45/0/4/76 |
| C4 D4 F Qualitative D4 | 40.8% | 53.8% | 100.0% | 7.5% | 45/0/6/74 |
| C4 D8 F Qualitative D8 | 44.0% | 56.2% | 100.0% | 12.5% | 45/0/10/70 |
| C5 D4 M Numeric D4 | 44.0% | 56.2% | 100.0% | 12.5% | 45/0/10/70 |
| C5 D8 M Numeric D8 | 40.0% | 53.1% | 100.0% | 6.2% | 45/0/5/75 |
| C6 D4 M Qualitative D4 | 44.8% | 56.9% | 100.0% | 13.8% | 45/0/11/69 |
| C6 D8 M Qualitative D8 | 36.8% | 50.6% | 100.0% | 1.2% | 45/0/1/79 |

## Accuracy by scene family

| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 40.0% | 0.0% | 20.0% | 100.0% | 80.0% |
| C1  R Numeric | 36.0% | 4.0% | 20.0% | 100.0% | 80.0% |
| C2 D4 R Qualitative D4 | 28.0% | 40.0% | 4.0% | 100.0% | 80.0% |
| C2 D8 R Qualitative D8 | 40.0% | 24.0% | 20.0% | 100.0% | 80.0% |
| C3 D4 F Numeric D4 | 16.0% | 0.0% | 0.0% | 100.0% | 80.0% |
| C3 D8 F Numeric D8 | 16.0% | 0.0% | 0.0% | 100.0% | 80.0% |
| C4 D4 F Qualitative D4 | 20.0% | 0.0% | 4.0% | 100.0% | 80.0% |
| C4 D8 F Qualitative D8 | 40.0% | 0.0% | 0.0% | 100.0% | 80.0% |
| C5 D4 M Numeric D4 | 40.0% | 0.0% | 0.0% | 100.0% | 80.0% |
| C5 D8 M Numeric D8 | 20.0% | 0.0% | 0.0% | 100.0% | 80.0% |
| C6 D4 M Qualitative D4 | 24.0% | 0.0% | 20.0% | 100.0% | 80.0% |
| C6 D8 M Qualitative D8 | 4.0% | 0.0% | 0.0% | 100.0% | 80.0% |

## Interpretation

- V4 has no false negatives, but 840/960 negative runs are false positives. The model answers true whenever it can locate a visible portion of the target rather than requiring a pixel-exact FULL silhouette.
- The original L1 FULL/PARTIAL/NOT_VISIBLE formulation performs substantially better on exactly the same Boolean target (balanced accuracy 74.1% versus 56.2%).
- This indicates that the semantic field name and output formulation matter: asking `direct_graspable` induces a permissive graspability prior even when the prompt defines it as visibility-only.
- The appropriate first-stage task is therefore the existing explicit visibility classification, or a direct `occluded` field, not a redefined `direct_graspable` Boolean.
