# L2 formal-predicate v2 rerun

## Integrity

- Completed: 1500/1500
- Parse errors: 0
- Post-validation failures: 0
- Infrastructure errors: 0

## Overall comparison

| Prompt | Accuracy | Balanced accuracy | True recall | Specificity | TP | FN | TN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 84.1% | 50.4% | 0.8% | 100.0% | 2 | 238 | 1260 | 0 |
| Formal v2 | 83.4% | 61.3% | 28.7% | 93.8% | 69 | 171 | 1182 | 78 |

## Formal-v2 condition metrics

| Condition | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP | Predicted true |
|---|---:|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 84.0% | 50.0% | 0.0% | 100.0% | 0/20/105/0 | 0 |
| C1  R Numeric | 86.4% | 57.5% | 15.0% | 100.0% | 3/17/105/0 | 3 |
| C2 D4 R Qualitative D4 | 84.8% | 52.5% | 5.0% | 100.0% | 1/19/105/0 | 1 |
| C2 D8 R Qualitative D8 | 83.2% | 49.5% | 0.0% | 99.0% | 0/20/104/1 | 1 |
| C3 D4 F Numeric D4 | 76.0% | 63.5% | 45.0% | 81.9% | 9/11/86/19 | 28 |
| C3 D8 F Numeric D8 | 80.8% | 74.4% | 65.0% | 83.8% | 13/7/88/17 | 30 |
| C4 D4 F Qualitative D4 | 84.0% | 50.0% | 0.0% | 100.0% | 0/20/105/0 | 0 |
| C4 D8 F Qualitative D8 | 85.6% | 57.0% | 15.0% | 99.0% | 3/17/104/1 | 4 |
| C5 D4 M Numeric D4 | 80.0% | 76.0% | 70.0% | 81.9% | 14/6/86/19 | 33 |
| C5 D8 M Numeric D8 | 85.6% | 79.3% | 70.0% | 88.6% | 14/6/93/12 | 26 |
| C6 D4 M Qualitative D4 | 88.8% | 69.0% | 40.0% | 98.1% | 8/12/103/2 | 10 |
| C6 D8 M Qualitative D8 | 81.6% | 56.7% | 20.0% | 93.3% | 4/16/98/7 | 11 |

## Accuracy by scene family

| Condition | TRANSLATE | ROTATE | LIFT_AND_RELOCATE | FC_CLEAR | FC_BLOCKED |
|---|---:|---:|---:|---:|---:|
| C0  Numbered RGB only | 100.0% | 100.0% | 100.0% | 20.0% | 100.0% |
| C1  R Numeric | 100.0% | 100.0% | 100.0% | 32.0% | 100.0% |
| C2 D4 R Qualitative D4 | 100.0% | 100.0% | 100.0% | 24.0% | 100.0% |
| C2 D8 R Qualitative D8 | 100.0% | 100.0% | 100.0% | 20.0% | 96.0% |
| C3 D4 F Numeric D4 | 100.0% | 100.0% | 100.0% | 36.0% | 44.0% |
| C3 D8 F Numeric D8 | 100.0% | 100.0% | 100.0% | 60.0% | 44.0% |
| C4 D4 F Qualitative D4 | 100.0% | 100.0% | 100.0% | 20.0% | 100.0% |
| C4 D8 F Qualitative D8 | 100.0% | 100.0% | 100.0% | 32.0% | 96.0% |
| C5 D4 M Numeric D4 | 100.0% | 100.0% | 100.0% | 60.0% | 40.0% |
| C5 D8 M Numeric D8 | 100.0% | 100.0% | 100.0% | 64.0% | 64.0% |
| C6 D4 M Qualitative D4 | 100.0% | 100.0% | 100.0% | 52.0% | 92.0% |
| C6 D8 M Qualitative D8 | 100.0% | 100.0% | 100.0% | 24.0% | 84.0% |

## Same-input prompt-only subset

There are 600 runs whose numbered RGB, geometry, schema, and target ID are byte-identical between versions.

| Prompt | Accuracy | Balanced accuracy | True recall | Specificity | TP/FN/TN/FP |
|---|---:|---:|---:|---:|---:|
| Original | 90.0% | 50.0% | 0.0% | 100.0% | 0/60/540/0 |
| Formal v2 | 88.5% | 63.2% | 31.7% | 94.8% | 19/41/512/28 |

## Interpretation

- The revised prompt broke the near-always-false behavior, but did not solve L2.
- Overall true recall rose substantially, while false positives reduced specificity and raw accuracy.
- R Numeric and R Qualitative still failed to apply their authoritative clearance rule reliably.
- On the same four GT-positive scenes, the original L1 task classified visibility as FULL in 20/20 runs for each of C0, C1, C2-D4, and C2-D8. Therefore the C1/C2 L2 false negatives are not explained by an inability to recognize FULL when it is queried directly; the remaining failure is in applying or combining the formal tests inside L2.
- F/M conditions produced more true answers but also most false positives; they do not directly encode the benchmark's exact clearance predicate.
- Because only four scenes are GT-positive, condition-level positive recall is based on 20 runs and should not be overinterpreted.
