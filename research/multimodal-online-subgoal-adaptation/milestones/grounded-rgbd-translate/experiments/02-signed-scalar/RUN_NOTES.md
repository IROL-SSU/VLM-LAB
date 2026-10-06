# Signed displacement and safe-interval training result

The model predicts one signed shelf-X displacement: negative is LEFT and positive is RIGHT. It was trained against a signed collision-free interval with an adaptive edge margin capped at 10 mm.

## Selected checkpoint

- Epoch: 30
- Validation static proxy valid: 63/100
- Validation direction accuracy: 73.0%
- Validation signed displacement MAE against the minimum label: 67.14 mm

## Reused test benchmark

- Direction accuracy: 68.0%
- Exact-minimum distance MAE: 35.24 mm
- Signed displacement MAE: 68.75 mm
- Safe interval hit: 32/100
- Static proxy valid: 43/100
- Corridor clear: 63/100
- Collision/boundary safe: 74/100

## Registered baseline comparison

| Metric | 8-output baseline | Signed safe-interval model | Change |
|---|---:|---:|---:|
| Direction accuracy | 74.0% | 68.0% | -6.0 pp |
| Exact-minimum distance MAE | 25.60 mm | 35.24 mm | +9.65 mm |
| Signed displacement MAE | 57.40 mm | 68.75 mm | +11.36 mm |
| Static proxy valid | 40/100 | 43/100 | +3 |

## Interpretation limit

The existing test split had already been inspected while developing the baseline and motivating this change. These numbers are useful for controlled comparison on the same scenes, but a newly generated untouched test set is required for a final generalization claim.
