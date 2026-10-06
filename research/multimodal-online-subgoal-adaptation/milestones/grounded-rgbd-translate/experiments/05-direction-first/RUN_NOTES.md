# Direction-first joint action training

The model classifies one action from 332 joint LEFT/RIGHT and distance classes. Direction is decoded by summing probability across every distance class on each side before selecting distance within the chosen side. Training restricts safe-distance supervision to the canonical direction, weights direction loss by 2.0, retains horizontal mirroring, and uses dropout 0.2 plus AdamW regularization.

## Selected checkpoint

- Epoch: 70
- Validation canonical direction: 90.0%
- Validation canonical direction and safe action: 70/100
- Validation safe action: 70/100
- Validation preferred action: 3/100
- Validation mean predicted movement: 83.37 mm

## Reused test benchmark

- Safe/static-proxy-valid action: 55/100
- Canonical direction accuracy: 79.0%
- Canonical direction and proxy valid: 53/100
- Preferred short-safe action: 1/100
- Mean predicted movement: 83.19 mm
- Exact-minimum distance MAE: 30.67 mm
- Corridor clear: 74/100
- Collision/boundary safe: 76/100
- Test scenes where the opposite direction also has a safe catalog action: 35/100

## Controlled comparison

| Model | Canonical direction | Exact-min distance MAE | Static proxy valid | Corridor clear | Collision safe |
|---|---:|---:|---:|---:|---:|
| 8-output baseline | 74.0% | 25.60 mm | 40/100 | 54/100 | 75/100 |
| Signed scalar + interval | 68.0% | 35.24 mm | 43/100 | 63/100 | 74/100 |
| Joint actions + mirror | 72.0% | 40.53 mm | 61/100 | 87/100 | 73/100 |
| Joint + short-safe preference | 71.0% | 38.72 mm | 64/100 | 87/100 | 76/100 |
| Direction-first joint actions | 79.0% | 30.67 mm | 55/100 | 74/100 | 76/100 |

## Interpretation limit

The test layouts were already inspected in earlier experiments. This is a controlled comparison on the same scenes, not an untouched final generalization result. Horizontal mirroring enforces symmetry but does not add genuinely new scene layouts.
