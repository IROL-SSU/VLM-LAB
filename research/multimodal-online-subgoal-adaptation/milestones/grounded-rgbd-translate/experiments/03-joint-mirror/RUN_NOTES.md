# Joint direction-distance action training with horizontal mirroring

The model classifies one action from 100 joint LEFT/RIGHT and distance classes. Training uses original plus horizontally mirrored RGB, masks, and shelf-frame point clouds. Every catalog action passing the static geometry check is an acceptable target; inference still emits one argmax action.

## Selected checkpoint

- Epoch: 19
- Validation safe action: 79/100
- Validation canonical direction: 79.0%
- Validation signed displacement MAE: 63.42 mm

## Reused test benchmark

- Safe/static-proxy-valid action: 61/100
- Canonical direction accuracy: 72.0%
- Canonical direction and proxy valid: 54/100
- Exact-minimum distance MAE: 40.53 mm
- Corridor clear: 87/100
- Collision/boundary safe: 73/100
- Test scenes where the opposite direction also has a safe catalog action: 35/100

## Controlled comparison

| Model | Canonical direction | Exact-min distance MAE | Static proxy valid | Corridor clear | Collision safe |
|---|---:|---:|---:|---:|---:|
| 8-output baseline | 74.0% | 25.60 mm | 40/100 | 54/100 | 75/100 |
| Signed scalar + interval | 68.0% | 35.24 mm | 43/100 | 63/100 | 74/100 |
| Joint actions + mirror | 72.0% | 40.53 mm | 61/100 | 87/100 | 73/100 |

## Interpretation limit

The test layouts were already inspected in earlier experiments. This is a controlled comparison on the same scenes, not an untouched final generalization result. Horizontal mirroring enforces symmetry but does not add genuinely new scene layouts.
