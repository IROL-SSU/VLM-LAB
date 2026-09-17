# Initial N3 75 scenes — missing cross-task completion

- Same 75 numbered RGB images as the original benchmark
- Added tasks: D0 27 scenes front-to-back; D1 48 scenes left-to-right
- Repeats: 5 per scene (375 new calls)
- Prompts, model, seeds, generation settings, and forced-JSON schemas match the original experiment

## Complete 2×2 task/dataset matrix

| Source scenes | Task | Status | Exact | All-pairs | Scene 5/5 | Scene 0/5 |
|---|---|---|---:|---:|---:|---:|
| N3-D0 (27) | Left-to-right | original | 100.000% (135/135) | 100.000% | 27/27 | 0/27 |
| N3-D0 (27) | Front-to-back | cross-task added | 34.815% (47/135) | 74.667% | 8/27 | 15/27 |
| N3-D1 (48) | Left-to-right | cross-task added | 100.000% (240/240) | 100.000% | 48/48 | 0/48 |
| N3-D1 (48) | Front-to-back | original | 100.000% (240/240) | 100.000% | 48/48 | 0/48 |

## Added-task breakdown by object count

| Source scenes | Task | Objects | Exact | All-pairs |
|---|---|---:|---:|---:|
| N3-D0 | D1_FRONT_TO_BACK | 2 | 46.667% | 46.667% |
| N3-D0 | D1_FRONT_TO_BACK | 3 | 26.667% | 75.556% |
| N3-D0 | D1_FRONT_TO_BACK | 4 | 31.111% | 78.889% |
| N3-D1 | D0_LEFT_TO_RIGHT | 2 | 100.000% | 100.000% |
| N3-D1 | D0_LEFT_TO_RIGHT | 3 | 100.000% | 100.000% |

## Added-task breakdown by size cue

| Source scenes | Task | Size cue | Exact | All-pairs |
|---|---|---|---:|---:|
| N3-D0 | D1_FRONT_TO_BACK | conflict | 35.556% | 76.667% |
| N3-D0 | D1_FRONT_TO_BACK | congruent | 55.556% | 83.333% |
| N3-D0 | D1_FRONT_TO_BACK | neutral | 13.333% | 64.000% |
| N3-D1 | D0_LEFT_TO_RIGHT | conflict | 100.000% | 100.000% |
| N3-D1 | D0_LEFT_TO_RIGHT | congruent | 100.000% | 100.000% |
| N3-D1 | D0_LEFT_TO_RIGHT | neutral | 100.000% | 100.000% |

## Audit

- New calls: 375/375
- Format failures: 0
- Non-stop finishes: 0
- Duplicate run IDs: 0
- Total failures: 88
