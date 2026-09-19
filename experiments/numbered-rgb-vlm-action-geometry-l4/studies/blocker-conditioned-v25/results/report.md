# l4_blocker_conditioned_from_l3_v19_v25 — L4 action × geometry results

Generated: 2026-09-18T08:22:22.812755+00:00

## Completion audit

- Inference: 1,750/1,750
- L4 Isaac Sim replay: 1,750/1,750
- Unique Isaac Sim artifacts: 227
- Parse failures: 0
- Semantic post-validation failures: 178

## Accuracy by condition

| Level | Condition | Resolution | n | Valid | Correct | Accuracy | TRANSLATE | ROTATE | LIFT | FC_CLEAR | FC_BLOCKED |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L4 | C0 Numbered RGB only | D4 | 125 | 100 | 15 | 12.00% | 12.00% | 0.00% | 4.00% | 44.00% | 0.00% |
| L4 | C0 Numbered RGB only | D8 | 125 | 100 | 16 | 12.80% | 16.00% | 0.00% | 4.00% | 44.00% | 0.00% |
| L4 | C1 R numeric | D4 | 125 | 115 | 18 | 14.40% | 36.00% | 0.00% | 16.00% | 20.00% | 0.00% |
| L4 | C1 R numeric | D8 | 125 | 110 | 19 | 15.20% | 32.00% | 0.00% | 16.00% | 24.00% | 4.00% |
| L4 | C2 R qualitative | D4 | 125 | 125 | 31 | 24.80% | 28.00% | 0.00% | 0.00% | 84.00% | 12.00% |
| L4 | C2 R qualitative | D8 | 125 | 123 | 41 | 32.80% | 44.00% | 4.00% | 0.00% | 96.00% | 20.00% |
| L4 | C3 F numeric | D4 | 125 | 113 | 26 | 20.80% | 16.00% | 0.00% | 16.00% | 56.00% | 16.00% |
| L4 | C3 F numeric | D8 | 125 | 108 | 11 | 8.80% | 16.00% | 0.00% | 0.00% | 16.00% | 12.00% |
| L4 | C4 F qualitative | D4 | 125 | 125 | 23 | 18.40% | 16.00% | 0.00% | 0.00% | 44.00% | 32.00% |
| L4 | C4 F qualitative | D8 | 125 | 125 | 23 | 18.40% | 20.00% | 0.00% | 0.00% | 60.00% | 12.00% |
| L4 | C5 M numeric | D4 | 125 | 94 | 22 | 17.60% | 20.00% | 0.00% | 0.00% | 52.00% | 16.00% |
| L4 | C5 M numeric | D8 | 125 | 99 | 31 | 24.80% | 28.00% | 0.00% | 0.00% | 72.00% | 24.00% |
| L4 | C6 M qualitative | D4 | 125 | 120 | 41 | 32.80% | 28.00% | 16.00% | 0.00% | 80.00% | 40.00% |
| L4 | C6 M qualitative | D8 | 125 | 115 | 33 | 26.40% | 20.00% | 8.00% | 0.00% | 64.00% | 40.00% |

## L4 replay

- unique_simulator_artifacts: 227
- simulator_executed_runs: 1572
- action_valid_runs: 350
- collision_runs: 321
- boundary_violation_runs: 67
- action_distribution: {'RETRIEVE': 181, 'TRANSLATE': 1569}
- all_artifacts_direct_stage_execution: True
- all_artifact_baselines_match_preparation: True
- isaac_sim_versions: ['5.1.0-rc.19+release.26219.9c81211b.gl']

## Paired comparisons against C0

Exact two-sided McNemar p-values are uncorrected exploratory values. The five stochastic repeats within a scene are not independent, so these p-values must not be read as confirmatory scene-level inference.

| Level | Condition | Res. | n | C0 | Condition | Delta | p |
|---|---|---:|---:|---:|---:|---:|---:|
| L4 | C1 R numeric | D4 | 125 | 12.00% | 14.40% | +2.40 pp | 0.629059 |
| L4 | C1 R numeric | D8 | 125 | 12.80% | 15.20% | +2.40 pp | 0.690038 |
| L4 | C2 R qualitative | D4 | 125 | 12.00% | 24.80% | +12.80 pp | 0.000144958 |
| L4 | C2 R qualitative | D8 | 125 | 12.80% | 32.80% | +20.00 pp | 4.17233e-07 |
| L4 | C3 F numeric | D4 | 125 | 12.00% | 20.80% | +8.80 pp | 0.0127258 |
| L4 | C3 F numeric | D8 | 125 | 12.80% | 8.80% | -4.00 pp | 0.226562 |
| L4 | C4 F qualitative | D4 | 125 | 12.00% | 18.40% | +6.40 pp | 0.0962524 |
| L4 | C4 F qualitative | D8 | 125 | 12.80% | 18.40% | +5.60 pp | 0.118469 |
| L4 | C5 M numeric | D4 | 125 | 12.00% | 17.60% | +5.60 pp | 0.189247 |
| L4 | C5 M numeric | D8 | 125 | 12.80% | 24.80% | +12.00 pp | 0.00592461 |
| L4 | C6 M qualitative | D4 | 125 | 12.00% | 32.80% | +20.80 pp | 2.16067e-07 |
| L4 | C6 M qualitative | D8 | 125 | 12.80% | 26.40% | +13.60 pp | 0.000910521 |
