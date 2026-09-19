# l4_action_full_from_v19_v24 — L4 action × geometry results

Generated: 2026-09-18T07:51:46.197903+00:00

## Completion audit

- Inference: 1,750/1,750
- L4 Isaac Sim replay: 1,750/1,750
- Unique Isaac Sim artifacts: 24
- Parse failures: 0
- Semantic post-validation failures: 1,289

## Accuracy by condition

| Level | Condition | Resolution | n | Valid | Correct | Accuracy | TRANSLATE | ROTATE | LIFT | FC_CLEAR | FC_BLOCKED |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L4 | C0 Numbered RGB only | D4 | 125 | 111 | 20 | 16.00% | 0.00% | 0.00% | 0.00% | 80.00% | 0.00% |
| L4 | C0 Numbered RGB only | D8 | 125 | 97 | 20 | 16.00% | 0.00% | 0.00% | 0.00% | 80.00% | 0.00% |
| L4 | C1 R numeric | D4 | 125 | 53 | 15 | 12.00% | 0.00% | 0.00% | 0.00% | 60.00% | 0.00% |
| L4 | C1 R numeric | D8 | 125 | 61 | 19 | 15.20% | 0.00% | 0.00% | 0.00% | 76.00% | 0.00% |
| L4 | C2 R qualitative | D4 | 125 | 20 | 11 | 8.80% | 0.00% | 0.00% | 0.00% | 44.00% | 0.00% |
| L4 | C2 R qualitative | D8 | 125 | 37 | 14 | 11.20% | 0.00% | 0.00% | 0.00% | 56.00% | 0.00% |
| L4 | C3 F numeric | D4 | 125 | 10 | 2 | 1.60% | 0.00% | 0.00% | 0.00% | 8.00% | 0.00% |
| L4 | C3 F numeric | D8 | 125 | 43 | 18 | 14.40% | 0.00% | 0.00% | 0.00% | 72.00% | 0.00% |
| L4 | C4 F qualitative | D4 | 125 | 3 | 2 | 1.60% | 0.00% | 0.00% | 0.00% | 8.00% | 0.00% |
| L4 | C4 F qualitative | D8 | 125 | 23 | 10 | 8.00% | 0.00% | 0.00% | 0.00% | 40.00% | 0.00% |
| L4 | C5 M numeric | D4 | 125 | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| L4 | C5 M numeric | D8 | 125 | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| L4 | C6 M qualitative | D4 | 125 | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| L4 | C6 M qualitative | D8 | 125 | 3 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## L4 replay

- unique_simulator_artifacts: 24
- simulator_executed_runs: 461
- action_valid_runs: 131
- collision_runs: 0
- boundary_violation_runs: 0
- action_distribution: {'RETRIEVE': 461, 'TRANSLATE': 1289}
- all_artifacts_direct_stage_execution: True
- all_artifact_baselines_match_preparation: True
- isaac_sim_versions: ['5.1.0-rc.19+release.26219.9c81211b.gl']

## Paired comparisons against C0

Exact two-sided McNemar p-values are uncorrected exploratory values. The five stochastic repeats within a scene are not independent, so these p-values must not be read as confirmatory scene-level inference.

| Level | Condition | Res. | n | C0 | Condition | Delta | p |
|---|---|---:|---:|---:|---:|---:|---:|
| L4 | C1 R numeric | D4 | 125 | 16.00% | 12.00% | -4.00 pp | 0.0625 |
| L4 | C1 R numeric | D8 | 125 | 16.00% | 15.20% | -0.80 pp | 1 |
| L4 | C2 R qualitative | D4 | 125 | 16.00% | 8.80% | -7.20 pp | 0.00390625 |
| L4 | C2 R qualitative | D8 | 125 | 16.00% | 11.20% | -4.80 pp | 0.03125 |
| L4 | C3 F numeric | D4 | 125 | 16.00% | 1.60% | -14.40 pp | 7.62939e-06 |
| L4 | C3 F numeric | D8 | 125 | 16.00% | 14.40% | -1.60 pp | 0.5 |
| L4 | C4 F qualitative | D4 | 125 | 16.00% | 1.60% | -14.40 pp | 7.62939e-06 |
| L4 | C4 F qualitative | D8 | 125 | 16.00% | 8.00% | -8.00 pp | 0.00195312 |
| L4 | C5 M numeric | D4 | 125 | 16.00% | 0.00% | -16.00 pp | 1.90735e-06 |
| L4 | C5 M numeric | D8 | 125 | 16.00% | 0.00% | -16.00 pp | 1.90735e-06 |
| L4 | C6 M qualitative | D4 | 125 | 16.00% | 0.00% | -16.00 pp | 1.90735e-06 |
| L4 | C6 M qualitative | D8 | 125 | 16.00% | 0.00% | -16.00 pp | 1.90735e-06 |
