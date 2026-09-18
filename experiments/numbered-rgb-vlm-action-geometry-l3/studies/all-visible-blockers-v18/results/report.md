# L3 all-visible-blockers rerun v18

25 scenes × 12 conditions × 5 seeds = 1,500 independent calls. The model must return blocker_ids as the exhaustive ascending set of visible causal blockers; scoring uses exact set match.

Dataset limitation: 4 scenes have zero visible blockers and 21 have exactly one; no scene has two or more visible GT blockers. This run therefore measures exhaustive-output prompting and exact-set formatting on empty/singleton sets, not true multi-blocker recall.

Completion: 1500/1500; parse errors: 0; semantic validation failures: 266; infrastructure errors: 0

Overall reason: 699/1500 (46.6%); blocker set exact: 778/1500 (51.9%); joint: 440/1500 (29.3%); micro F1: 63.3%.

| Condition | Reason /125 | Set exact /125 | Joint /125 | Set accuracy | Micro P/R/F1 |
|---|---:|---:|---:|---:|---:|
| C0 | 73 | 81 | 51 | 64.8% | 63.6% / 86.7% / 73.4% |
| C1 | 65 | 85 | 51 | 68.0% | 61.3% / 93.3% / 74.0% |
| C2_D4 | 66 | 88 | 50 | 70.4% | 59.4% / 99.0% / 74.3% |
| C2_D8 | 52 | 86 | 42 | 68.8% | 56.5% / 100.0% / 72.2% |
| C3_D4 | 57 | 57 | 32 | 45.6% | 44.7% / 88.6% / 59.4% |
| C3_D8 | 45 | 64 | 25 | 51.2% | 46.3% / 90.5% / 61.3% |
| C4_D4 | 63 | 48 | 38 | 38.4% | 41.6% / 99.0% / 58.6% |
| C4_D8 | 54 | 53 | 29 | 42.4% | 42.9% / 100.0% / 60.0% |
| C5_D4 | 60 | 66 | 37 | 52.8% | 52.4% / 93.3% / 67.1% |
| C5_D8 | 58 | 63 | 33 | 50.4% | 44.1% / 95.2% / 60.2% |
| C6_D4 | 57 | 45 | 29 | 36.0% | 39.9% / 96.2% / 56.4% |
| C6_D8 | 49 | 42 | 23 | 33.6% | 36.8% / 95.2% / 53.1% |
