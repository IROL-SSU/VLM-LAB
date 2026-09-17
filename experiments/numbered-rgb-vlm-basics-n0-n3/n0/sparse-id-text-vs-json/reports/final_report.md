# B0-N0 sparse-ID recognition: TEXT vs forced JSON

## Protocol

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Scenes: 60 (L3 20, L4 20, L5 20)
- Repeats: 5 paired seeds per scene and output mode
- Calls: 600, batch size 1
- IDs: scene-specific unique two-digit sparse IDs; matched occlusion scenes share the same IDs
- Primary metric: order-independent exact ID multiset match
- Generation: temperature 0.7, top-p 0.9, max tokens 1024, max model length 8192
- JSON: strict structured output with an unbounded integer array

## Main results

| Mode | ID-set exact | Micro precision | Micro recall | Parse success |
|---|---:|---:|---:|---:|
| TEXT | 100.000% (300/300) | 100.000% | 100.000% | 100.000% |
| JSON forced | 100.000% (300/300) | 100.000% | 100.000% | 100.000% |

## ID-set exact by level

| Level | TEXT | JSON forced |
|---|---:|---:|
| L3 | 100.000% | 100.000% |
| L4 | 100.000% | 100.000% |
| L5 | 100.000% | 100.000% |

## Paired comparison

- Both correct: 300/300
- TEXT only correct: 0/300
- JSON only correct: 0/300
- Neither correct: 0/300
- Same predicted ID multiset: 100.000%
- Exact McNemar p: 1

## Error breakdown

- TEXT: {"correct": 300}
- JSON forced: {"correct": 300}
- Parse failures: 0; length finishes: 0

## Failing scene-mode groups

| Level | Scene | Mode | Exact | Missing IDs | Extra IDs |
|---|---|---|---:|---:|---:|
| — | No failing scenes | — | — | — | — |

## Actual user prompts

Common:

```text
List all integer IDs visible inside the red circular markers in the image.
```

TEXT suffix:

```text
Return only the integer IDs as a comma-separated line.
Do not use brackets, labels, words, or explanations.
```

JSON suffix:

```text
Return only a JSON object with exactly one key named "visible_ids".
The value must be an array of integers.
Do not use Markdown and do not add any other keys or text.
```
