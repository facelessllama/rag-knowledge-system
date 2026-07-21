# Generator comparison — 2026-07-21

Contexts: `eval/mixed_corpus/generator_ab_contexts.json` (47 cases, git `8c089c7b`)
Judge: `qwen2.5:7b` (local, never one of the compared models)

## All cases (n=47)

| model | correct | refused_despite_context | avg tokens |
|---|---|---|---|
| qwen2.5:7b | 19/47 (40.4%) | 4/47 (8.5%) | 1399 |
| deepseek-v4-flash | 36/47 (76.6%) | 0/47 (0.0%) | 1601 |
| deepseek-v4-pro | 34/47 (72.3%) | 0/47 (0.0%) | 1695 |

### Pairwise (2x2 correctness contingency)

#### qwen2.5:7b vs deepseek-v4-flash

| | deepseek-v4-flash correct | deepseek-v4-flash wrong |
|---|---|---|
| **qwen2.5:7b correct** | 18 | 1 |
| **qwen2.5:7b wrong** | 18 | 10 |

#### qwen2.5:7b vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **qwen2.5:7b correct** | 17 | 2 |
| **qwen2.5:7b wrong** | 17 | 11 |

#### deepseek-v4-flash vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **deepseek-v4-flash correct** | 30 | 6 |
| **deepseek-v4-flash wrong** | 4 | 7 |

## Subset: meta.saved_by_change (n=18)

| model | correct | refused_despite_context | avg tokens |
|---|---|---|---|
| qwen2.5:7b | 7/18 (38.9%) | 1/18 (5.6%) | 1506 |
| deepseek-v4-flash | 14/18 (77.8%) | 0/18 (0.0%) | 1710 |
| deepseek-v4-pro | 12/18 (66.7%) | 0/18 (0.0%) | 1843 |

### Pairwise (2x2 correctness contingency)

#### qwen2.5:7b vs deepseek-v4-flash

| | deepseek-v4-flash correct | deepseek-v4-flash wrong |
|---|---|---|
| **qwen2.5:7b correct** | 7 | 0 |
| **qwen2.5:7b wrong** | 7 | 4 |

#### qwen2.5:7b vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **qwen2.5:7b correct** | 6 | 1 |
| **qwen2.5:7b wrong** | 6 | 5 |

#### deepseek-v4-flash vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **deepseek-v4-flash correct** | 11 | 3 |
| **deepseek-v4-flash wrong** | 1 | 3 |
