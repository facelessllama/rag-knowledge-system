# Generator comparison — 2026-07-21

Contexts: `eval/mixed_corpus/generator_ab_postfix_v3_contexts.json` (162 cases, git `82df6043`)
Judge: `qwen2.5:7b` (local, never one of the compared models)

## All cases (n=162)

| model | correct | refused_despite_context | avg tokens |
|---|---|---|---|
| qwen2.5:7b | 77/162 (47.5%) | 4/162 (2.5%) | 1565 |
| deepseek-v4-flash | 120/162 (74.1%) | 1/162 (0.6%) | 1806 |
| deepseek-v4-pro | 112/162 (69.1%) | 0/162 (0.0%) | 1870 |

### Pairwise (2x2 correctness contingency)

#### qwen2.5:7b vs deepseek-v4-flash

| | deepseek-v4-flash correct | deepseek-v4-flash wrong |
|---|---|---|
| **qwen2.5:7b correct** | 70 | 7 |
| **qwen2.5:7b wrong** | 50 | 35 |

#### qwen2.5:7b vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **qwen2.5:7b correct** | 66 | 11 |
| **qwen2.5:7b wrong** | 46 | 39 |

#### deepseek-v4-flash vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **deepseek-v4-flash correct** | 102 | 18 |
| **deepseek-v4-flash wrong** | 10 | 32 |
