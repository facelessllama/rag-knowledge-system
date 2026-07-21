# Generator comparison — 2026-07-21

Contexts: `eval/mixed_corpus/generator_ab_postfix_v3_titlefreescoped_contexts.json` (168 cases, git `89e44519`)
Judge: `qwen2.5:7b` (local, never one of the compared models)

## All cases (n=168)

| model | correct | refused_despite_context | avg tokens |
|---|---|---|---|
| qwen2.5:7b | 82/168 (48.8%) | 1/168 (0.6%) | 1118 |
| deepseek-v4-flash | 95/168 (56.5%) | 0/168 (0.0%) | 1230 |
| deepseek-v4-pro | 90/168 (53.6%) | 0/168 (0.0%) | 1295 |

### Pairwise (2x2 correctness contingency)

#### qwen2.5:7b vs deepseek-v4-flash

| | deepseek-v4-flash correct | deepseek-v4-flash wrong |
|---|---|---|
| **qwen2.5:7b correct** | 68 | 14 |
| **qwen2.5:7b wrong** | 27 | 59 |

#### qwen2.5:7b vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **qwen2.5:7b correct** | 64 | 18 |
| **qwen2.5:7b wrong** | 26 | 60 |

#### deepseek-v4-flash vs deepseek-v4-pro

| | deepseek-v4-pro correct | deepseek-v4-pro wrong |
|---|---|---|
| **deepseek-v4-flash correct** | 72 | 23 |
| **deepseek-v4-flash wrong** | 18 | 55 |
