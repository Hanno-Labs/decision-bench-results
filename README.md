---
benchmark: decision-bench
type: evaluation
submission_name: DecisionBench
---

# DecisionBench Results

This repository is the reviewed, append-only result registry for
[DecisionBench](https://github.com/Hanno-Labs/decision-bench).

| Reference | |
|---|---|
| 📈 **[Leaderboard]** | Browse reviewed model results |
| 📚 **[DecisionBench]** | Run evaluations and load result records |
| 🧾 **[Submission guide]** | Validate and submit a new result |
| 🐛 **[Issues]** | Report benchmark or result problems |

[Leaderboard]: https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard
[DecisionBench]: https://github.com/Hanno-Labs/decision-bench
[Submission guide]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/contributing/submitting_results/
[Issues]: https://github.com/Hanno-Labs/decision-bench/issues

## Layout

```text
results/<model-name>/<immutable-revision>/
├── model_meta.json
└── DecisionBench.json
```

Each compact record contains the reviewed metrics and immutable model and dataset identities. A
submission may also point to complete row-level artifacts in durable storage. Unsupported and error
rows count as misses in the primary leaderboard score.

Each model declares its own reviewed `model_type`: `decision-model` for checkpoints trained across
the benchmark's `noul`, `choice`, and `score` primitives with a native decision output;
`language-model` for text-generating models; or `classifier` for fixed-purpose class, relevance, or
scalar scorers. Architecture names and API access do not determine this field.

Model names use an `owner/name` identity. For serving recipes without their own Hugging Face
checkpoint, `owner` credits the upstream recipe project, while `model_meta.json`'s `url` and
`revision` identify the exact base weights evaluated. A recipe's name must not imply that its
author published separate model weights.

| Recipe result | Upstream project | Evaluated base checkpoint |
|---|---|---|
| `Octalab-Inc/jqv` | [Octalab-Inc/jqv](https://github.com/Octalab-Inc/jqv) | `Qwen/Qwen3-32B` |
| `zhengxuyu/litjev` | [zhengxuyu/litjev](https://github.com/zhengxuyu/litjev) | `Qwen/Qwen3.8-27B` |
| `kshetrajna12/reflex-27b` | [kshetrajna12/reflex](https://github.com/kshetrajna12/reflex) | `Qwen/Qwen3.8-27B` |
| `featherless-ai/simplejev-qwen3.6-35b-a3b` | [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | `Qwen/Qwen3.6-35B-A3B` |
| `featherless-ai/simplejev-qwen3.8-27b` | [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) | `Qwen/Qwen3.8-27B` |
