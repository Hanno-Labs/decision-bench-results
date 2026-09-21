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
[Submission guide]: https://hanno-labs.github.io/decision-bench/contributing/submitting_results/
[Issues]: https://github.com/Hanno-Labs/decision-bench/issues

## Layout

```text
results/<model-name>/<immutable-revision>/
├── model_meta.json
└── DecisionBench.json
```

Each compact record points to complete row-level artifacts in durable storage. Unsupported and
error rows count as misses in the primary leaderboard score.
