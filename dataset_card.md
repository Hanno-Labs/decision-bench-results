---
license: cc0-1.0
pretty_name: DecisionBench Results
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/decisionbench.parquet
  - config_name: results
    data_files:
      - split: train
        path: leaderboard.parquet
---

# DecisionBench Results

Reviewed results from [Hanno-Labs/decision-bench-results](https://github.com/Hanno-Labs/decision-bench-results), with the frozen evaluation inputs used by the native leaderboard registration.

The `default` configuration contains all 23,900 evaluation rows from [Hanno-Labs/decision-bench at revision 071b7b2](https://huggingface.co/datasets/Hanno-Labs/decision-bench/tree/071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb). Its original columns are preserved. The added `input` column contains the model-visible instruction, state, and ordered candidates as JSON, without gold labels, gold probabilities, or source annotations. The `target` column encodes the gold candidate ID as `dbid` followed by lowercase UTF-8 hex. This prevents Inspect's exact-match normalization from merging distinct candidate IDs.

Use the [DecisionBench native model adapters](https://github.com/Hanno-Labs/decision-bench) to obtain the full candidate probability vector. The selected candidate uses the same encoded ID for Inspect's built-in `generate` and `exact` components. Primary accuracy counts unsupported or failed rows as incorrect. Soft-label negative log-likelihood, 15-bin equal-width top-label ECE, coverage, and the original candidate IDs remain in the evaluator's canonical artifacts.

The `results` configuration, `leaderboard.json`, and `leaderboard.parquet` contain the reviewed result records. Existing scores are preserved; adding the Inspect registration does not change their execution history or imply a Hugging Face verified badge. Each record retains its original model, input revision, adapter, and artifact provenance.

`input_manifest.json` records the frozen input file hash and generated data-view hash. Native Hugging Face leaderboard activation is subject to Hugging Face's benchmark allowlisting.
