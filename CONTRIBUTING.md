# Contributing results

Run DecisionBench with the reference model adapter, retain the complete content-addressed run
artifact, then stage the compact record:

```bash
decision-bench stage-result /path/to/run /path/to/decision-bench-results \
  --model-id org/model \
  --model-revision immutable-revision \
  --artifact-uri hf://buckets/org/bucket/run \
  --dataset-revision immutable-dataset-revision \
  --adapter reference-adapter \
  --probability-source complete-candidate-distribution
```

Before opening a pull request, run `uv run validate-results` and `uv run pytest`. A submission must
identify immutable model and dataset revisions, use the reference adapter, preserve all row-level
artifacts, and disclose benchmark contamination.
