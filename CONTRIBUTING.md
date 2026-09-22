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

For every result pull request, CI validates the submitted record and posts an automated comparison
against the pinned `typesafe/jev-1.13` and `openai/gpt-5.6-luna` reference results. The comparison
covers overall accuracy, coverage, calibration, and per-family and per-primitive metrics. It is a
review aid, not a server-side rerun: maintainers may inspect the linked row-level artifact and ask
for additional reproducibility evidence before accepting a result.
