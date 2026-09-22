"""Import a previously verified DecisionBench run into the compact registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.common import load_object, non_reasoning_suite_metrics, safe_model_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def normalized_views(summary: dict[str, Any]) -> dict[str, dict[str, object]]:
    metrics = summary["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("summary metrics must be an object")
    views: dict[str, dict[str, object]] = {}
    for name, value in metrics.items():
        if not isinstance(value, dict):
            raise TypeError(f"summary metric view must be an object: {name}")
        views[str(name)] = {
            "rows": int(value["rows"]),
            "accuracy": optional_float(value.get("accuracy")),
            "mean_negative_log_likelihood": optional_float(
                value.get("mean_negative_log_likelihood")
            ),
            "expected_calibration_error": optional_float(
                value.get("expected_calibration_error")
            ),
            "mean_latency_seconds": optional_float(value.get("mean_latency_seconds")),
        }
    return views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-type",
        choices=("decision-model", "language-model", "classifier"),
        required=True,
    )
    parser.add_argument("--model-url")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--probability-source", required=True)
    parser.add_argument("--artifact-uri")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--unsupported-rows", type=int, required=True)
    parser.add_argument("--error-rows", type=int, required=True)
    parser.add_argument("--parameter-count", type=int)
    parser.add_argument("--open-weights", action="store_true")
    parser.add_argument("--submitted-at")
    args = parser.parse_args()

    root = Path.cwd()
    summary = load_object(args.summary)
    manifest = load_object(args.manifest)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("manifest files must be an object")
    if files.get("summary.json") != sha256(args.summary):
        raise ValueError("summary hash does not match manifest")
    if files.get("raw.jsonl") != summary.get("raw_sha256"):
        raise ValueError("raw hash differs between summary and manifest")

    requested_rows = int(summary["requested_rows"])
    successful_rows = int(summary["successful_rows"])
    if successful_rows + args.unsupported_rows + args.error_rows != requested_rows:
        raise ValueError("classified row counts do not equal requested rows")
    overall = normalized_views(summary)["overall"]
    supported_accuracy = float(overall["accuracy"])
    coverage = successful_rows / requested_rows
    model = {
        "name": args.model_id,
        "revision": args.model_revision,
        "model_type": args.model_type,
        "url": args.model_url,
        "adapter": args.adapter,
        "probability_source": args.probability_source,
        "open_weights": args.open_weights,
        "parameter_count": args.parameter_count,
    }
    views = normalized_views(summary)
    raw_path = args.summary.parent / "raw.jsonl"
    suite_metrics = non_reasoning_suite_metrics(raw_path) if raw_path.is_file() else None
    if suite_metrics is not None:
        views["suite:DecisionBench(eng, v1)"] = suite_metrics

    record = {
        "schema_version": "decision-bench-result-v1",
        "benchmark_name": "DecisionBench",
        "benchmark_version": "1.0",
        "dataset_repo": "Hanno-Labs/decision-bench",
        "dataset_revision": args.dataset_revision,
        "task_spec_sha256": summary["task_spec_sha256"],
        "model": model,
        "requested_rows": requested_rows,
        "successful_rows": successful_rows,
        "unsupported_rows": args.unsupported_rows,
        "error_rows": args.error_rows,
        "coverage": coverage,
        "primary_accuracy": supported_accuracy * coverage,
        "supported_accuracy": supported_accuracy,
        "mean_negative_log_likelihood": overall["mean_negative_log_likelihood"],
        "expected_calibration_error": overall["expected_calibration_error"],
        "mean_latency_seconds": overall["mean_latency_seconds"],
        "views": views,
        "submitted_at": args.submitted_at or datetime.now(UTC).isoformat(),
    }
    if args.artifact_uri:
        record["artifact"] = {
            "uri": args.artifact_uri,
            "manifest_sha256": sha256(args.manifest),
            "summary_sha256": sha256(args.summary),
            "raw_sha256": files["raw.jsonl"],
        }
    output_dir = root / "results" / safe_model_name(args.model_id) / args.model_revision
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_meta.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "DecisionBench.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
