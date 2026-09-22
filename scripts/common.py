"""Shared loading and flattening helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def result_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in (root / "results").glob("*/*/*.json")
        if path.name != "model_meta.json"
    )


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def safe_model_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def non_reasoning_suite_metrics(raw_path: Path) -> dict[str, float | int] | None:
    """Compute the English suite metrics from successful raw predictions.

    ECE is nonlinear, so it must be recomputed from the row-level confidence
    values after excluding the reasoning family; it cannot be derived from
    the overall and reasoning-family scalar ECE values.
    """

    latest_by_row_id: dict[str, dict[str, Any]] = {}
    with raw_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("raw prediction must be an object")
            latest_by_row_id[str(record["row_id"])] = record

    records = [
        record
        for record in latest_by_row_id.values()
        if record.get("status") == "ok"
        and _record_dimension(record, "family") != "reasoning"
        and isinstance(record.get("scored"), dict)
        and "negative_log_likelihood" in record
        and "latency_seconds" in record
    ]
    if not records:
        return None

    bins = 15
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    correctness_sums = [0.0] * bins
    for record in records:
        scored = record["scored"]
        probabilities = scored["probabilities"]
        confidence = max(float(value) for value in probabilities)
        bin_index = min(int(confidence * bins), bins - 1)
        counts[bin_index] += 1
        confidence_sums[bin_index] += confidence
        correctness_sums[bin_index] += float(bool(scored["correct"]))
    ece = 0.0
    total = len(records)
    for count, confidence_sum, correctness_sum in zip(
        counts, confidence_sums, correctness_sums, strict=True
    ):
        if count:
            ece += (count / total) * abs(
                correctness_sum / count - confidence_sum / count
            )

    return {
        "rows": total,
        "accuracy": sum(bool(record["scored"]["correct"]) for record in records)
        / total,
        "mean_negative_log_likelihood": sum(
            float(record["negative_log_likelihood"]) for record in records
        )
        / total,
        "expected_calibration_error": ece,
        "mean_latency_seconds": sum(
            float(record["latency_seconds"]) for record in records
        )
        / total,
    }


def _record_dimension(record: dict[str, Any], name: str) -> str | None:
    value = record.get(name)
    example = record.get("example")
    if value is None and isinstance(example, dict):
        value = example.get(name)
    return str(value) if value is not None else None


def leaderboard_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in result_paths(root):
        result = load_object(path)
        model = result["model"]
        artifact = result.get("artifact")
        for view, metrics in result["views"].items():
            view_kind, separator, view_name = str(view).partition(":")
            if not separator:
                view_kind = "benchmark"
                view_name = str(view)
            successful_rows = int(metrics["rows"])
            is_overall = view == "overall"
            rows.append(
                {
                    "model": model["name"],
                    "revision": model["revision"],
                    "model_type": model.get("model_type"),
                    "model_url": model.get("url"),
                    "adapter": model["adapter"],
                    "probability_source": model["probability_source"],
                    "open_weights": model.get("open_weights"),
                    "parameter_count": model.get("parameter_count"),
                    "benchmark": result["benchmark_name"],
                    "benchmark_version": result["benchmark_version"],
                    "dataset_revision": result["dataset_revision"],
                    "view": view,
                    "view_kind": view_kind,
                    "view_name": view_name,
                    "requested_rows": result["requested_rows"] if is_overall else None,
                    "successful_rows": successful_rows,
                    "unsupported_rows": result["unsupported_rows"] if is_overall else None,
                    "error_rows": result["error_rows"] if is_overall else None,
                    "coverage": result["coverage"] if is_overall else None,
                    "primary_accuracy": result["primary_accuracy"] if is_overall else None,
                    "supported_accuracy": metrics.get("accuracy"),
                    "mean_negative_log_likelihood": metrics.get(
                        "mean_negative_log_likelihood"
                    ),
                    "expected_calibration_error": metrics.get(
                        "expected_calibration_error"
                    ),
                    "mean_latency_seconds": metrics.get("mean_latency_seconds"),
                    "artifact_uri": artifact["uri"] if artifact else None,
                    "artifact_manifest_sha256": (
                        artifact["manifest_sha256"] if artifact else None
                    ),
                    "result_path": str(path.relative_to(root)),
                }
            )
    return rows
