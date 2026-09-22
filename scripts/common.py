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
