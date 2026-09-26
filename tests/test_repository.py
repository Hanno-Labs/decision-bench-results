import json
from pathlib import Path

import pytest

from scripts.build_leaderboard import build
from scripts.common import (
    leaderboard_rows,
    load_object,
    non_reasoning_suite_metrics,
    result_paths,
    safe_model_name,
)
from scripts.validate_results import validate_repository


def test_repository_is_valid() -> None:
    assert validate_repository(Path.cwd())


def test_leaderboard_build(tmp_path: Path) -> None:
    parquet_path, json_path = build(Path.cwd(), tmp_path)
    assert parquet_path.is_file()
    assert json_path.is_file()


def test_result_without_artifact_is_valid_and_buildable(tmp_path: Path) -> None:
    source = load_object(result_paths(Path.cwd())[0])
    source.pop("artifact")
    model = source["model"]
    result_dir = tmp_path / "results" / safe_model_name(model["name"]) / model["revision"]
    result_dir.mkdir(parents=True)
    (result_dir / "model_meta.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n"
    )
    (result_dir / "DecisionBench.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n"
    )
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "result.schema.json").write_text(
        (Path.cwd() / "schemas" / "result.schema.json").read_text()
    )

    assert validate_repository(tmp_path)
    rows = leaderboard_rows(tmp_path)
    assert rows
    assert all(row["model_type"] == model.get("model_type") for row in rows)
    assert all(row["artifact_uri"] is None for row in rows)
    assert all(row["artifact_manifest_sha256"] is None for row in rows)


def test_compact_result_coexists_with_untagged_result(tmp_path: Path) -> None:
    source = load_object(result_paths(Path.cwd())[0])
    model = source["model"]
    result_dir = tmp_path / "results" / safe_model_name(model["name"]) / model["revision"]
    result_dir.mkdir(parents=True)
    (result_dir / "model_meta.json").write_text(json.dumps(model))
    (result_dir / "DecisionBench.json").write_text(json.dumps(source))
    compact = {**source, "tags": ["compact"]}
    (result_dir / "DecisionBench--compact.json").write_text(json.dumps(compact))
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "result.schema.json").write_text(
        (Path.cwd() / "schemas" / "result.schema.json").read_text()
    )

    assert len(validate_repository(tmp_path)) == 2
    assert {row["tags"] for row in leaderboard_rows(tmp_path)} == {"", "compact"}
    (result_dir / "duplicate.json").write_text(json.dumps(compact))
    with pytest.raises(ValueError, match="duplicate result identity"):
        validate_repository(tmp_path)


def test_non_reasoning_suite_metrics_recompute_ece_from_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_id": "row-1",
                        "status": "ok",
                        "family": "technical",
                        "latency_seconds": 0.1,
                        "negative_log_likelihood": 0.1,
                        "scored": {"correct": True, "probabilities": [0.9, 0.1]},
                    }
                ),
                json.dumps(
                    {
                        "row_id": "row-2",
                        "status": "ok",
                        "family": "reasoning",
                        "latency_seconds": 0.2,
                        "negative_log_likelihood": 0.2,
                        "scored": {"correct": False, "probabilities": [0.6, 0.4]},
                    }
                ),
                json.dumps(
                    {
                        "row_id": "row-3",
                        "status": "ok",
                        "family": "legal",
                        "latency_seconds": 0.3,
                        "negative_log_likelihood": 0.3,
                        "scored": {"correct": False, "probabilities": [0.6, 0.4]},
                    }
                ),
            ]
        )
        + "\n"
    )

    metrics = non_reasoning_suite_metrics(raw)

    assert metrics is not None
    assert metrics["rows"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["expected_calibration_error"] == 0.35
