import json
from pathlib import Path

from scripts.build_leaderboard import build
from scripts.common import leaderboard_rows, load_object, result_paths, safe_model_name
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
    assert all(row["artifact_uri"] is None for row in rows)
    assert all(row["artifact_manifest_sha256"] is None for row in rows)
