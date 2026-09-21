from pathlib import Path

from scripts.build_leaderboard import build
from scripts.validate_results import validate_repository


def test_repository_is_valid() -> None:
    assert validate_repository(Path.cwd())


def test_leaderboard_build(tmp_path: Path) -> None:
    parquet_path, json_path = build(Path.cwd(), tmp_path)
    assert parquet_path.is_file()
    assert json_path.is_file()
