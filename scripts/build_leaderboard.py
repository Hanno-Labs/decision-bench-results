"""Build the generated leaderboard mirror from reviewed JSON records."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.common import leaderboard_rows
from scripts.validate_results import validate_repository


def build(root: Path, output_dir: Path) -> tuple[Path, Path]:
    validate_repository(root)
    rows = leaderboard_rows(root)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "leaderboard.parquet"
    json_path = output_dir / "leaderboard.json"
    frame.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(root / "eval.yaml", output_dir / "eval.yaml")
    shutil.copyfile(root / "dataset_card.md", output_dir / "README.md")
    return parquet_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    parquet_path, json_path = build(Path.cwd(), args.output_dir)
    print(f"Built {parquet_path} and {json_path}")


if __name__ == "__main__":
    main()
