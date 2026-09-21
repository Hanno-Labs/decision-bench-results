"""Publish the generated leaderboard mirror to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from scripts.build_leaderboard import build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Hanno-Labs/decision-bench-results")
    parser.add_argument("--repo-type", choices=("dataset", "space"), default="dataset")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = root / "dist"
    build(root, output_dir)
    api = HfApi()
    if args.repo_type == "dataset":
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=output_dir,
        )
    else:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="space",
            path_or_fileobj=output_dir / "leaderboard.parquet",
            path_in_repo="leaderboard.parquet",
        )


if __name__ == "__main__":
    main()
