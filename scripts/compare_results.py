"""Render a compact comparison for result pull requests."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import leaderboard_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("result-comparison.md"))
    args = parser.parse_args()
    rows = [row for row in leaderboard_rows(Path.cwd()) if row["view"] == "overall"]
    rows.sort(key=lambda row: float(row["primary_accuracy"] or 0.0), reverse=True)
    lines = [
        "## DecisionBench result comparison",
        "",
        "| Model | Primary accuracy | Supported accuracy | Coverage | Errors | Unsupported |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {float(row['primary_accuracy']):.2%} | "
            f"{float(row['supported_accuracy']):.2%} | {float(row['coverage']):.2%} | "
            f"{row['error_rows']} | {row['unsupported_rows']} |"
        )
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
