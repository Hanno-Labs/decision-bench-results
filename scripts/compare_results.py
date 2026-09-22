"""Render a trusted comparison for DecisionBench result pull requests."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from scripts.common import load_object, result_paths
from scripts.validate_results import validate_result_file

COMMENT_MARKER = "<!-- decision-bench-results-comparison -->"
MAX_CHANGED_RESULTS = 20
MAX_RESULT_BYTES = 2_000_000
MAX_COMMENT_CHARS = 60_000
DEFAULT_REFERENCES = (
    ("typesafe/jev-1.13", "openrouter-service-snapshot-2026-09-21"),
    (
        "openai/gpt-5.6-luna",
        "openrouter-service-snapshot-2026-09-21-reasoning-aware-v1",
    ),
)
REFERENCE_LABELS = {DEFAULT_REFERENCES[0]: "Jev", DEFAULT_REFERENCES[1]: "Luna"}

Result = dict[str, Any]
Reference = tuple[str, str]


def load_results(root: Path) -> list[tuple[Path, Result]]:
    return [(path, load_object(path)) for path in result_paths(root.resolve())]


def find_exact_reference(
    results: Sequence[tuple[Path, Result]], name: str, revision: str
) -> Result:
    matches = [
        result
        for _, result in results
        if result["model"]["name"] == name and result["model"]["revision"] == revision
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one reference result for {name}@{revision}; found {len(matches)}"
        )
    return matches[0]


def _changed_entries(path: Path | None, pr_root: Path) -> list[tuple[str, str]]:
    if path is None:
        return [("modified", str(item.relative_to(pr_root))) for item in result_paths(pr_root)]

    entries: list[tuple[str, str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        status, separator, filename = line.partition("\t")
        if not separator:
            status = "modified"
            filename = line
        entries.append((status, filename))
    return entries


def select_changed_results(
    pr_root: Path, base_root: Path, changed_files: Path | None
) -> list[Result]:
    """Load changed result records while treating the PR checkout as hostile data."""
    pr_root = pr_root.resolve()
    schema = load_object(base_root.resolve() / "schemas" / "result.schema.json")
    validator = Draft202012Validator(schema)
    selected: list[Result] = []
    seen: set[Path] = set()
    identities: set[tuple[str, str, str, str, str]] = set()

    for status, filename in _changed_entries(changed_files, pr_root):
        if status == "removed":
            continue
        relative = Path(filename)
        if (
            len(relative.parts) != 4
            or relative.parts[0] != "results"
            or relative.name == "model_meta.json"
            or relative.suffix != ".json"
        ):
            continue
        candidate = (pr_root / relative).resolve()
        if not candidate.is_relative_to(pr_root):
            raise ValueError(f"changed result path escapes PR checkout: {filename}")
        if candidate in seen:
            continue
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size > MAX_RESULT_BYTES:
            raise ValueError(f"changed result is larger than {MAX_RESULT_BYTES} bytes: {filename}")
        seen.add(candidate)
        selected.append(validate_result_file(pr_root, candidate, validator, identities))
        if len(selected) > MAX_CHANGED_RESULTS:
            raise ValueError(f"more than {MAX_CHANGED_RESULTS} changed result records")
    return selected


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("`", "&#96;")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def _percent(value: object) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.2%}"


def _decimal(value: object) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.3f}"


def _delta(value: object, reference: object, *, percentage_points: bool) -> str:
    number = _number(value)
    baseline = _number(reference)
    if number is None or baseline is None:
        return "n/a"
    delta = number - baseline
    suffix = " pp" if percentage_points else ""
    scale = 100.0 if percentage_points else 1.0
    return f"{delta * scale:+.2f}{suffix}" if percentage_points else f"{delta:+.3f}"


def _reference_label(reference: Reference) -> str:
    return REFERENCE_LABELS.get(reference, reference[0])


def _view_rows(
    submissions: Sequence[Result],
    references: Sequence[tuple[Reference, Result]],
    prefix: str,
) -> list[str]:
    names = sorted(
        {
            str(view_name)
            for submission in submissions
            for view_name in submission["views"]
            if str(view_name).startswith(prefix)
        }
    )
    if not names:
        return []

    lines = [
        "| Model | View | Acc | Acc Δ Jev | Acc Δ Luna | ECE | ECE Δ Jev | "
        "ECE Δ Luna | NLL | NLL Δ Jev | NLL Δ Luna | Rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for submission in sorted(
        submissions, key=lambda item: (item["model"]["name"], item["model"]["revision"])
    ):
        for view_name in names:
            view = submission["views"].get(view_name)
            if view is None:
                continue
            reference_views = [reference["views"].get(view_name, {}) for _, reference in references]
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(submission["model"]["name"]),
                        _escape(view_name.removeprefix(prefix)),
                        _percent(view.get("accuracy")),
                        *[
                            _delta(
                                view.get("accuracy"),
                                reference_view.get("accuracy"),
                                percentage_points=True,
                            )
                            for reference_view in reference_views
                        ],
                        _decimal(view.get("expected_calibration_error")),
                        *[
                            _delta(
                                view.get("expected_calibration_error"),
                                reference_view.get("expected_calibration_error"),
                                percentage_points=False,
                            )
                            for reference_view in reference_views
                        ],
                        _decimal(view.get("mean_negative_log_likelihood")),
                        *[
                            _delta(
                                view.get("mean_negative_log_likelihood"),
                                reference_view.get("mean_negative_log_likelihood"),
                                percentage_points=False,
                            )
                            for reference_view in reference_views
                        ],
                        str(view["rows"]),
                    ]
                )
                + " |"
            )
    return lines


def render_report(
    submissions: Sequence[Result], references: Sequence[tuple[Reference, Result]]
) -> str:
    lines = [COMMENT_MARKER, "## DecisionBench result comparison", ""]
    if not submissions:
        lines.append(
            "No added or modified DecisionBench result record was found in this pull request."
        )
        return "\n".join(lines) + "\n"

    reference_names = " and ".join(
        f"**{_reference_label(key)}** (`{_escape(key[0])}`)" for key, _ in references
    )
    lines.extend(
        [
            "Submitted results are compared with the pinned "
            f"{reference_names} records from `main`.",
            "Accuracy deltas are percentage points; positive is better. For ECE and NLL deltas, "
            "negative is better. Per-view accuracy is calculated on supported rows.",
            "",
            "### Overall",
            "",
            "| Model | Revision | Primary acc | Δ Jev | Δ Luna | Supported acc | Coverage | "
            "ECE | NLL | Errors | Unsupported |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    reference_primary = [reference["primary_accuracy"] for _, reference in references]
    for submission in sorted(
        submissions, key=lambda item: (item["model"]["name"], item["model"]["revision"])
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(submission["model"]["name"]),
                    _escape(submission["model"]["revision"]),
                    _percent(submission["primary_accuracy"]),
                    *[
                        _delta(
                            submission["primary_accuracy"], baseline, percentage_points=True
                        )
                        for baseline in reference_primary
                    ],
                    _percent(submission["supported_accuracy"]),
                    _percent(submission["coverage"]),
                    _decimal(submission.get("expected_calibration_error")),
                    _decimal(submission.get("mean_negative_log_likelihood")),
                    str(submission["error_rows"]),
                    str(submission["unsupported_rows"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Pinned reference baselines</summary>",
            "",
            "| Reference | Revision | Primary acc | Supported acc | Coverage | ECE | NLL | "
            "Errors | Unsupported |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, reference in references:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(f"{_reference_label(key)} ({key[0]})"),
                    _escape(reference["model"]["revision"]),
                    _percent(reference["primary_accuracy"]),
                    _percent(reference["supported_accuracy"]),
                    _percent(reference["coverage"]),
                    _decimal(reference.get("expected_calibration_error")),
                    _decimal(reference.get("mean_negative_log_likelihood")),
                    str(reference["error_rows"]),
                    str(reference["unsupported_rows"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "</details>"])

    for title, prefix in (("Per primitive", "primitive:"), ("Per family", "family:")):
        rows = _view_rows(submissions, references, prefix)
        if rows:
            lines.extend(["", f"### {title}", "", *rows])

    report = "\n".join(lines) + "\n"
    if len(report) <= MAX_COMMENT_CHARS:
        return report
    notice = "\n\n_Comparison truncated to fit the GitHub comment limit._\n"
    boundary = report.rfind("\n", 0, MAX_COMMENT_CHARS - len(notice))
    return report[:boundary] + notice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=Path.cwd())
    parser.add_argument("--pr-root", type=Path, default=Path.cwd())
    parser.add_argument("--changed-files", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result-comparison.md"))
    args = parser.parse_args()
    base_results = load_results(args.base_root)
    references = [
        (key, find_exact_reference(base_results, key[0], key[1])) for key in DEFAULT_REFERENCES
    ]
    submissions = select_changed_results(args.pr_root, args.base_root, args.changed_files)
    args.output.write_text(render_report(submissions, references))


if __name__ == "__main__":
    main()
