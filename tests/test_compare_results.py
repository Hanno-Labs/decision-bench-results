import shutil
from pathlib import Path

import pytest

from scripts.compare_results import (
    COMMENT_MARKER,
    DEFAULT_REFERENCES,
    _delta,
    find_exact_reference,
    load_results,
    render_report,
    select_changed_results,
)


def _copy_result(root: Path, destination: Path, model_dir: str, revision: str) -> str:
    relative = Path("results") / model_dir / revision
    shutil.copytree(root / relative, destination / relative)
    return str(relative / "DecisionBench.json")


def test_default_references_resolve_exactly() -> None:
    results = load_results(Path.cwd())
    references = [
        find_exact_reference(results, model_name, revision)
        for model_name, revision in DEFAULT_REFERENCES
    ]
    assert [reference["model"]["name"] for reference in references] == [
        "typesafe/jev-1.13",
        "openai/gpt-5.6-luna",
    ]


def test_changed_result_is_validated_and_compared(tmp_path: Path) -> None:
    root = Path.cwd()
    pr_root = tmp_path / "pr"
    filename = _copy_result(
        root,
        pr_root,
        "openai__gpt-5.6-luna",
        "openrouter-service-snapshot-2026-09-21-reasoning-aware-v1",
    )
    changed_files = tmp_path / "changed.tsv"
    changed_files.write_text(f"modified\t{filename}\nmodified\tREADME.md\n")

    submissions = select_changed_results(pr_root, root, changed_files)
    base_results = load_results(root)
    references = [
        (key, find_exact_reference(base_results, key[0], key[1]))
        for key in DEFAULT_REFERENCES
    ]
    report = render_report(submissions, references)

    assert len(submissions) == 1
    assert report.startswith(COMMENT_MARKER)
    assert "openai/gpt-5.6-luna" in report
    assert "typesafe/jev-1.13" in report
    assert "**Jev** (`typesafe/jev-1.13`)" in report
    assert "**Luna** (`openai/gpt-5.6-luna`)" in report
    assert "-2.13 pp" in report
    assert "### Per primitive" in report
    assert "### Per family" in report


def test_removed_and_metadata_files_are_ignored(tmp_path: Path) -> None:
    root = Path.cwd()
    pr_root = tmp_path / "pr"
    changed_files = tmp_path / "changed.tsv"
    changed_files.write_text(
        "removed\tresults/example/revision/DecisionBench.json\n"
        "modified\tresults/example/revision/model_meta.json\n"
    )
    assert select_changed_results(pr_root, root, changed_files) == []


def test_missing_model_metadata_is_rejected(tmp_path: Path) -> None:
    root = Path.cwd()
    pr_root = tmp_path / "pr"
    filename = _copy_result(
        root,
        pr_root,
        "typesafe__jev-1.13",
        "openrouter-service-snapshot-2026-09-21",
    )
    (pr_root / filename).with_name("model_meta.json").unlink()
    changed_files = tmp_path / "changed.tsv"
    changed_files.write_text(f"added\t{filename}\n")

    with pytest.raises(FileNotFoundError):
        select_changed_results(pr_root, root, changed_files)


def test_reference_lookup_rejects_missing_revision() -> None:
    with pytest.raises(ValueError, match="found 0"):
        find_exact_reference(load_results(Path.cwd()), "typesafe/jev-1.13", "missing")


def test_delta_handles_null_and_direction() -> None:
    assert _delta(None, 0.5, percentage_points=True) == "n/a"
    assert _delta(0.75, 0.70, percentage_points=True) == "+5.00 pp"
    assert _delta(0.10, 0.20, percentage_points=False) == "-0.100"
