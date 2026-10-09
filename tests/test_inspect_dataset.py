import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_inspect_dataset import inspect_fields, prepare


def row(row_id: str, gold: str = "A-B") -> dict:
    return {
        "row_id": row_id,
        "task_name": "fixture",
        "primitive": "candidate_selection",
        "family": "fixture",
        "domain": "fixture",
        "instruction": "Choose.",
        "state_json": json.dumps({"text": "候補"}),
        "candidates_json": json.dumps(
            [{"id": "A-B", "label": "One"}, {"id": "ab", "label": "Two"}]
        ),
        "gold_candidate_id": gold,
        "gold_probabilities": [0.7, 0.3] if gold == "A-B" else [0.3, 0.7],
        "source_json": json.dumps({"hidden": "label"}),
    }


def test_view_preserves_original_rows_and_excludes_labels_from_input(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    original = pa.Table.from_pylist([row("one"), row("two", "ab")])
    pq.write_table(original, source)
    destination = tmp_path / "view.parquet"
    manifest = prepare(source, destination, expected_rows=2, expected_source_sha256=None)
    view = pq.read_table(destination)
    assert view.select(original.column_names).equals(original)
    assert manifest["rows"] == 2
    assert manifest["model_evaluation_performed"] is False
    inputs = [json.loads(item) for item in view["input"].to_pylist()]
    assert all("gold_candidate_id" not in item for item in inputs)
    assert all("gold_probabilities" not in item for item in inputs)
    assert all("source" not in item for item in inputs)
    assert inputs[0]["candidates"][0]["description"] is None
    assert len(set(view["target"].to_pylist())) == 2


@pytest.mark.parametrize(
    "rows,count,message",
    [([row("one")], 2, "expected 2 rows"), ([row("one"), row("one")], 2, "duplicate row ID")],
)
def test_invalid_view_is_not_published(
    tmp_path: Path, rows: list, count: int, message: str
) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    destination = tmp_path / "view.parquet"
    destination.write_bytes(b"previous view")
    with pytest.raises(ValueError, match=message):
        prepare(source, destination, expected_rows=count, expected_source_sha256=None)
    assert destination.read_bytes() == b"previous view"


def test_invalid_target_rejected() -> None:
    with pytest.raises(ValueError, match="invalid candidate IDs"):
        inspect_fields(row("one", "unknown"))


def test_frozen_source_hash_is_enforced(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([row("one")]), source)
    with pytest.raises(ValueError, match="source hash"):
        prepare(source, tmp_path / "view.parquet")
