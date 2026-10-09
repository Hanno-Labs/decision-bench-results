"""Add Inspect input/target columns to frozen rows without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_REPO = "Hanno-Labs/decision-bench"
INPUT_REVISION = "071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb"
INPUT_FILE = "data/eval-00000-of-00001.parquet"
INPUT_SHA256 = "d9dd5be8e0944a7ceadfcac6cf8f9d17aab256093acff23c6de6a037ad6e55f6"


def inspect_fields(row: dict[str, Any]) -> tuple[str, str]:
    candidates = json.loads(row["candidates_json"])
    candidate_ids = [candidate["id"] for candidate in candidates]
    gold = row["gold_candidate_id"]
    if len(candidate_ids) != len(set(candidate_ids)) or gold not in candidate_ids:
        raise ValueError(f"{row['row_id']}: invalid candidate IDs or target")
    payload = {
        key: row[key]
        for key in ("row_id", "task_name", "primitive", "family", "domain", "instruction")
    }
    payload["state"] = json.loads(row["state_json"])
    payload["candidates"] = [
        {
            "id": candidate["id"],
            "label": candidate["label"],
            "description": candidate.get("description"),
            "ordinal_value": (
                float(candidate["ordinal_value"])
                if candidate.get("ordinal_value") is not None
                else None
            ),
        }
        for candidate in candidates
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), "dbid" + gold.encode().hex()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    source: Path,
    destination: Path,
    *,
    expected_rows: int = 23900,
    expected_source_sha256: str | None = INPUT_SHA256,
) -> dict[str, Any]:
    source_sha256 = sha256_file(source)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("source hash does not match the frozen input revision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    seen: set[str] = set()
    try:
        with pq.ParquetFile(source) as parquet:
            if "input" in parquet.schema_arrow.names or "target" in parquet.schema_arrow.names:
                raise ValueError("source already contains Inspect fields")
            schema = parquet.schema_arrow.append(pa.field("input", pa.string())).append(
                pa.field("target", pa.string())
            )
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                for batch in parquet.iter_batches(batch_size=256):
                    inputs: list[str] = []
                    targets: list[str] = []
                    for row in batch.to_pylist():
                        row_id = row["row_id"]
                        if row_id in seen:
                            raise ValueError(f"duplicate row ID: {row_id}")
                        seen.add(row_id)
                        model_input, target = inspect_fields(row)
                        inputs.append(model_input)
                        targets.append(target)
                    table = pa.Table.from_batches([batch])
                    table = table.append_column("input", pa.array(inputs, type=pa.string()))
                    table = table.append_column("target", pa.array(targets, type=pa.string()))
                    writer.write_table(table)
        if len(seen) != expected_rows:
            raise ValueError(f"expected {expected_rows} rows, found {len(seen)}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": "decisionbench-inspect-input-v1",
        "source_repo": INPUT_REPO,
        "source_revision": INPUT_REVISION,
        "source_file": INPUT_FILE,
        "source_sha256": source_sha256,
        "prepared_sha256": sha256_file(destination),
        "rows": len(seen),
        "original_columns_preserved": True,
        "target_encoding": "dbid + lowercase UTF-8 hex of candidate ID",
        "model_evaluation_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
