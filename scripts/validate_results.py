"""Validate reviewed result schemas, paths, identities, and metric invariants."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from scripts.common import load_object, result_paths, safe_model_name

ResultIdentity = tuple[str, str, str, str, str]


def validate_result_file(
    root: Path,
    path: Path,
    validator: Draft202012Validator,
    identities: set[ResultIdentity] | None = None,
) -> dict[str, Any]:
    """Validate one result using a caller-provided, trusted schema validator."""
    root = root.resolve()
    path = path.resolve()
    results_root = (root / "results").resolve()
    if not path.is_relative_to(results_root):
        raise ValueError(f"result path escapes results directory: {path}")

    result = load_object(path)
    errors = sorted(validator.iter_errors(result), key=lambda error: list(error.path))
    if errors:
        formatted = "; ".join(error.message for error in errors)
        raise ValueError(f"{path}: {formatted}")

    model = result["model"]
    expected_dir = root / "results" / safe_model_name(model["name"]) / model["revision"]
    if path.parent != expected_dir:
        raise ValueError(f"unexpected result path: {path}; expected under {expected_dir}")
    meta_path = expected_dir / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if load_object(meta_path) != model:
        raise ValueError(f"model metadata mismatch: {meta_path}")

    classified = result["successful_rows"] + result["unsupported_rows"] + result["error_rows"]
    if classified != result["requested_rows"]:
        raise ValueError(f"row counts do not close: {path}")
    coverage = result["successful_rows"] / result["requested_rows"]
    if not math.isclose(result["coverage"], coverage, abs_tol=1e-12):
        raise ValueError(f"coverage mismatch: {path}")
    supported_accuracy = result["supported_accuracy"] or 0.0
    primary_accuracy = supported_accuracy * coverage
    if not math.isclose(result["primary_accuracy"], primary_accuracy, abs_tol=1e-9):
        raise ValueError(f"primary accuracy mismatch: {path}")
    overall = result["views"].get("overall")
    if overall is None or overall["rows"] != result["successful_rows"]:
        raise ValueError(f"overall view row mismatch: {path}")

    identity: ResultIdentity = (
        model["name"],
        model["revision"],
        result["benchmark_name"],
        result["benchmark_version"],
        result["dataset_revision"],
    )
    if identities is not None:
        if identity in identities:
            raise ValueError(f"duplicate result identity: {identity}")
        identities.add(identity)
    return result


def validate_repository(root: Path) -> list[Path]:
    root = root.resolve()
    schema = load_object(root / "schemas" / "result.schema.json")
    validator = Draft202012Validator(schema)
    paths = result_paths(root)
    identities: set[ResultIdentity] = set()
    for path in paths:
        validate_result_file(root, path, validator, identities)
    return paths


def main() -> None:
    paths = validate_repository(Path.cwd())
    print(f"Validated {len(paths)} DecisionBench result records.")


if __name__ == "__main__":
    main()
