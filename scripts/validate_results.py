"""Validate reviewed result schemas, paths, identities, and metric invariants."""

from __future__ import annotations

import math
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.common import load_object, result_paths, safe_model_name


def validate_repository(root: Path) -> list[Path]:
    schema = load_object(root / "schemas" / "result.schema.json")
    validator = Draft202012Validator(schema)
    paths = result_paths(root)
    identities: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
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

        classified = (
            result["successful_rows"] + result["unsupported_rows"] + result["error_rows"]
        )
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

        identity = (
            model["name"],
            model["revision"],
            result["benchmark_name"],
            result["benchmark_version"],
            result["dataset_revision"],
        )
        if identity in identities:
            raise ValueError(f"duplicate result identity: {identity}")
        identities.add(identity)
    return paths


def main() -> None:
    paths = validate_repository(Path.cwd())
    print(f"Validated {len(paths)} DecisionBench result records.")


if __name__ == "__main__":
    main()
