.PHONY: check build

check:
	uv run ruff check scripts tests
	uv run validate-results
	uv run pytest

build:
	uv run build-leaderboard --output-dir dist
