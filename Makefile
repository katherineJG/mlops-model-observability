.PHONY: install lint format typecheck test check stack-test

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy app monitoring tests

test:
	uv run python -m pytest  # pytest.exe is blocked by Application Control on this machine

check: lint typecheck test

# Builds and starts the real Docker stack (about 3 minutes). Needs Docker and ports 8000, 9090, 9091, 3000.
stack-test:
	RUN_STACK_TESTS=1 uv run python -m pytest tests/integration -v
