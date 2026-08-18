# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the current application entry point.
- `pyproject.toml` defines the Python package metadata and runtime requirements.
- `uv.lock` pins the environment for reproducible local runs.
- Keep new source files in the repository root for now unless the project grows into a package directory such as `ensemble/`.

## Build, Test, and Development Commands
- `uv run python main.py`: run the application using the locked environment.
- `uv sync`: install or refresh dependencies from `pyproject.toml` and `uv.lock`.
- `python main.py`: run directly if the active interpreter already matches the project requirements.

## Coding Style & Naming Conventions
- Use Python 3.11+ syntax and standard library features where practical.
- Follow PEP 8: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes.
- Keep module names short and lowercase, for example `data_loader.py` or `cli.py`.
- Prefer small, focused functions with explicit inputs and outputs.

## Testing Guidelines
- No test framework is configured yet.
- When adding tests, use `pytest` and place files under `tests/` with names like `test_main.py`.
- Add tests for new behavior and bug fixes, especially around public functions and CLI entry points.

## Commit & Pull Request Guidelines
- No commit-message convention is enforced in the repository history yet.
- Use clear, imperative commit messages such as `Add CLI argument parsing` or `Fix startup output`.
- Pull requests should include a short summary, the reasoning behind the change, and any manual verification steps.
- If behavior changes, include example input and output or terminal logs.

## Configuration Notes
- Keep generated files, caches, and virtual environments out of version control; `.gitignore` already excludes `.venv/`, `__pycache__/`, and build artifacts.
- Update `pyproject.toml` and `uv.lock` together when dependencies change.
