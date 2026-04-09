# Code Conventions

## Linting and Formatting

- **Tool:** Ruff
- **Line length:** 100 characters
- **Python version:** 3.11+
- **Pre-commit hooks:** ruff format, ruff check, pytest
- **Enabled rules:** E (errors), F (pyflakes), I (import sorting), UP (upgrade), B (flake8-bugbear), SIM (simplify)

## Code Style

### Type Hints
- Type hints used throughout the codebase
- Modern union syntax: `str | Path` instead of `Union[str, Path]`
- Protocol-based interfaces for models

### Naming Conventions
- Enum classes for constants with conversion utilities
- Context managers for resource management
- Descriptive variable names

### Data Manipulation
- Polars is the primary data manipulation library
- Dataframes passed between components
- Centralized constants in `core/constants.py`

## Error Handling

- Specific error types with clear messages
- Validation at component boundaries
- Resource cleanup via context managers

## Logging

- Structured logging patterns
- Monitoring integration for tracking

## Module Organization

Core modules organized by domain:
- `data/` - Data loading, adapters, quality checks
- `models/` - Model definitions and protocols
- `engine/` - Experiment execution engine
- `monitoring/` - Metrics and monitoring
- `core/` - Shared utilities and constants

## Configuration

- Project configuration via `pyproject.toml`
- Environment-specific settings handled through config modules
