# Contributing

## Code Style

We use `ruff` for linting and formatting. All code must pass:

```bash
uv run ruff check .
uv run ruff format .
```

## Pull Request Guidelines

1. **One logical change per PR** — keep commits focused
2. **Run tests** — `uv run pytest tests/unit -v`
3. **Update tests** — add tests for new functionality
4. **Follow conventions** — match existing code style

## Architecture Principles

- **Functional over class-heavy** — prefer pure functions
- **Explicit over magic** — no hidden state or silent fallbacks
- **Polars as single source of truth** — no Pandas in `src/tabular_blueprint/`
- **Config is code** — Pydantic models, not YAML
- **Observability first** — JSONL events, not separate databases
- **Hardware-aware by default** — auto-route based on dataset size + VRAM

## Adding a New Model

1. Create wrapper in `src/tabular_blueprint/models/` conforming to `AbstractModel` protocol.
2. Register the wrapper in `src/tabular_blueprint/models/factory.py` (`_MODEL_REGISTRY`).
3. Add defaults/search space in `src/tabular_blueprint/models/model_configs.py`.
4. Update `src/tabular_blueprint/models/selector.py` routing if needed.
5. Add unit and integration tests.

## Testing

```bash
# Unit tests (fast, no heavy ML)
uv run pytest tests/unit -v

# Integration tests (full pipeline)
uv run pytest tests/integration -v
```
