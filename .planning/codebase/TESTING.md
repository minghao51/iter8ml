# Testing

## Framework

- **Primary framework:** pytest
- **Test command:** `uv run pytest tests/unit -v`

## Test Organization

```
tests/
├── unit/           # Isolated unit tests with mocks
├── integration/    # End-to-end tests with real datasets
└── fixtures/       # Shared test data and utilities
```

## Unit Tests

- Use mocking to isolate components
- Temporary fixtures for test data
- Fast execution for rapid feedback
- Focus on individual component behavior

## Integration Tests

- Full workflow testing
- Real datasets for realistic scenarios
- Tests complete experiment pipelines
- Validates component integration

## Coverage

- Coverage tracked for unit tests
- Aim for high coverage on core logic
- Integration tests cover end-to-end scenarios

## Running Tests

```bash
# Run unit tests
uv run pytest tests/unit -v

# Run integration tests
uv run pytest tests/integration -v

# Run all tests
uv run pytest tests/ -v
```

## Pre-commit

- Tests run as part of pre-commit hooks
- Ensures code quality before commits
