# Tabular Blueprint - Code Structure Documentation

## Directory Layout

```
iter8ml/
├── main.py                     # CLI entry point
├── pyproject.toml             # Project configuration and dependencies
├── README.md                  # Project documentation
├── configs/                   # Configuration modules
│   ├── __init__.py
│   ├── experiment.py          # Experiment configuration (Pydantic model)
│   ├── hardware.py            # Hardware profile detection
│   └── examples/              # Example configuration files
├── core/                      # Core library modules
│   ├── __init__.py            # Core library entry
│   ├── constants.py           # Enums and type definitions
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── adapters.py        # Data cleaning and adaptation
│   │   ├── loaders.py         # CSV/Parquet/SQLite loaders
│   │   └── preprocessing.py   # Data preprocessing utilities
│   ├── engine/                # Experiment orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py         # Main training orchestrator
│   │   ├── evaluator.py       # Cross-validation and metrics
│   │   ├── tracker.py         # Experiment tracking
│   │   ├── state_observer.py  # Experiment state management
│   │   └── hpo.py             # Hyperparameter optimization
│   ├── models/                # ML models
│   │   ├── __init__.py
│   │   ├── base.py            # AbstractModel Protocol
│   │   ├── factory.py         # Model factory and registry
│   │   ├── selector.py        # Model selection logic
│   │   ├── conventional/      # Traditional ML models
│   │   │   ├── __init__.py
│   │   │   ├── catboost.py    # CatBoost implementation
│   │   │   ├── lightgbm.py   # LightGBM implementation
│   │   │   └── xgboost.py    # XGBoost implementation
│   │   ├── deep/              # Deep learning models
│   │   │   ├── __init__.py
│   │   │   └── ft_transformer.py
│   │   └── tabular_foundation/ # Tabular foundation models
│   │       ├── __init__.py
│   │       └── tabpfn.py
│   ├── monitoring/            # Monitoring and drift detection
│   │   ├── __init__.py
│   │   ├── drift.py           # Distribution drift detection
│   │   └── metrics.py         # Monitoring metrics
│   ├── services/              # Service layer
│   │   ├── __init__.py
│   │   ├── registry_service.py # Model registry management
│   │   └── report_service.py  # Reporting and leaderboard
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── io.py             # File I/O utilities
│       └── logging.py         # Logging configuration
├── mcp_server/                # MCP server implementation
│   ├── __init__.py
│   ├── server.py             # MCP server entry
│   └── handlers/             # MCP protocol handlers
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── fixtures/             # Test fixtures
│   ├── unit/                 # Unit tests
│   └── integration/           # Integration tests
├── examples/                  # Usage examples
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
├── workspace/                 # Runtime workspace
│   ├── artifacts/            # Model artifacts
│   ├── experiments.jsonl     # Experiment tracking log
│   └── registry.json         # Model registry
├── .planning/                # Planning and documentation
│   └── codebase/             # Architecture docs
└── .github/                  # GitHub Actions
    └── workflows/            # CI/CD pipelines
```

## Key Locations

### Configuration
- **Experiment Config**: `configs/experiment.py`
  - Pydantic BaseModel for experiment configuration
  - Automatic task-based defaults
  - Validation and serialization
- **Hardware Profile**: `configs/hardware.py`
  - GPU/CPU detection
  - Thread configuration
  - Resource optimization

### Data Pipeline
- **Data Loaders**: `core/data/loaders.py`
  - `load_csv()`, `load_parquet()`, `load_sqlite()`
  - `get_data_hash()` for versioning
- **Data Adapter**: `core/data/adapters.py`
  - Schema cleaning and normalization
  - Type inference and conversion
  - Feature engineering utilities

### Model System
- **Abstract Interface**: `core/models/base.py`
  - Protocol defines fit/predict/save interface
- **Model Factory**: `core/models/factory.py`
  - `get_model_class()` - instantiates models
  - `available_model_names()` - lists supported models
  - `validate_model_name()` - model name validation
- **Model Selector**: `core/models/selector.py`
  - Auto-selection logic based on data characteristics
  - Model recommendation system

### Experiment Orchestration
- **Trainer**: `core/engine/trainer.py`
  - Main experiment orchestrator
  - Handles concurrent model training
  - Manages tracking and registry
- **Evaluator**: `core/engine/evaluator.py`
  - Cross-validation execution
  - Metrics computation
  - Strategy-specific splitting

### Services
- **Registry Service**: `core/services/registry_service.py`
  - Model promotion workflow
  - Version management
  - Artifact storage
- **Report Service**: `core/services/report_service.py`
  - Leaderboard formatting
  - Console and file reports
  - Experiment summary

### Tracking
- **JSONL Tracker**: `core/engine/tracker.py`
  - Line-based JSON experiment tracking
  - Run ID management
  - Metrics logging

## Naming Conventions

### Files
- **PascalCase** for classes: `ExperimentConfig`, `Trainer`, `AbstractModel`
- **snake_case** for functions and variables: `load_data`, `run_experiment`
- **kebab-case** for CLI commands: `tabblueprint run`, `tabblueprint hpo`

### Methods
- **Public methods**: descriptive verbs (`load_data`, `train_model`)
- **Private methods**: underscore prefix (`_validate_config`)
- **Property decorators**: `@property` for computed attributes

### Variables
- **Constants**: UPPER_SNAKE_CASE (`MAX_WORKERS`, `DEFAULT_CV_FOLDS`)
- **Instance variables**: snake_case with `self.` prefix
- **Loop variables**: single letters for simple cases (`i`, `x`, `y`)

### Configuration
- **Enum values**: UPPER_SNAKE_CASE with string values
- **Config fields**: snake_case matching CLI argument names
- **Default values**: explicitly typed, minimal magic numbers

### Error Handling
- **Custom exceptions**: specific and descriptive
- **Error messages**: clear action-oriented language
- **Exit codes**: 0 for success, 1 for general errors

## Module Organization

### Core Library
- **Minimal exports**: `__init__.py` files only re-export public API
- **Internal imports**: relative imports within modules
- **Circular dependency avoidance**: careful import ordering

### Test Organization
- **Unit tests**: test individual components in isolation
- **Integration tests**: test full workflows
- **Fixtures**: reusable test data and utilities

### Documentation
- **Type hints**: comprehensive throughout codebase
- **Docstrings**: following Google style guide
- **README.md**: project overview and quick start
