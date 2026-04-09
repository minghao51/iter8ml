# STRUCTURE

## Complete Directory Layout

### Root Level
```
/Users/minghao/Desktop/personal/iter8ml/
├── README.md                    # Project documentation
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Dependency lock file
├── main.py                     # CLI entry point
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # Docker container definition
├── technical_roadmap.md        # Development roadmap
├── AGENTS.md                   # Agent-based workflow documentation
├── CONTRIBUTING.md             # Contribution guidelines
├── CLAUDE.md                   # Claude AI integration guide
├── LICENSE                     # MIT license
├── configs/                    # Configuration modules
├── core/                       # Core library code
├── examples/                   # Example implementations
├── mcp_server/                 # MCP (Model Context Protocol) server
├── notebooks/                  # Jupyter notebooks
├── pipelines/                  # Pipeline definitions
├── tests/                      # Test suites
├── workspace/                  # Experiment workspace
└── .planning/codebase/         # Codebase analysis (created)
```

### Configuration Layer (`configs/`)
```
configs/
├── __init__.py                 # Configuration package
├── experiment.py               # Experiment configuration models
├── hardware.py                 # Hardware profile detection
├── model_configs.py           # Model-specific configurations
└── examples/                   # Configuration examples
    └── credit_risk.py         # Credit risk modeling example
```

### Core Engine (`core/`)
```
core/
├── __init__.py                 # Core library entry
├── constants.py               # Type-safe constants and enums
├── py.typed                   # Type hints declaration
├── data/                     # Data handling pipeline
│   ├── __init__.py
│   ├── adapter.py             # Data format adapters
│   ├── loaders.py             # Data loading utilities
│   ├── processors.py          # Data preprocessing
│   └── quality.py             # Data quality validation
├── engine/                    # Experiment orchestration
│   ├── __init__.py
│   ├── trainer.py             # Main training orchestration
│   ├── evaluator.py           # Cross-validation & evaluation
│   ├── hpo.py                 # Hyperparameter optimization
│   ├── state_observer.py      # Experiment state tracking
│   ├── tracker.py             # Experiment tracking backends
│   └── tracker.py             # Event logging system
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── base.py                # Abstract model base class
│   ├── gbdt_base.py          # GBDT common functionality
│   ├── selector.py            # Automatic model selection
│   ├── conventional/          # Traditional ML models
│   │   ├── __init__.py
│   │   ├── catboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── xgboost_model.py
│   ├── deep/                  # Deep learning models
│   │   ├── __init__.py
│   │   ├── ft_transformer.py
│   │   └── text_encoder.py
│   └── tabular_foundation/     # Pre-trained foundation models
│       ├── __init__.py
│       └── tabpfn_model.py
├── monitoring/                # Model monitoring
│   ├── __init__.py
│   └── drift.py               # Data drift detection
└── utils/                     # Utilities
    └── jsonl.py               # JSONL handling utilities
```

### Integration Components
```
examples/
└── zenml_pipeline.py          # ZenML pipeline integration

mcp_server/
├── __init__.py
└── tools.py                   # MCP server tools

notebooks/
└── quick_start.py            # Quick start example
```

### Testing Infrastructure (`tests/`)
```
tests/
├── __init__.py
├── integration/               # End-to-end tests
│   ├── __init__.py
│   ├── test_full_pipeline.py
│   └── test_gdbt_models.py
└── unit/                      # Unit tests
    ├── __init__.py
    ├── test_adapter.py        # Data adapter tests
    ├── test_cli.py            # CLI command tests
    ├── test_config.py         # Configuration tests
    ├── test_drift.py          # Drift detection tests
    ├── test_ft_transformer.py # Deep learning model tests
    ├── test_hpo.py            # Hyperparameter optimization tests
    ├── test_loaders.py        # Data loader tests
    ├── test_mcp_tools.py      # MCP server tests
    ├── test_model_selector.py # Model selection tests
    ├── test_processors.py     # Data processor tests
    ├── test_quality.py        # Data quality tests
    ├── test_state_observer.py # State observer tests
    ├── test_tabpfn.py         # TabPFN model tests
    └── test_tracker_rotation.py # Tracking system tests
```

### Workspace (`workspace/`)
```
workspace/
├── artifacts/                 # Experiment artifacts
│   └── [model]_exp_[timestamp]_[hash]/  # Model-specific artifacts
├── current_state.md          # Current experiment status
├── experiments.jsonl         # Experiment log
├── leaderboard.md            # Performance leaderboard
├── registry.json             # Model registry
└── registry.lock             # Registry lock file
```

## Key File Locations and Purposes

### Entry Points
- `/Users/minghao/Desktop/personal/iter8ml/main.py` - CLI entry point (`tabblueprint` command)
- `/Users/minghao/Desktop/personal/iter8ml/pyproject.toml` - Package configuration and scripts

### Core Components
- `/Users/minghao/Desktop/personal/iter8ml/core/engine/trainer.py` - Main training orchestration
- `/Users/minghao/Desktop/personal/iter8ml/core/models/base.py` - Abstract model base class
- `/Users/minghao/Desktop/personal/iter8ml/core/engine/evaluator.py` - Model evaluation
- `/Users/minghao/Desktop/personal/iter8ml/core/data/adapter.py` - Data format conversion

### Configuration
- `/Users/minghao/Desktop/personal/iter8ml/configs/experiment.py` - Experiment configuration
- `/Users/minghao/Desktop/personal/iter8ml/configs/model_configs.py` - Model configurations
- `/Users/minghao/Desktop/personal/iter8ml/configs/hardware.py` - Hardware detection

### Integrations
- `/Users/minghao/Desktop/personal/iter8ml/mcp_server/tools.py` - MCP server implementation
- `/Users/minghao/Desktop/personal/iter8ml/examples/zenml_pipeline.py` - ZenML integration

## Naming Conventions

### Files and Directories
- **snake_case** for Python files and function names
- **PascalCase** for class names
- **kebab-case** for CLI commands and experiment names
- **UPPER_CASE** for constants and enum values

### Module Organization
- **Core functionality**: `core/` directory with submodules by concern
- **Models**: Organized by type (`conventional/`, `deep/`, `tabular_foundation/`)
- **Tests**: Separated into `unit/` and `integration/` directories
- **Workspace**: Experiment artifacts stored in timestamped directories

### Configuration Files
- `.py` files for Python configurations
- JSONL for event logging (`experiments.jsonl`)
- JSON for registry (`registry.json`)
- Markdown for human-readable status (`current_state.md`)

## Module Organization

### Core Principles
1. **Separation of Concerns**: Each module has a single responsibility
2. **Plugin Architecture**: Models and trackers are pluggable
3. **Configuration-Driven**: Behavior controlled through typed configurations
4. **Event-Driven**: Experiment state tracked through events

### Key Design Decisions
- **Type Safety**: Extensive use of Pydantic models and Python enums
- **Performance**: Polars for data handling, optimized for tabular data
- **Extensibility**: Abstract base classes for easy plugin addition
- **Reproducibility**: Deterministic experiment tracking and artifact management
