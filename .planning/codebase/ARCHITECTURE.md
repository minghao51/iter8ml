# Tabular Blueprint - Architecture Documentation

## Overview
Tabular Blueprint is a high-velocity iteration framework for tabular machine learning, built around a modular architecture with clear separation of concerns.

## Core Patterns

### 1. Command Layer Pattern
- **Entry Point**: `main.py` - CLI interface using Typer
- **Commands**: `init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`
- **Flow**: CLI → Config Parser → Data Loading → Trainer → Models → Evaluation → Registry

### 2. Configuration-Driven Architecture
- **Base Config**: `configs/experiment.py` - Pydantic-based experiment configuration
- **Hardware Profile**: `configs/hardware.py` - System resource detection and optimization
- **Validation**: Automatic metric/CV strategy defaults based on task type
- **Serialization**: Enum-to-string conversion for JSON compatibility

### 3. Engine-Orchestrated Flow
- **Trainer**: `core/engine/trainer.py` - Central orchestrator
- **Evaluator**: `core/engine/evaluator.py` - Cross-validation and metrics computation
- **Tracker**: `core/engine/tracker.py` - Experiment tracking (JSONL, WandB, MLFlow)
- **State Observer**: `core/engine/state_observer.py` - Experiment state management
- **HPO**: `core/engine/hpo.py` - Hyperparameter optimization with Optuna

### 4. Model Registry Pattern
- **Abstract Model**: `core/models/base.py` - Protocol for structural subtyping
- **Factory Pattern**: `core/models/factory.py` - Model instantiation and validation
- **Selector**: `core/models/selector.py` - Auto-selection logic
- **Concrete Models**: `core/models/conventional/`, `core/models/deep/`, `core/models/tabular_foundation/`

### 5. Service Layer
- **Registry Service**: `core/services/registry_service.py` - Model promotion and versioning
- **Report Service**: `core/services/report_service.py` - Leaderboard formatting and display
- **Promotion Result**: Structured model promotion workflow

## Abstractions

### Data Flow
```
CLI Command → ExperimentConfig → Data Loading → Model Selection → Training → Evaluation → Registry
```

### Key Abstractions:
1. **AbstractModel Protocol**: Defines fit/predict/save interface for all models
2. **Task Type**: Classification vs Regression with appropriate defaults
3. **CV Strategy**: KFold, Stratified, TimeSeries with smart defaults
4. **Tracker Interface**: Pluggable tracking backends
5. **Hardware Profile**: Auto-detection and thread configuration

### Data Handling
- **Format**: Polars DataFrames for performance
- **Loaders**: CSV, Parquet, SQLite support
- **Adapter**: Cleaned data with consistent schema
- **Hashing**: Data versioning via SHA256

## Layer Architecture

### Presentation Layer
- `main.py` - CLI interface
- Commands validate inputs and delegate

### Service Layer
- Core services for registry, reporting, state management
- No business logic, pure coordination

### Engine Layer
- Experiment orchestration
- Cross-validation execution
- Hyperparameter optimization
- Hardware resource management

### Data Layer
- Data loading and preprocessing
- Format conversion
- Version tracking

### Model Layer
- Abstract model interface
- Concrete implementations
- Auto-selection logic

## Entry Points

### Primary Entry Point
- **File**: `main.py`
- **Function**: `app()` - Typer CLI application
- **Command**: `tabblueprint` (script defined in pyproject.toml)

### Module Exports
- **Core**: `core/__init__.py` - Minimal import
- **Engine**: `core/engine/__init__.py` - Evaluator, Tracker, Trainer
- **Models**: `core/models/__init__.py` - AbstractModel, ModelSelector, Factory
- **Data**: `core/data/__init__.py` - Loaders
- **Services**: `core/services/__init__.py` - RegistryService, ReportService

## Configuration Flow

1. **CLI Parsing**: Typer converts CLI args to ExperimentConfig
2. **Validation**: Pydantic validates and applies defaults
3. **Task Alignment**: Auto-sets metrics and CV strategy based on task type
4. **Hardware Detection**: Auto-configures OpenMP threads and GPU detection
5. **Execution**: Trainer orchestrates the full workflow

## Key Design Decisions

1. **Polars over Pandas**: Performance for tabular data
2. **Protocol over ABC**: Structural subtyping for models
3. **JSONL Tracking**: Simple, file-based experiment tracking
4. **Enum-Driven Config**: Type-safe configuration with serialization
5. **Modular Models**: Separate packages for conventional, deep, foundation models
6. **Hardware Auto-Detection**: Optimizes resource usage automatically
