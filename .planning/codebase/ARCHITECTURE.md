# ARCHITECTURE

## Overall Architectural Pattern

Tabular Blueprint follows a **Layered Plugin Architecture** with clear separation of concerns. The system is designed as a modular machine learning framework for tabular data with the following key characteristics:

- **Plugin-based model system** with abstract base classes for extensibility
- **Configuration-driven experiments** using Pydantic models
- **Event-driven tracking** with pluggable backends (JSONL, Weights & Biases, MLflow)
- **Pipeline-oriented** data flow with adapters and processors

## Key Layers and Their Responsibilities

### 1. **CLI Layer** (`main.py`)
- Entry point using Typer for command-line interface
- Commands: `init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`
- Handles user input validation and parameter parsing
- Manages workspace initialization and experiment orchestration

### 2. **Configuration Layer** (`configs/`)
- **ExperimentConfig**: Core experiment definition (task, target, data path, models)
- **HardwareProfile**: System resource detection (GPU, RAM, CPU)
- **ModelConfigs**: Predefined model configurations and HPO search spaces
- Type-safe configuration using Pydantic enums

### 3. **Core Engine Layer** (`core/`)
- **Trainer**: Main orchestration engine for model training
- **Evaluator**: Cross-validation and performance evaluation
- **HPO**: Hyperparameter optimization using Optuna
- **StateObserver**: Experiment state tracking and reporting
- **Tracker**: Event logging and experiment tracking backends

### 4. **Data Layer** (`core/data/`)
- **Loaders**: Data loading from various formats (CSV, Parquet, etc.)
- **Adapter**: Data transformation between formats (pandas, numpy, etc.)
- **Processors**: Data preprocessing and feature engineering
- **Quality**: Data validation and quality checks

### 5. **Model Layer** (`core/models/`)
- **Base Model**: Abstract base class for all models
- **Conventional**: Gradient boosting models (CatBoost, LightGBM, XGBoost)
- **Deep Learning**: Neural network models (FT-Transformer)
- **Tabular Foundation**: Pre-trained models (TabPFN)
- **Selector**: Automatic model selection based on data characteristics

### 6. **Monitoring Layer** (`core/monitoring/`)
- **Drift Detection**: Statistical monitoring of data drift
- Performance tracking over time
- Model versioning and registry management

## Data Flow and Communication Patterns

1. **Initialization Flow**:
   ```
   CLI → Workspace Setup → Configuration Loading → Data Loading → Model Selection
   ```

2. **Training Flow**:
   ```
   Data → Preprocessing → Model Selection → Training → Cross-validation → Tracking
   ```

3. **HPO Flow**:
   ```
   Data → Search Space Definition → Optuna Optimization → Best Model Selection → Registration
   ```

4. **Event Tracking**:
   ```
   Engine Events → Tracker Backend → JSONL/WandB/MLflow → Leaderboard & Registry
   ```

## Abstractions and Interfaces

### Core Abstract Classes
- `BaseModel`: Interface for all model implementations
- `DataLoader`: Abstract data loading interface
- `TrackerBackend`: Pluggable tracking backends
- `DriftDetector`: Interface for drift detection algorithms

### Key Design Patterns
- **Strategy Pattern**: For cross-validation strategies and model selection
- **Factory Pattern**: For model instantiation based on names
- **Observer Pattern**: For experiment state tracking
- **Plugin Architecture**: For adding new models and tracking backends

## Entry Points and Initialization Sequence

1. **Primary Entry Point**: `main.py` (CLI via `tabblueprint` command)
2. **Secondary Entry Points**: 
   - `mcp_server/`: MCP (Model Context Protocol) server integration
   - `examples/`: Example pipelines and integrations

### Initialization Steps:
1. Workspace creation with `tabblueprint init`
2. Configuration loading (CLI args or config files)
3. Hardware profile detection
4. Data loading and validation
5. Model selection or explicit model configuration
6. Experiment execution via Trainer

## Key Integrations

- **MCP Server**: For AI model context protocol integration
- **ZenML**: Pipeline orchestration support
- **Hamilton**: Dataflow programming integration
- **Transformers**: Deep learning model support
- **Various ML Libraries**: CatBoost, LightGBM, XGBoost, TabPFN, PyTorch
