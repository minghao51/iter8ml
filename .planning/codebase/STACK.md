# Tech Stack Analysis

## Languages and Versions
- **Python**: 3.11+ (primary language)
- **Runtime**: uv (Python package and dependency manager)

## Frameworks and Libraries
### Core ML Framework
- **Polars**: 1.0+ (DataFrame library for data processing)
- **Pydantic**: 2.0+ (data validation and serialization)
- **Pydantic Settings**: 2.0+ (configuration management)

### Machine Learning Libraries
- **Gradient Boosting**: 
  - CatBoost: 1.2+
  - LightGBM: 4.0+
  - XGBoost: 2.0+
- **Deep Learning**:
  - PyTorch: 2.3+
  - Transformers: 4.40+
  - Accelerate: 0.30+
- **Other ML**:
  - Scikit-learn: 1.4+
  - TabPFN: 2.0+ (tabular foundation models)
  - Scrub: 0.3+ (feature engineering)
  - CleanLab: 2.6+ (data cleaning)
  - Optuna: 3.6+ (hyperparameter optimization)

### Optional Integrations
- **LLM Support**: Anthropic: 0.25+ (via MCP protocol)
- **Hamilton**: sf-hamilton: 1.70+ (dataflow framework)
- **Experiment Tracking**: 
  - Weights & Biases (wandb): 0.17+
  - MLflow: 2.13+
  - ZenML: 0.57+

### CLI and Development
- **Typer**: 0.12+ (CLI framework)
- **Rich**: 13.0+ (terminal styling)
- **Ruff**: 0.4+ (linter and formatter)
- **Psutil**: 5.9+ (system monitoring)
- **Pre-commit**: Local hooks for formatting, linting, and testing

## Configuration Files
- `pyproject.toml`: Project dependencies and tool configuration
- `.pre-commit-config.yaml`: Git hooks for code quality
- `.devcontainer/devcontainer.json`: Development container setup
- `Dockerfile`: Container deployment with CUDA support

## Execution Model
- **CLI Tool**: `tabblueprint` command-line interface
- **Entry Point**: `main.py` with Typer app
- **Package Management**: uv for dependency management and execution
- **Container Support**: Docker with CUDA runtime for GPU acceleration
