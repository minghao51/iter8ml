# Tech Stack - Tabular Blueprint

## Languages & Runtime
- **Python** 3.11+ (`requires-python = ">=3.11"`)
- **Operating System**: Ubuntu 22.04 (Docker)
- **CUDA**: 12.4.0 (GPU support)

## Frameworks & Dependencies

### Core ML Libraries
- **Polars** >=1.0 - DataFrame library
- **Pydantic** >=2.0 - Data validation
- **Pydantic Settings** >=2.0 - Configuration management

### ML Frameworks
- **CatBoost** >=1.2 - Gradient boosting
- **LightGBM** >=4.0 - Gradient boosting
- **XGBoost** >=2.0 - Gradient boosting
- **TabPFN** >=2.0 - Transformer-based tabular models
- **skrub** >=0.3 - Preprocessing utilities
- **CleanLab** >=2.6 - Data cleaning and label noise detection
- **Optuna** >=3.6 - Hyperparameter optimization
- **PyTorch** >=2.3 - Deep learning framework
- **Transformers** >=4.40 - HuggingFace models
- **scikit-learn** >=1.4 - ML utilities

### Data & Utilities
- **NumPy** >=1.26 - Numerical computing
- **Accelerate** >=0.30 - HuggingFace acceleration
- **Typer** >=0.12 - CLI framework
- **PSUtil** >=5.9 - System monitoring
- **Rich** >=13.0 - Terminal formatting

### Optional Dependencies
- **Hamilton** (sf-hamilton>=1.70) - Workflow orchestration
- **LLM Support** (mcp>=0.9, anthropic>=0.25) - Model Context Protocol & Anthropic
- **WandB** (wandb>=0.17) - Experiment tracking
- **MLflow** (mlflow>=2.13) - ML lifecycle management
- **ZenML** (zenml>=0.57) - ML pipeline orchestration
- **Transformers Extended** (datasets>=2.14) - Dataset loading
- **Dev Tools** (pytest>=8.0, ruff>=0.4) - Testing & linting

## Configuration Tools
- **Ruff** - Python linter & formatter
  - Line length: 100
  - Target Python 3.11
  - Rules: E, F, I, UP, B, SIM
- **Pre-commit** - Git hooks for ruff-format, ruff-check, pytest
- **pytest** - Unit testing framework
  - Test paths: tests/
  - Options: -v --tb=short

## Build System
- **UV** - Python package manager (Docker setup)
- **pyproject.toml** - Project configuration

## Entry Point
- **main.py** - CLI application using Typer
- **Script name**: `tabblueprint` (defined in pyproject.toml)

## Docker Configuration
- **Base image**: nvidia/cuda:12.4.0-runtime-ubuntu22.04
- **Python**: 3.11
- **GPU support**: Enabled via CUDA
