# Integrations - Tabular Blueprint

## External APIs & Services

### LLM Integration
- **Anthropic** >=0.25 - LLM provider (optional dependency)
- **MCP** >=0.9 - Model Context Protocol for LLM interactions
- **File**: pyproject.toml (optional dependencies section)

### Experiment Tracking
- **MLflow** >=2.13 - Experiment tracking and model registry
  - **Docker service**: mlflow (port 5000)
  - **Volume**: mlflow-data:/mlruns
  - **Command**: mlflow server --host 0.0.0.0 --backend-store-uri /mlruns
  - **File**: docker-compose.yml

### Monitoring & Reporting
- **WandB** >=0.17 - Experiment tracking (optional)

### Pipeline Orchestration
- **ZenML** >=0.57 - ML pipeline orchestration (optional)
- **Hamilton** >=1.70 - Workflow orchestration (optional)

## Databases & Storage

### Local Storage
- **Workspace** - Local directory for experiments
  - Path: `/workspace/`
  - Subdirectories:
    - `artifacts/` - Model artifacts
    - `experiments.jsonl` - Experiment logs
    - `registry.json` - Model registry

### Model Registry
- **JSON-based registry** - Local model registry at `workspace/registry.json`
  - Stores model metadata, scores, and registration timestamps

## Compute & Hardware

### GPU Support
- **NVIDIA CUDA** 12.4.0 - GPU acceleration
- **Docker configuration**: GPU-enabled service in docker-compose.yml
  ```yaml
  services:
    app:
      # ...
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: all
                capabilities: [gpu]
  ```

### Hardware Detection
- **HardwareProfile** class - Detects available hardware (CPU, GPU, VRAM)
- **File**: main.py (hardware command)

## Data Processing

### Data Loading
- **Polars** - Primary DataFrame library
- **skrub** - Preprocessing utilities
- **Custom loaders** - Located in `core/data/loaders.py`

### Model Management
- **Model registry** - Built-in model registry
- **ReportService** - Experiment reporting and leaderboard generation
- **File**: main.py (leaderboard and registry commands)

## Monitoring & Drift Detection

### Drift Detection
- **Custom DriftDetector** - Distribution drift detection
- **File**: main.py (drift command)
- **Features**:
  - Statistical testing for distribution shifts
  - Column-by-column drift reporting
  - P-value and test statistics

## CLI Interface

### Commands
- **init** - Initialize workspace
- **run** - Run experiments
- **leaderboard** - Show experiment results
- **registry** - Manage model registry
- **hardware** - Show hardware profile
- **drift** - Detect data drift
- **state** - Generate experiment state
- **hpo** - Hyperparameter optimization

### Configuration
- **ExperimentConfig** - Experiment configuration
- **HardwareProfile** - Hardware profile configuration
- **File**: configs/experiment.py, configs/hardware.py

## Integration Points

### Data Flow
1. Data loaded via `core/data/loaders.py`
2. Experiments configured via `configs/experiment.py`
3. Models trained via `core/engine/trainer.py`
4. Results stored in `workspace/experiments.jsonl`
5. Registry updated in `workspace/registry.json`

### External Service Integration
- No external APIs required - designed for local execution
- Optional integrations available: MLflow, WandB, ZenML
- GPU support through local NVIDIA CUDA installation

## Docker Services
- **app** - Main application service
- **mlflow** - MLflow server for experiment tracking
- **Volumes**: workspace-data, mlflow-data
