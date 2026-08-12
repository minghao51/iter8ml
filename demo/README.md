# iter8ml live demo

Upload a CSV (or use the bundled Telco Churn sample) → pick the target column →
get a cross-validated **leaderboard** + a **SHAP** explanation of the champion,
in the browser. Powered by [`iter8ml`](https://github.com/minghao51/iter8ml).

The heavy lifting lives in `run_analysis()` in `app.py` (no Gradio dependency,
importable and unit-testable); `create_demo()` wraps it in the Gradio UI and is
built lazily.

## Guard rails (free-tier friendly)

- Rows capped at **20,000** (larger uploads are sampled down).
- Models: **CatBoost + XGBoost** only (LightGBM is skipped — slow on CPU).
- 5-fold CV, single worker, isolated throwaway workspace per request — no
  persisted user data, no cross-request state.
- OpenMP threads capped at import (`HardwareProfile.configure_omp_threads()`)
  to avoid the libgomp deadlock on hybrid-core CPUs.

## Run locally

```bash
uv run --with gradio python demo/app.py
# -> http://127.0.0.1:7860
```

To exercise the core without Gradio:

```bash
uv run python -c "import sys; sys.path.insert(0,'demo'); \
  from app import run_analysis, SAMPLE_PATH; \
  print(run_analysis(SAMPLE_PATH, 'Churn', 'classification')[2])"
```

## Deploy to Hugging Face Spaces (manual, one-time)

1. Create a Space: <https://huggingface.co/new-space>
   - **SDK:** Gradio · **Hardware:** CPU basic (free) · **Visibility:** Public.
2. Clone it locally (auth with a **write**-scope token from
   <https://huggingface.co/settings/tokens>):
   ```bash
   git clone https://huggingface.co/spaces/<YOUR_USER>/iter8ml-demo
   ```
3. Copy the demo files into the Space root and push:
   ```bash
   cp demo/* iter8ml-demo/
   cd iter8ml-demo
   git add .
   git commit -m "iter8ml live demo"
   git push
   ```
4. The Space builds from `requirements.txt` and serves the `demo` object in
   `app.py` at `https://<YOUR_USER>-iter8ml-demo.hf.space`.

### Automated deploys (optional, later)

Store `HF_TOKEN` as a GitHub Actions secret and add a workflow that runs
`huggingface_hub.upload_folder(..., repo_type="space")` on push to `main`.
Deferred until the manual Space is stable.
