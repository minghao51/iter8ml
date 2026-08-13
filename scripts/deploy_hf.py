"""Deploy the Gradio demo to a public Hugging Face Space.

Self-discovers the HF username from the token, creates the Space (idempotent),
and uploads only the runtime files under ``demo/`` (excluding ``demo/README.md``
so the Space's own YAML-front-matter README is not clobbered).

Usage::

    set -a; source .env; set +a
    uv run --with huggingface_hub python scripts/deploy_hf.py
"""

from __future__ import annotations

import os

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_NAME = "iter8ml-demo"


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{REPO_NAME}"

    url = create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
        token=token,
    )
    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path="demo",
        ignore_patterns=["README.md", "__pycache__/*"],
        token=token,
    )
    print("Created/updated:", url)
    print("Space page:     ", f"https://huggingface.co/spaces/{repo_id}")
    print("App URL:        ", f"https://{user}-{REPO_NAME}.hf.space")


if __name__ == "__main__":
    main()
