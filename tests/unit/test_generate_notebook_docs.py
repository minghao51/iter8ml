from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_notebook_docs.py"
    spec = importlib.util.spec_from_file_location("generate_notebook_docs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_frontmatter_uses_yaml(tmp_path: Path) -> None:
    module = _load_module()
    qmd = tmp_path / "sample.qmd"
    qmd.write_text(
        "---\n"
        'title: "Sample Title"\n'
        'description: "Sample Description"\n'
        "date: today\n"
        "format:\n"
        "  html:\n"
        "    self-contained: true\n"
        "---\n"
        "\n"
        "Body\n",
        encoding="utf-8",
    )

    frontmatter = module.parse_frontmatter(qmd)

    assert frontmatter["title"] == "Sample Title"
    assert frontmatter["description"] == "Sample Description"
    assert frontmatter["date"] == "today"
    assert "format" in frontmatter


def test_generate_stub_hardens_new_tab_links(tmp_path: Path) -> None:
    module = _load_module()
    qmd = tmp_path / "04_feature_engineering_explainability.qmd"
    qmd.write_text(
        '---\ntitle: "Feature Engineering"\ndescription: "Notebook description"\n---\n\nBody\n',
        encoding="utf-8",
    )

    stub = module.generate_stub(qmd, "/iter8ml")

    assert 'target="_blank"' in stub
    assert 'rel="noopener noreferrer"' in stub
    assert "onclick=" not in stub
    assert 'class="md-button notebook-expand-btn"' in stub
