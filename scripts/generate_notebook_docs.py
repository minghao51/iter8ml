"""Generate MkDocs stub pages from Quarto .qmd notebook frontmatter."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

NOTEBOOKS_DIR = Path("notebooks")
DOCS_NOTEBOOKS_DIR = Path("docs/notebooks")
HTML_DIR = DOCS_NOTEBOOKS_DIR / "html"
MKDOCS_YML = Path("mkdocs.yml")


def parse_frontmatter(qmd_path: Path) -> dict[str, str]:
    text = qmd_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    loaded = yaml.safe_load(match.group(1))
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items()}


def slug_from_stem(stem: str) -> str:
    stripped = re.sub(r"^\d+_", "", stem)
    return stripped.replace("_", "-")


def get_base_path() -> str:
    if not MKDOCS_YML.exists():
        return ""
    text = MKDOCS_YML.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("site_url:"):
            url = line.split(":", 1)[1].strip()
            break
    else:
        return ""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return path


def generate_stub(qmd_path: Path, base_path: str) -> str:
    fm = parse_frontmatter(qmd_path)
    title = fm.get("title", qmd_path.stem)
    description = fm.get("description", "")
    html_src = f"{base_path}/notebooks/html/{qmd_path.stem}.html"
    slug = slug_from_stem(qmd_path.stem)

    return (
        "---\n"
        "hide:\n"
        "  - navigation\n"
        "  - toc\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        f"{description}\n"
        "\n"
        f'<div class="iframe-container" id="iframe-wrapper-{slug}">\n'
        '  <div class="iframe-controls">\n'
        '    <button type="button" class="md-button notebook-expand-btn">Expand</button>\n'
        f'    <a href="{html_src}" target="_blank" rel="noopener noreferrer" '
        'class="md-button">Open in New Tab</a>\n'
        "  </div>\n"
        f'  <iframe src="{html_src}" allowfullscreen loading="lazy"></iframe>\n'
        "</div>\n"
        "\n"
        "## Run Locally\n"
        "\n"
        "```bash\n"
        f"uv run quarto preview notebooks/{qmd_path.name}\n"
        "```\n"
    )


def generate_index(entries: list[tuple[str, str, str]]) -> str:
    lines = [
        "# Notebooks\n",
        "Interactive tutorials demonstrating the **iter8ml** framework.\n",
    ]
    for slug, title, description in entries:
        lines.append(f"## [{title}]({slug}.md)")
        lines.append(f"\n{description}\n")
    return "\n".join(lines)


def main() -> None:
    DOCS_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    qmd_files = sorted(NOTEBOOKS_DIR.glob("*.qmd"))
    if not qmd_files:
        print("No .qmd files found in notebooks/")
        sys.exit(1)

    base_path = get_base_path()
    entries: list[tuple[str, str, str]] = []

    for qmd in qmd_files:
        fm = parse_frontmatter(qmd)
        title = fm.get("title", qmd.stem)
        description = fm.get("description", "")
        slug = slug_from_stem(qmd.stem)

        stub_content = generate_stub(qmd, base_path)
        stub_path = DOCS_NOTEBOOKS_DIR / f"{slug}.md"
        stub_path.write_text(stub_content, encoding="utf-8")
        print(f"Generated {stub_path}")

        entries.append((slug, title, description))

    index_content = generate_index(entries)
    index_path = DOCS_NOTEBOOKS_DIR / "index.md"
    index_path.write_text(index_content, encoding="utf-8")
    print(f"Generated {index_path}")

    print(f"\nDone. Generated {len(entries)} notebook stubs.")


if __name__ == "__main__":
    main()
