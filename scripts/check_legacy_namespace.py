"""Fail if legacy tabular_blueprint namespace remains in maintained files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCAN_GLOBS = [
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "docs/**/*.md",
    "docs/notebooks/html/**/*.html",
    "notebooks/**/*.qmd",
    "notebooks/_freeze/**/*.json",
    "README.md",
    "mkdocs.yml",
]

BLOCKED_TOKENS = (
    "tabular_blueprint",
    "tabular-blueprint",
)


def _iter_candidate_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(ROOT.glob(pattern))
    return sorted({p for p in files if p.is_file()})


def main() -> int:
    failures: list[str] = []
    for path in _iter_candidate_files():
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for token in BLOCKED_TOKENS:
            if token in text:
                failures.append(f"{path.relative_to(ROOT)}: contains '{token}'")

    if failures:
        print("Legacy namespace check failed:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("Legacy namespace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
