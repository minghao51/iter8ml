## 1. Workflow
- **Analyze First:** Read relevant files before proposing solutions. Never hallucinate.
- **Approve Changes:** Present a plan for approval before modifying code.
- **Minimal Scope:** Change as little code as possible. No new abstractions.

## 2. Output Style
- High-level summaries only.
- No speculation about code you haven't read.

## 3. Technical Stack
- **Python:**
  - Package manager: `uv`.
  - Execution: Always `uv run <command>`. Never `python`.
  - Sync: `uv sync`.
- **Frontend:**
  - Verify: Run `npm run check` and `npm test` after changes.
- **Docs:**
  - Update `ARCHITECTURE.md` if structure changes.
  - Build: `bash scripts/export-notebooks.sh && uv run mkdocs build`.
  - Preview: `uv run mkdocs serve` (re-export notebooks first if they changed).
  - Deploy: `bash scripts/export-notebooks.sh && uv run mike deploy --push --update-aliases <version> latest`. See `.github/workflows/docs.yml` for CI.
- **Files:** Markdown files must follow `YYYYMMDD-filename.md` format.
