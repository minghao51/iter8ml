## 1. Workflow
- **Analyze First:** Read relevant files before proposing solutions. Never hallucinate.
- **Approve Changes:** Present a plan for approval before modifying code.
- **Minimal Scope:** Change as little code as possible. No new abstractions.
- **Check Skills:** Before any task, check + follow matching skill.
- **Verify:** Lint + type-check after changes. Ask user for command.
- **No Commits:** Never commit unless explicitly asked.

## 2. Output Style
- Concise. Bulletpoints > paragraphs.
- `file:line` references.
- No preamble/postamble. Answer directly.
- No speculation about code you haven't read.

## 3. Technical Stack
- **Package mgr:** `uv`
- **Execution:** `uv run <cmd>` (not `python` directly)
- **Install:** `uv add <pkg>`
- **Sync:** `uv sync`

## 4. File Operations
- **Read** before edit
- **Edit** > **Write** for surgical changes
- **Edit existing** > new files

## 5. Project Context
- Architecture & modules → @.planning/codebase/ARCHITECTURE.md
- Stack & dependencies → @.planning/codebase/STACK.md
- Coding conventions → @.planning/codebase/CONVENTIONS.md
- External integrations → @.planning/codebase/INTEGRATIONS.md
- Structure & layout → @.planning/codebase/STRUCTURE.md
- Testing patterns → @.planning/codebase/TESTING.md
- Concerns & decisions → @.planning/codebase/CONCERNS.md
