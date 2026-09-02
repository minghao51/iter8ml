# ADR-0004: Hardware-aware model routing (and the OpenMP war story)

- **Status:** Accepted
- **Provenance:** recorded 2026-08-29, split verbatim from `docs/design-decisions.md`

## Context

"Which models should I run?" should not be a manual decision, and
the defaults should not depend on hoping the host has a GPU. Worse, the naive
defaults actively break: on hybrid (P+E-core) CPUs under Linux/WSL2, the GBDT
libraries' libgomp **deadlocks across all cores** — the process hangs silently
and exits `124` (Phase-1 issue 1.6b). `n_jobs=-1` is a footgun.

## Decision

`models="auto"` resolves through a `ModelSelector` keyed on the
task and **detected VRAM**; `max_workers` is auto-reduced to 1 on low-VRAM GPUs;
and OpenMP threads are capped (`HardwareProfile.configure_omp_threads()`,
≤8 on Linux) **before any GBDT library is allowed to load libgomp**.

## Consequences

- **+** A sensible, host-appropriate default with zero configuration; no
  silent deadlocks; reproducible CPU runs.
- **−** "auto" can surprise a user who expected a specific model — pin an
  explicit `models=[...]` list to opt out. The lazy GBDT load also means the OMP
  guard must run early in any entrypoint (notebook/demo/CLI) — see the case
  study's hidden setup cell.
