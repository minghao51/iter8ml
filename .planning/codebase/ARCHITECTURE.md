# Codebase Architecture Reference

- **Last updated:** 2026-08-29
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

This directory is the internal codebase reference set (the "what lives where
+ how it's built" companion to the published docs). Start at the repo root
`ARCHITECTURE.md` and `docs/README.md` for the high-level map; ADRs in
`docs/decisions/` record *why* the architecture is shaped this way.

## Index

| Doc | Scope |
|-----|-------|
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) (repo root) | Module map, Hamilton DAG modes, data/training flow, trainer seams, export, guardrails |
| [`STACK.md`](STACK.md) | Python floor, dependencies + why, extras, tooling (ruff/mypy/pytest), docs toolchain, CI |
| [`STRUCTURE.md`](STRUCTURE.md) | Repo + per-module layout, tests layout, generated-docs pipeline |
| [`CONVENTIONS.md`](CONVENTIONS.md) | Polars-first, Pydantic boundaries, config-over-code, errors, OMP guard, serialization, commit rules |
| [`TESTING.md`](TESTING.md) | pytest tiers, fixtures, property tests, how to run, contract invariants |
| [`INTEGRATIONS.md`](INTEGRATIONS.md) | MCP, LLM/litellm, W&B/MLflow, JSONL tracker, HF Spaces, Pages/mkdocs — gating + status |
| [`CONCERNS.md`](CONCERNS.md) | Risks, gotchas, debt register, deferred scope (cross-linked to ADRs) |

## Design decisions (ADRs)

Architecture-shaping records live in `docs/decisions/` (ADR-0001…0006):
single Hamilton DAG, data-driven `n` config, medallion artifact contract,
hardware-aware routing + OpenMP guard, two reliability seams, CPU-first/GPU-ready.
New boundary changes require an ADR first (see `AGENTS.md` → architectural
boundaries).
