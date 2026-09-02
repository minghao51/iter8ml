# ADR-0006: CPU-first, GPU-ready

- **Status:** Accepted
- **Provenance:** recorded 2026-08-29, split verbatim from `docs/design-decisions.md`

## Context

Most tabular datasets fit comfortably on a laptop CPU, and CPU
access is near-universal while GPU access is uneven. Yet many frameworks treat
CPU as an afterthought or assume a GPU. That inverts the real distribution of
tabular work.

## Decision

iter8ml is **CPU-first**: a dedicated CPU benchmark suite and the
OpenMP hardening above (ADR-0004) make the CPU path a first-class, reproducible
target — the path that runs in CI and on any machine.

This is **not** CPU-only. The same hardware detection that powers model routing
(ADR-0004) auto-detects VRAM when a GPU is present and routes to it —
GPU-appropriate models are selected, `max_workers` scales up, and the GPU path is
exercised, not merely permitted. Both paths are real and tested; CPU is the
default, GPU is a first-class opt-in the host enables by having one.

## Consequences

- **+** Runs anywhere; reproducible, free CI; low friction for the common
  (small-to-medium tabular) case.
- **−** For very large data or deep models the CPU path is slower — which is
  exactly when the auto-detected GPU path takes over.
