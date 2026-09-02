# Design Decisions

> **Moved (2026-08-29):** the ADR records now live in
> [`docs/decisions/`](decisions/README.md). This page remains as a redirect for
> existing links.

- [ADR index & process](decisions/README.md)
- [ADR-0001 — One Hamilton DAG for training, not imperative orchestration](decisions/0001-single-hamilton-dag.md)
- [ADR-0002 — Pipeline behavior is *data* (`PipelineSpec` + `@config.when`), not branches](decisions/0002-pipeline-spec-config-when.md)
- [ADR-0003 — A medallion artifact contract for durable lineage](decisions/0003-medallion-artifact-contract.md)
- [ADR-0004 — Hardware-aware model routing (and the OpenMP war story)](decisions/0004-hardware-aware-model-routing.md)
- [ADR-0005 — Two reliability seams: best-effort events, required state](decisions/0005-two-reliability-seams.md)
- [ADR-0006 — CPU-first, GPU-ready](decisions/0006-cpu-first-gpu-ready.md)
