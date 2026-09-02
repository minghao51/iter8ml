# Architecture Decision Records

iter8ml's architecture is the product of a handful of deliberate tradeoffs.
Each record follows the ADR shape: **Context → Decision → Consequences** —
the *why*, not the *what*, including what is honestly not finished yet.
Reference detail lives in [Pipeline Architecture](../pipeline-architecture.md)
and [Medallion Runtime](../medallion.md).

## Index

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-single-hamilton-dag.md) | One Hamilton DAG for training, not imperative orchestration | Accepted |
| [0002](0002-pipeline-spec-config-when.md) | Pipeline behavior is *data* (`PipelineSpec` + `@config.when`), not branches | Accepted |
| [0003](0003-medallion-artifact-contract.md) | A medallion artifact contract for durable lineage | Accepted (hardened local slice) |
| [0004](0004-hardware-aware-model-routing.md) | Hardware-aware model routing (and the OpenMP war story) | Accepted |
| [0005](0005-two-reliability-seams.md) | Two reliability seams: best-effort events, required state | Accepted |
| [0006](0006-cpu-first-gpu-ready.md) | CPU-first, GPU-ready | Accepted |

## Process

1. Copy [`0000-adr-template.md`](0000-adr-template.md) to `NNNN-slug.md`
   (next free number, zero-padded to 4).
2. Write **Context → Decision → Consequences**. Be honest about negatives and
   unfinished work; consequences carry `+`/`−` markers.
3. Add the row to the index above with its status.
4. To supersede an ADR, write the new one and flip the old record's status to
   *Superseded by ADR-NNNN* — never delete or silently edit an accepted ADR.

Boundary-level changes (see `AGENTS.md` → *Architectural boundaries*) require
an ADR **before** implementation.

## Standing convention — Polars-native, with a narrow numpy seam

End-to-end data is [Polars](https://pola.rs) (`pl.DataFrame`); numpy `(X, y)`
arrays appear only at the model boundary via `DataAdapter`. This gives
lazy/Arrow-native throughput and keeps transforms memoizable inside the DAG, at
the cost of one conversion at the seam — kept deliberately narrow.

## Status of the records

These decisions are stable; the implementation around them is staged. The
medallion contract (ADR-0003) in particular is explicitly a hardened local
slice today, with a real catalog and further Platinum execution on the roadmap
(see [deferred research](../plan/deferred-research.md)). See the
[German Credit case study](../notebooks/case-study-german-credit.md) for these
decisions exercised end-to-end on a real dataset.
