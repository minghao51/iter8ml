# ADR-0002: Pipeline behavior is *data* (`PipelineSpec` + `@config.when`), not branches

- **Status:** Accepted
- **Provenance:** recorded 2026-08-29, split verbatim from `docs/design-decisions.md`

## Context

A real pipeline has many behavior switches: drift method
(psi / domain-classifier / both), target transform (none / log1p / yeo-johnson /
box-cox), calibration (none / platt / isotonic), feature strategy. Encoding these
as `if/else` inside node code produces an exponential tangle that is impossible
to introspect or serialize.

## Decision

Steps and their parameters live in a serializable
[`PipelineSpec`](../pipeline-architecture.md); nodes
select implementations with Hamilton's `@config.when(...)`. The node code itself
has no branching — the config activates the right variant.

## Consequences

- **+** A pipeline is queryable and describable — `describe_pipeline(spec)`
  returns exactly what will run, and `visualize_pipeline(spec)` annotates disabled
  steps. That same data drives `iter8 plan --graph`.
- **+** Adding a variant is purely additive (a new `@config.when` function); no
  existing node changes.
- **−** Indirection: to know *what* runs you read the spec, not the code. The
  spec→config resolver (`_resolve_hamilton_config`) is a thing that can be
  forgotten when adding a param.
