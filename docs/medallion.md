# Medallion runtime

iter8ml now exposes a local-first artifact contract around the existing Hamilton
and Polars execution path. The logical layers are:

| Layer | Durable product |
|---|---|
| Bronze | Immutable source snapshot plus observed schema and fingerprint |
| Silver | Validated canonical frame before learned transforms |
| Gold | Features, labels, and split membership with overlap checks |
| Platinum | Run metrics and model evidence linked to the Gold product |

Products live under `workspace/lake/01_bronze` through
`workspace/lake/04_platinum`. A product is readable only after its manifest and
`_SUCCESS` marker are committed atomically. `workspace/control/catalog/catalog.duckdb`
is a rebuildable local index; manifests and artifacts remain the source of truth.

## Commands

```shell
iter8 plan --config experiment.yaml --graph
iter8 data ingest --data data.csv --name customer_data
iter8 verify <product-id> --deep
iter8 catalog rebuild
iter8 docs export
```

The compatibility `iter8 run` command continues to use `ExperimentConfig` and
`Trainer`. The `MedallionExecutionService` adds explicit product boundaries for
callers that need durable Bronze-to-Platinum lineage.

`MedallionExecutionService.resume(run_id)` trusts only a terminal `run.json`
whose recorded stage products pass deep checksum verification; event history
alone is never treated as a checkpoint.

Learned transforms must remain downstream of split construction. The Gold
split artifact records `row_id`, `fold`, `role`, and `repeat`; verification
rejects train/validation overlap within a fold.
