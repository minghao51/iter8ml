# Bundled demo datasets

Small, recognizable tabular datasets shipped inside the `iter8ml` package so
`iter8 init --demo` works with no download step.

## telco_churn.parquet

- **Source:** IBM Telco Customer Churn (public sample dataset).
- **Original location:** <https://github.com/IBM/telco-customer-churn-on-icp4d>
- **Shape:** 7,043 rows × 21 columns.
- **Target column:** `Churn` (`Yes` / `No`); class balance 1,869 Yes / 5,174 No.
- **Processing applied (for stable typing):**
  - `TotalCharges` coerced from string to `Float64`; 11 blank values → `null`.
  - All other columns retained as-is, faithful to the original distribution.
  - `customerID` kept (high-cardinality identifier; the framework's feature
    engineering handles it).

Please retain this attribution if you redistribute the dataset.
