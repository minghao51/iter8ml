# Data Loading

Reference for data ingestion from various sources and data hashing.

---

## Supported Formats

**Source:** `src/iter8ml/data/loaders.py`

All loading functions return a **Polars DataFrame**.

---

## CSV Loading

**Source:** `loaders.py:10`

**Function:** `load_csv(path, separator, infer_schema_length, low_memory)`

**Library:** `polars`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | File path |
| `separator` | `str` | `","` | Column delimiter |
| `infer_schema_length` | `int` | 1000 | Number of rows to scan for type inference |
| `low_memory` | `bool` | `False` | Reduce memory at cost of slower parsing |

---

## Parquet Loading

**Source:** `loaders.py:26`

**Function:** `load_parquet(path)`

**Library:** `polars`

Direct `pl.read_parquet()` — zero-copy Arrow-based loading.

---

## Auto-Detection

**Source:** `loaders.py:31`

**Function:** `load_data(path)`

Dispatches based on file extension:

| Extension | Loader |
|-----------|--------|
| `.csv` | `load_csv()` |
| `.parquet` | `load_parquet()` |
| Other | Raises `ValueError` |

---

## SQLite Loading

**Source:** `loaders.py:41`

**Function:** `load_sqlite(db_path, query)`

**Library:** `polars`, `sqlite3`

Executes a SQL query against a SQLite database and returns results as a Polars DataFrame.

### Security Measures

| Check | Implementation |
|-------|---------------|
| SELECT-only | Query must start with `SELECT` |
| Multi-statement rejection | Rejects queries containing `;` after stripping trailing semicolons |
| Keyword blocklist | Scans for: `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`, `EXEC`, `EXECUTE` |
| Empty query rejection | Rejects whitespace-only or empty queries |
| Path validation | Raises `FileNotFoundError` if database doesn't exist |

The blocklist check strips common SQL keywords (`SELECT`, `FROM`, `WHERE`, `JOIN`, `GROUP BY`, etc.) from the query before scanning for destructive keywords to avoid false positives.

---

## Data Hashing

**Source:** `loaders.py:102`

**Function:** `get_data_hash(df)`

**Library:** `polars`, `numpy`, `hashlib`

Computes a deterministic SHA-256 hash for data fingerprinting:

```
row_hashes = df.hash_rows()           # per-row hash via Polars
combined   = XOR-reduce(row_hashes)   # combine into single value
hash       = SHA256(combined)[:16]    # first 16 hex chars
```

**Output format:** `"sha256:<16 hex characters>"`

Used for cache invalidation and experiment deduplication.
