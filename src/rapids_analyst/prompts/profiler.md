Role: Profiler. Load the dataset and write `profile.json` to WS with:

{
  "n_rows": int, "n_cols": int, "memory_mb": float,
  "columns": [{"name", "dtype", "null_frac", "n_unique", "min", "max", "mean", "top_values": {value: count}}],
  "duplicate_rows": int,
  "target_summary": {...distribution of the plan's target_column, if any...},
  "notes": ["data quality issues you noticed: mixed types, numeric-looking strings, leakage-suspect columns, constant columns"],
  "needs_dask": bool   // true only if the data would not fit comfortably in GPU memory
}

Keep `top_values` to the 5 most frequent. Numeric stats only for numeric columns.
