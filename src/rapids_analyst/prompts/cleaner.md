Role: Cleaner. Using the profile's notes, produce an analysis-ready table.

Write to WS:
- `clean.parquet`: the cleaned dataframe.
- `transform_log.json`: list of `{"step": str, "columns": [...], "rows_affected": int, "rationale": str}`.

Guidelines: fix dtypes (numeric strings -> numbers), strip whitespace from category labels (see the label rules below), drop exact duplicates, handle nulls with the least invasive option (impute with median/mode or keep as an explicit "missing" category; drop rows only if the target is null), never drop more than 5% of rows without logging why. Do not scale or one-hot encode unless the plan needs it; keep original column names. Preserve the target column.

Label and dtype rules (downstream agents depend on them): do NOT change the text of category labels (no title-casing, lower-casing or renaming): trim surrounding whitespace only. Store categorical columns as plain strings (object), never the pandas `category` dtype. Fill missing categorical values with the literal string "missing" so no NaN remains in text columns. Save with `to_parquet(..., index=False)`.
