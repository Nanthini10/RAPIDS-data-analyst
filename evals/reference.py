"""Reference (ground-truth) numbers computed directly with pandas on the raw synthetic data.

Two views:
  reference_metrics - a small set of named scalars (overall rate, contract gaps, correlations)
  reference_groups  - (churn_rate, n) for EVERY group under many plausible groupings/binnings.
                      A finding that quotes "0.424 (n=3302)" is correct iff that exact (rate, n) pair
                      exists here. n identifies the group, so accidental matches are very unlikely.
"""
from __future__ import annotations

import pandas as pd

TENURE_EDGES = [
    [0, 12, 24, 72], [0, 12, 24, 60, 72], [0, 6, 12, 24, 48, 72], [0, 12, 24, 36, 48, 60, 72],
    [0, 12, 72], [0, 24, 72], [0, 6, 12, 18, 24, 72], [0, 12, 24, 48, 72], [0, 24, 48, 72],
]
QUANTILES = (2, 3, 4, 5, 10)


def _dedup(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    df["y"] = (df["churn"] == "Yes").astype(float)
    df["region"] = df["region"].fillna("missing")
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    return df


def reference_metrics(raw: pd.DataFrame) -> dict[str, float]:
    df = _dedup(raw)
    y = df["y"]
    m: dict[str, float] = {"churn_rate_overall": float(y.mean())}
    for c, g in y.groupby(df["contract"]):
        m[f"churn_rate[contract={c}]"] = float(g.mean())
    for c, g in y.groupby(df["gender"]):
        m[f"churn_rate[gender={c}]"] = float(g.mean())
    m["rate_diff_m2m_vs_two_year"] = m["churn_rate[contract=Month-to-month]"] - m["churn_rate[contract=Two year]"]
    m["rate_ratio_m2m_vs_two_year"] = m["churn_rate[contract=Month-to-month]"] / m["churn_rate[contract=Two year]"]
    for col in ("tenure", "monthly_charges", "total_charges"):
        m[f"corr_{col}_churn"] = float(df[col].corr(y))
    m["mean_tenure_churned"] = float(df.loc[y == 1, "tenure"].mean())
    m["mean_tenure_retained"] = float(df.loc[y == 0, "tenure"].mean())
    m["n_rows_dedup"] = float(len(df))
    return m


def reference_groups(raw: pd.DataFrame) -> list[tuple[float, int]]:
    df = _dedup(raw)
    groupings: list[pd.Series] = [df["contract"], df["gender"], df["region"], df["region"].where(df["region"] != "missing")]
    for edges in TENURE_EDGES:
        groupings.append(pd.cut(df["tenure"], edges, include_lowest=True))
    for col in ("tenure", "monthly_charges", "total_charges"):
        for q in QUANTILES:
            groupings.append(pd.qcut(df[col], q, duplicates="drop"))
    groupings.append(pd.Series("all", index=df.index))

    out: set[tuple[float, int]] = set()
    for g in groupings:
        for _, sub in df.groupby(g, observed=True):
            out.add((float(sub["y"].mean()), int(len(sub))))
    return sorted(out)
