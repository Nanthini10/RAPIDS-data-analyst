"""Reference (ground-truth) metrics computed directly with pandas on the raw synthetic data."""
from __future__ import annotations

import pandas as pd


def reference_metrics(raw: pd.DataFrame) -> dict[str, float]:
    df = raw.drop_duplicates().copy()
    y = (df["churn"] == "Yes").astype(float)
    m: dict[str, float] = {"churn_rate_overall": float(y.mean())}
    for c, g in y.groupby(df["contract"]):
        m[f"churn_rate[contract={c}]"] = float(g.mean())
    for c, g in y.groupby(df["gender"]):
        m[f"churn_rate[gender={c}]"] = float(g.mean())
    m["rate_diff_m2m_vs_two_year"] = (
        m["churn_rate[contract=Month-to-month]"] - m["churn_rate[contract=Two year]"]
    )
    m["rate_ratio_m2m_vs_two_year"] = (
        m["churn_rate[contract=Month-to-month]"] / m["churn_rate[contract=Two year]"]
    )
    m["corr_tenure_churn"] = float(df["tenure"].corr(y))
    m["mean_tenure_churned"] = float(df.loc[y == 1, "tenure"].mean())
    m["mean_tenure_retained"] = float(df.loc[y == 0, "tenure"].mean())
    m["n_rows_dedup"] = float(len(df))
    return m
