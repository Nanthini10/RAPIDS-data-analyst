"""Synthetic churn dataset with *planted* ground truth.

Why synthetic: the true drivers are known by construction, there is nothing to download,
and we can inject realistic mess (numeric strings, blanks, duplicates) for the Cleaner.

Planted structure
  - contract: month-to-month churns far more than 1-/2-year   (strong driver)
  - tenure:   longer tenure -> lower churn                    (strong driver)
  - monthly_charges: slight positive effect                   (weak driver)
  - gender, region: NO effect                                 (null traps)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRIVERS = ["contract", "tenure"]
NULL_TRAPS = ["gender", "region"]


def make_churn(n: int = 20_000, seed: int = 0, dirty: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 73, size=n)
    monthly = np.clip(rng.normal(70, 25, size=n), 18, 120).round(2)
    gender = rng.choice(["Male", "Female"], size=n)
    region = rng.choice(["North", "South", "East", "West"], size=n)

    logit = (
        -0.9
        + np.where(contract == "Month-to-month", 1.7, np.where(contract == "One year", 0.0, -0.8))
        - 0.045 * tenure
        + 0.008 * (monthly - 70)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(n)],
            "gender": gender,
            "region": region,
            "contract": contract,
            "tenure": tenure,
            "monthly_charges": monthly,
            "total_charges": (monthly * tenure).round(2),
            "churn": np.where(churn, "Yes", "No"),
        }
    )
    if dirty:
        df["total_charges"] = df["total_charges"].astype(str)
        blank = rng.random(n) < 0.01
        df.loc[blank, "total_charges"] = " "  # numeric column stored as string with blanks
        df.loc[rng.random(n) < 0.02, "region"] = None
        df = pd.concat([df, df.sample(frac=0.005, random_state=seed)], ignore_index=True)  # dup rows
    return df


def make_bench(n: int, seed: int = 0) -> pd.DataFrame:
    """Wide numeric + key columns for the GPU-vs-pandas benchmark."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "k1": rng.integers(0, 1000, n),
            "k2": rng.integers(0, 50, n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.random(n),
        }
    )
