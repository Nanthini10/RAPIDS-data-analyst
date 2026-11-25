"""Questions on the NYC taxi data, each with an answer key computed by plain, independent pandas.

The population for every question is stated in the question text so the answer is well defined, but the data
is NOT cleaned for the system: it still has negative fares, zero distances, stray dates, null passenger counts.

A check is one of:
  {"kind": "number", "label": str, "value": float, "tol": rel tolerance}  # some number in the answer matches
  {"kind": "null", "topic": regex, "label": str}   # answer says the topic has negligible/no effect (trap)
  {"kind": "effect", "topic": regex, "label": str} # answer says the topic matters
Percent and fraction forms are both accepted for numbers (43.5 matches 0.435).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

NULL_ETA = 0.05  # standardised effect below this counts as "negligible" in the answer key (same floor as the gate)


@dataclass
class Question:
    id: str
    text: str
    columns: list[str]
    key: Callable[[pd.DataFrame], list[dict]]
    notes: str = ""
    checks: list[dict] = field(default_factory=list)


def _num(label, value, tol=0.03):
    return {"kind": "number", "label": label, "value": float(value), "tol": tol}


def q_payment(df):
    n = len(df)
    shares = df["payment_type"].value_counts() / n
    fare = df[df["fare_amount"] > 0].groupby("payment_type")["fare_amount"].mean()
    return [
        _num("share credit card", shares.get(1, 0), 0.02), _num("share cash", shares.get(2, 0), 0.02),
        _num("mean fare credit card", fare[1]), _num("mean fare cash", fare[2]),
    ]


def q_tip_hour(df):
    d = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0)].copy()
    d["tip_pct"] = d["tip_amount"] / d["fare_amount"] * 100
    d["hour"] = d["tpep_pickup_datetime"].dt.hour
    by = d.groupby("hour")["tip_pct"].median()   # median: a few absurd tip/fare ratios wreck the mean (checked on real data)
    return [_num("overall median tip %", d["tip_pct"].median()), _num(f"median tip % at best hour ({by.idxmax()}h)", by.max()),
            _num(f"median tip % at worst hour ({by.idxmin()}h)", by.min())]


def q_airport(df):
    d = df[(df["fare_amount"] > 0) & (df["trip_distance"] > 0)]
    j, s = d[d["RatecodeID"] == 2], d[d["RatecodeID"] == 1]
    return [_num("JFK mean fare", j["fare_amount"].mean()), _num("JFK mean distance", j["trip_distance"].mean()),
            _num("standard mean fare", s["fare_amount"].mean()), _num("standard mean distance", s["trip_distance"].mean()),
            _num("JFK trips", len(j), 0.05)]


def q_passengers(df):
    d = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0) & df["passenger_count"].between(1, 6)].copy()
    d["tip_pct"] = d["tip_amount"] / d["fare_amount"] * 100
    g = d.groupby("passenger_count")["tip_pct"]
    ssb = (g.count() * (g.mean() - d["tip_pct"].mean()) ** 2).sum()
    eta = float((ssb / ((d["tip_pct"] - d["tip_pct"].mean()) ** 2).sum()) ** 0.5)
    kind = "null" if eta < NULL_ETA else "effect"
    return [{"kind": kind, "topic": r"passenger", "label": f"passenger count vs tip % (eta={eta:.3f})"}]


def q_zones(df):
    top = df["PULocationID"].value_counts().head(3)
    out = []
    for z, c in top.items():
        out.append(_num(f"trips from zone {z}", c, 0.02))
        out.append(_num(f"mean fare from zone {z}", df[(df["PULocationID"] == z) & (df["fare_amount"] > 0)]["fare_amount"].mean()))
    return out


QUESTIONS = [
    Question("payment", "What share of all trips is paid by credit card (payment_type 1) and by cash (2), and what is the "
             "average fare_amount of credit-card vs cash trips (fares above 0 only)?",
             ["payment_type", "fare_amount"], q_payment),
    Question("tip_hour", "Among credit-card trips with fare_amount above 0, what is the MEDIAN tip as a percentage of "
             "fare_amount, and which pickup hours of the day have the highest and lowest median tip percentage?",
             ["payment_type", "fare_amount", "tip_amount", "tpep_pickup_datetime"], q_tip_hour),
    Question("airport", "Compare JFK-rate trips (RatecodeID 2) with standard-rate trips (RatecodeID 1): average fare_amount, "
             "average trip_distance and number of JFK trips (fare and distance above 0 only).",
             ["RatecodeID", "fare_amount", "trip_distance"], q_airport),
    Question("passengers", "Does passenger_count (1 to 6) affect the tip percentage on credit-card trips with fare above 0? "
             "Say whether it matters in practice, not only whether it is statistically detectable.",
             ["payment_type", "fare_amount", "tip_amount", "passenger_count"], q_passengers,
             notes="trap: huge n makes any difference 'significant'; the key says whether the effect is practically negligible"),
    Question("zones", "Which 3 pickup zones (PULocationID) have the most trips, with their trip counts and the average "
             "fare_amount of each (fares above 0 only)?", ["PULocationID", "fare_amount"], q_zones),
]


def build_checks(q: Question, parquet) -> list[dict]:
    """Compute the answer key by reading ONLY the needed columns with pyarrow + pandas (independent code path)."""
    import pyarrow.parquet as pq

    df = pq.read_table(parquet, columns=q.columns).to_pandas()
    return q.key(df)
