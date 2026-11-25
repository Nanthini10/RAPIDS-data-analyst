"""Offline tests for the real-data harness (no network, no API key)."""
import numpy as np
import pandas as pd

from evals.real.questions import QUESTIONS, build_checks
from evals.real.score import number_matches, numbers, score_answer
from evals.real.run_real import time_backends, trace_stats


def _fake_taxi(tmp_path, n=4000):
    r = np.random.default_rng(0)
    df = pd.DataFrame({
        "payment_type": r.choice([1, 2, 3, 4], n, p=[.7, .25, .03, .02]),
        "fare_amount": r.gamma(3, 6, n).round(2),
        "tip_amount": r.gamma(2, 2, n).round(2),
        "trip_distance": r.gamma(2, 1.5, n).round(2),
        "RatecodeID": r.choice([1, 2], n, p=[.9, .1]).astype(float),
        "passenger_count": r.integers(1, 7, n).astype(float),
        "PULocationID": r.choice(np.arange(1, 30), n),
        "tpep_pickup_datetime": pd.Timestamp("2023-01-01") + pd.to_timedelta(r.integers(0, 86400 * 30, n), unit="s"),
    })
    p = tmp_path / "taxi.parquet"
    df.to_parquet(p)
    return p


def test_numbers_parsing_and_matching():
    assert numbers("fare $18.45, 1,234,567 trips, 12.3% tip, 3.1M rides") == [18.45, 1234567.0, 12.3, 3.1e6]
    assert number_matches(0.123, [12.3], 0.01) and number_matches(12.3, [0.123], 0.01)
    assert not number_matches(0.123, [14.0], 0.03)


def test_answer_keys_build_and_self_score(tmp_path):
    p = _fake_taxi(tmp_path)
    for q in QUESTIONS:
        key = build_checks(q, p)
        assert key, q.id
        if q.id == "passengers":
            assert key[0]["kind"] == "null"          # random data: passenger count is a trap here
            assert score_answer(key, [{"claim": "Passenger count has a negligible effect on tip percentage", "value": 0.0}])["fully_correct"]
            assert not score_answer(key, [{"claim": "Passenger count has a strong significant effect on tip percentage", "value": 0.0}])["fully_correct"]
        else:  # an answer quoting the key's own numbers must score 1.0, an empty one 0.0
            claim = "; ".join(f"{c['label']} is {c['value']:.4f}" for c in key)
            assert score_answer(key, [{"claim": claim, "value": 0.0}])["fully_correct"], q.id
            assert score_answer(key, [])["score"] == 0.0


def test_time_backends_and_trace_stats(tmp_path):
    (tmp_path / "analysis.py").write_text("import os\nopen(os.path.join(os.environ['RA_WORKSPACE'], 'x'), 'w').write('1')\n")
    out = time_backends(tmp_path)
    assert out["default"]["ok"] and out["pandas"]["ok"]
    (tmp_path / "trace.jsonl").write_text(
        '{"t": 1, "event": "llm_call", "output_tokens": 10, "input_tokens": 5, "seconds": 2}\n'
        '{"t": 5, "event": "exec", "ok": false, "seconds": 1.5}\n')
    s = trace_stats(tmp_path)
    assert s["output_tokens"] == 10 and s["exec_failures"] == 1 and s["wall_seconds"] == 4.0


def test_baseline_validates_output(tmp_path):
    from evals.real.baseline import Baseline
    from rapids_analyst.workspace import Workspace

    d = tmp_path / "d.csv"
    d.write_text("a\n1\n")
    ws = Workspace.create(tmp_path / "r", d)
    b = Baseline(None, ws, [], "q", "", "")
    ws.write_json("findings.json", [{"claim": "c", "value": 1.0}])
    assert b.validate() is None
    ws.write_json("findings.json", [])
    assert b.validate()
