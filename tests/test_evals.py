"""Offline checks that the eval harness itself is sound."""
import json

from evals.reference import reference_groups, reference_metrics
from evals.score import quoted_pairs, score_run
from evals.synth import make_churn

RAW = make_churn(10_000, seed=0)


def test_planted_structure_is_present():
    ref = reference_metrics(make_churn(30_000, seed=1))
    assert ref["rate_diff_m2m_vs_two_year"] > 0.25      # contract is a strong driver
    assert ref["corr_tenure_churn"] < -0.15             # tenure is a strong driver
    assert abs(ref["churn_rate[gender=Male]"] - ref["churn_rate[gender=Female]"]) < 0.02  # null trap


def test_dirty_data_is_dirty():
    df = make_churn(5_000, seed=0)
    assert df["total_charges"].eq(" ").any() and df.duplicated().any() and df["region"].isna().any()


def test_quoted_pair_parsing():
    assert quoted_pairs("rate 0.330 (n=10,913) vs 0.101 (n=5065)") == [(0.330, 10913), (0.101, 5065)]
    assert quoted_pairs("churn (0.330, n=10913)") == [(0.330, 10913)]
    assert quoted_pairs("churn 33.0% (n=10913)") == [(0.33, 10913)]
    assert quoted_pairs("correlation is -0.31") == []


def _write(tmp_path, findings):
    (tmp_path / "findings.json").write_text(json.dumps(findings))
    return tmp_path


def _claim_for_contract(raw):
    g = reference_groups(raw)
    m2m = next(x for x in g if x[1] == int((raw.drop_duplicates()["contract"] == "Month-to-month").sum()))
    return f"Month-to-month churn is {m2m[0]:.3f} (n={m2m[1]})", m2m


def test_scoring_rewards_truth_and_penalises_false_drivers(tmp_path):
    ref = reference_metrics(RAW)
    claim, _ = _claim_for_contract(RAW)
    good = [
        {"id": "f1", "status": "verified", "claim": f"Contract type drives churn. {claim}", "value": 0.3},
        {"id": "f2", "status": "verified", "claim": "Shorter tenure raises churn (r=-0.31)", "value": ref["corr_tenure_churn"]},
        {"id": "f3", "status": "verified", "claim": "Gender has no effect on churn", "value": ref["churn_rate[gender=Male]"]},
    ]
    s = score_run(_write(tmp_path, good), RAW)
    assert s["insight_recall"] == 1.0 and s["trap_precision"] == 1.0 and s["numeric_accuracy"] == 1.0

    bad = good + [{"id": "f4", "status": "verified", "claim": "Region drives churn strongly (0.4 (n=999))", "value": 0.4}]
    s = score_run(_write(tmp_path, bad), RAW)
    assert s["trap_precision"] == 0.5 and s["numeric_accuracy"] < 1.0 and s["incorrect_numbers"] == ["f4"]


def test_wrong_rate_for_real_group_is_caught(tmp_path):
    _, (rate, n) = _claim_for_contract(RAW)
    wrong = [{"id": "f1", "status": "verified", "claim": f"Contract drives churn at {rate + 0.05:.3f} (n={n})", "value": 0.1}]
    assert score_run(_write(tmp_path, wrong), RAW)["numeric_accuracy"] == 0.0
