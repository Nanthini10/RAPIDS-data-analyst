"""Offline checks that the eval harness itself is sound."""
import json

from evals.reference import reference_metrics
from evals.score import score_run
from evals.synth import make_churn


def test_planted_structure_is_present():
    ref = reference_metrics(make_churn(30_000, seed=1))
    assert ref["rate_diff_m2m_vs_two_year"] > 0.25      # contract is a strong driver
    assert ref["corr_tenure_churn"] < -0.15             # tenure is a strong driver
    assert abs(ref["churn_rate[gender=Male]"] - ref["churn_rate[gender=Female]"]) < 0.02  # null trap


def test_dirty_data_is_dirty():
    df = make_churn(5_000, seed=0)
    assert df["total_charges"].eq(" ").any() and df.duplicated().any() and df["region"].isna().any()


def _write(tmp_path, findings):
    (tmp_path / "findings.json").write_text(json.dumps(findings))
    return tmp_path


def test_scoring_rewards_truth_and_penalises_false_drivers(tmp_path):
    raw = make_churn(10_000, seed=0)
    ref = reference_metrics(raw)
    good = [
        {"id": "f1", "status": "verified", "claim": "Contract type drives churn", "value": ref["rate_diff_m2m_vs_two_year"]},
        {"id": "f2", "status": "verified", "claim": "Shorter tenure raises churn", "value": ref["corr_tenure_churn"]},
        {"id": "f3", "status": "verified", "claim": "Gender has no effect on churn", "value": ref["churn_rate[gender=Male]"]},
    ]
    s = score_run(_write(tmp_path, good), ref)
    assert s["insight_recall"] == 1.0 and s["trap_precision"] == 1.0 and s["numeric_accuracy"] == 1.0

    bad = good + [{"id": "f4", "status": "verified", "claim": "Region drives churn strongly", "value": 0.123456}]
    s = score_run(_write(tmp_path, bad), ref)
    assert s["trap_precision"] == 0.5 and s["numeric_accuracy"] < 1.0
