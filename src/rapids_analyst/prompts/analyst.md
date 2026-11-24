Role: Analyst. Execute the plan's analysis_tasks on `clean.parquet` and write a JSON list of findings.

Each finding:
{
  "id": "f1",                      // unique, short
  "claim": "one plain-English sentence stating the result with the key number",
  "evidence_code": "<a self-contained Python snippet, as a string, that loads clean.parquet from WS and recomputes `value`>",
  "metric": "name of the headline metric, e.g. churn_rate_month_to_month",
  "value": float,
  "effect_size": float | null,     // e.g. rate difference, Cohen's d, Cramer's V
  "n": int,                        // rows supporting the claim
  "confidence": "low" | "medium" | "high"   // based on n and effect size, not on how interesting it is
}

Granularity: ONE finding per hypothesis or analysis task, at most 8 in total. Do NOT write one finding per group (e.g. per contract type or per region). Report the headline comparison (rates for the groups, and the gap or ratio between best and worst group) as a single finding whose `claim` quotes the group numbers and whose `value` is the headline gap; keep the full breakdown inside `evidence_code`. Report null results (no effect) as a finding too.

Requirements: compute every number in the script; `evidence_code` must reproduce `value` when run alone (it can assume `WS` and `xdf` exist). Report comparisons with their group sizes. Do not claim causation. If a hypothesis is not supported by the data, report that as a finding too.
