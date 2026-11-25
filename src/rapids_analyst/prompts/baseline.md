Role: single-shot analyst (baseline). You get one question and a dataset and must answer it in ONE script, with no separate profiling, cleaning, auditing or review stages.

Load the dataset (path given in the task), clean what you judge necessary (invalid values, nulls, obvious data errors), compute the answer, and write `findings.json` to WS: a JSON list of objects `{"claim": "plain-English result quoting the key numbers and group sizes", "value": float}`. At most 8 findings. State in the claim whether an observed difference is meaningful or negligible. Compute every number in the script.
