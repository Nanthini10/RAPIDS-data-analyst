"""Code-writing agents: Profiler, Cleaner, Analyst, Visualizer, Critic."""
from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from ..schemas import Critique, Finding
from .base import CodeAgent

MAX_FINDINGS = 10
_FINDINGS = TypeAdapter(list[Finding])
_CRITIQUES = TypeAdapter(list[Critique])


class Profiler(CodeAgent):
    name = "profiler"

    def required_outputs(self):
        return ["profile.json"]

    def task(self, ctx):
        return (
            f"Dataset file (relative to WS): {self.dataset_rel()}\n"
            f"Plan:\n{self.dump(self.ws.read_json('plan.json'))}\n\n"
            "Write profile.json to WS."
        )


def snapshot_schema(df) -> dict:
    """Authoritative description of the CLEANED table: exact column names, dtypes and category labels."""
    cols = []
    for c in df.columns:
        s = df[c]
        info = {"name": c, "dtype": str(s.dtype), "null_frac": round(float(s.isna().mean()), 4), "n_unique": int(s.nunique())}
        if s.dtype == object or str(s.dtype) in ("string", "bool"):
            if info["n_unique"] <= 30:
                info["levels"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}
        else:
            info.update(min=float(s.min()), max=float(s.max()), mean=round(float(s.mean()), 4))
        cols.append(info)
    return {"n_rows": int(len(df)), "columns": cols}


class Cleaner(CodeAgent):
    name = "cleaner"

    def finalize(self):
        import pandas as pd

        path = self.ws.path("clean.parquet")
        df = pd.read_parquet(path)
        cats = [c for c in df.columns if str(df[c].dtype) == "category"]
        for c in cats:  # categorical dtype breaks later steps (cannot add new levels); plain strings are safer
            df[c] = df[c].astype(object)
        texts = [c for c in df.columns if df[c].dtype == object]
        for c in texts:  # no NaN in text columns: downstream labels would print as "nan"
            df[c] = df[c].fillna("missing")
        if cats or texts:
            df.to_parquet(path, index=False)
        self.ws.write_json("clean_schema.json", snapshot_schema(df))

    def required_outputs(self):
        return ["clean.parquet", "transform_log.json"]

    def task(self, ctx):
        return (
            f"Dataset file: {self.dataset_rel()}\n"
            f"Plan:\n{self.dump(self.ws.read_json('plan.json'))}\n"
            f"Profile:\n{self.dump(self.ws.read_json('profile.json'))}\n\n"
            "Write clean.parquet and transform_log.json to WS."
        )


class Analyst(CodeAgent):
    name = "analyst"

    def __init__(self, *a, output_file: str = "findings.json", **kw):
        super().__init__(*a, **kw)
        self.output_file = output_file

    def required_outputs(self):
        return [self.output_file]

    def validate(self):
        try:
            found = _FINDINGS.validate_python(self.ws.read_json(self.output_file))
        except (ValidationError, ValueError) as e:
            return str(e)[:800]
        if len(found) > MAX_FINDINGS:
            return (f"{len(found)} findings written; the limit is {MAX_FINDINGS}. Merge per-group findings: "
                    "report ONE headline finding per hypothesis (e.g. the gap between best and worst group) "
                    "and keep the group breakdown inside its evidence_code.")
        ids = [f.id for f in found]
        if len(ids) != len(set(ids)):
            return "finding ids must be unique"

    def task(self, ctx):
        t = (
            f"Plan:\n{self.dump(self.ws.read_json('plan.json'))}\n"
            f"CLEANED DATA SCHEMA (authoritative: use exactly these column names and category labels; "
            f"clean.parquet is what you load):\n{self.dump(self.ws.read_json('clean_schema.json'), 4000)}\n"
            f"Transform log:\n{self.dump(self.ws.read_json('transform_log.json'), 2000)}\n\n"
        )
        if ctx.get("revise"):
            t += (
                "REVISION ROUND. The Critic rejected these findings. Produce corrected replacements "
                "(same ids) or drop them if they cannot be supported:\n"
                f"{self.dump(ctx['revise'])}\n\n"
                f"Write the corrected list to {self.output_file}."
            )
        else:
            t += f"Write 4-8 findings (never more than {MAX_FINDINGS}) to {self.output_file}."
        return t


class Visualizer(CodeAgent):
    name = "visualizer"

    def required_outputs(self):
        return ["figure_index.json"]

    def task(self, ctx):
        findings = [f for f in self.ws.read_json("findings.json") if f.get("status") != "unverified"]
        return (
            f"Findings to chart:\n{self.dump(findings)}\n\n"
            "Write one PNG per finding to WS/figures/<finding_id>.png and figure_index.json "
            "mapping finding id -> relative png path."
        )


class Critic(CodeAgent):
    name = "critic"

    def required_outputs(self):
        return ["critique.json"]

    def validate(self):
        try:
            crit = _CRITIQUES.validate_python(self.ws.read_json("critique.json"))
        except (ValidationError, ValueError) as e:
            return str(e)[:800]
        have = {c.id for c in crit}
        want = {f["id"] for f in self.ws.read_json("findings.json")}
        if want - have:
            return f"critique.json is missing findings: {sorted(want - have)}"
        kinds = {f["id"]: f.get("kind", "effect") for f in self.ws.read_json("findings.json")}
        lacking = [c.id for c in crit if kinds.get(c.id) != "descriptive" and (c.p_value is None or c.effect_size is None)]
        if lacking:
            return (f"p_value, min_group_n and effect_size are required for every non-descriptive finding; missing for {lacking}. "
                    "Compute a chi-square test for rate comparisons across groups, Welch t-test for mean "
                    "comparisons, or a Pearson correlation test.")

    def task(self, ctx):
        return (
            f"Plan:\n{self.dump(self.ws.read_json('plan.json'), 2000)}\n"
            f"CLEANED DATA SCHEMA (authoritative column names and category labels):\n"
            f"{self.dump(self.ws.read_json('clean_schema.json'), 3000)}\n"
            f"Findings to audit (id, claim, metric, value, n, and the Analyst's evidence_code):\n"
            f"{self.dump(self.ws.read_json('findings.json'), 12000)}\n\n"
            "In your script, load the findings yourself with json.load(open(os.path.join(WS, 'findings.json'))); "
            "do not assume any variable holds them. Write critique.json to WS."
        )
