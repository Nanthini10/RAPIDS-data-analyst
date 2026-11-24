"""Streamlit dashboard: upload a dataset, enter an API key, watch the crew work, read the report.

    pip install -e ".[dashboard]"
    rapids-dashboard
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import threading
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rapids_analyst.llm import DEFAULT_MODELS
from rapids_analyst.orchestrator import run_pipeline
from rapids_analyst.progress import STAGES, load_trace, recent_lines, summarise
from rapids_analyst.workspace import Workspace

PROVIDERS = {
    "NVIDIA NIM (build.nvidia.com)": ("nvidia", "NVIDIA_API_KEY"),
    "OpenAI-compatible endpoint": ("openai", "OPENAI_API_KEY"),
    "Anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
}
BADGE = {"pending": ":gray[· pending]", "running": ":orange[▶ running]", "done": ":green[✓ done]", "FAILED": ":red[✗ failed]"}
DEFAULT_QUESTION = "What drives customer churn, and which factors do not?"


# ----------------------------------------------------------------------------- job handling
class Job:
    """A pipeline run in a background thread. Holds NO secrets: the key lives only in the thread's closure."""

    def __init__(self, ws: Workspace):
        self.ws, self.error, self.done = ws, None, False


def start_job(dataset: Path, question: str, out_dir: Path, provider: str, model: str | None,
              api_key: str | None, base_url: str | None) -> Job:
    ws = Workspace.create(out_dir, dataset)
    ws.verbose = False  # no stderr echo; the UI polls trace.jsonl
    job = Job(ws)

    def target():
        try:
            run_pipeline(dataset, question, out_dir, model, ws=ws, provider=provider, api_key=api_key, base_url=base_url)
        except Exception as e:  # surfaced in the UI
            msg = f"{type(e).__name__}: {e}"
            job.error = msg.replace(api_key, "***") if api_key else msg
        finally:
            job.done = True

    threading.Thread(target=target, daemon=True).start()
    return job


# ----------------------------------------------------------------------------- helpers
def has_gpu_stack() -> bool:
    return importlib.util.find_spec("cudf") is not None


def read_json(root: Path, name: str, default=None):
    f = root / name
    return json.loads(f.read_text()) if f.exists() else default


def list_runs(out_dir: Path) -> list[str]:
    if not out_dir.exists():
        return []
    return sorted((d.name for d in out_dir.iterdir() if (d / "trace.jsonl").exists()), reverse=True)


def save_upload(upload, out_dir: Path) -> Path:
    dest = out_dir / "_uploads" / Path(upload.name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(upload.getvalue())
    return dest


def make_sample(out_dir: Path) -> Path | None:
    try:
        from evals.synth import make_churn  # only present in a source checkout
    except ImportError:
        return None
    dest = out_dir / "_uploads" / "sample_churn.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    make_churn(20_000, seed=0).to_csv(dest, index=False)
    return dest


def preview(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).head(200)
    return pd.read_csv(path, nrows=200)


def zip_run(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file() and not f.name.startswith("_no_gpu"):
                z.write(f, f.relative_to(root.parent))
    return buf.getvalue()


# ----------------------------------------------------------------------------- views
def render_progress(root: Path, running: bool) -> None:
    sm = summarise(root)
    done_n = sum(r["state"] == "done" for r in sm["rows"].values())
    mins, secs = divmod(int(sm["elapsed"]), 60)
    label = f"{done_n}/{len(STAGES)} stages · {mins:02d}:{secs:02d} elapsed · {sm['tokens']:,} tokens generated"
    st.progress(done_n / len(STAGES), text=label)
    cols = st.columns(len(STAGES))
    for col, stage in zip(cols, STAGES):
        r = sm["rows"][stage]
        with col:
            st.markdown(f"**{stage}**  \n{BADGE[r['state']]}")
            if r["attempts"] > 1:
                st.caption(f"{r['attempts']} attempts")
            if r["last"] and r["state"] != "done":
                st.caption(r["last"][:60])
    with st.expander("Live log", expanded=running):
        st.code("\n".join(recent_lines(root, 25)) or "waiting for the first event…", language=None)


def render_report(text: str, root: Path) -> None:
    pos = 0
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):  # report embeds charts as relative paths
        st.markdown(text[pos:m.start()])
        img = root / m.group(1)
        if img.exists():
            st.image(str(img))
        pos = m.end()
    st.markdown(text[pos:])


def findings_frame(root: Path) -> pd.DataFrame:
    findings = read_json(root, "findings.json", [])
    crit = {c["id"]: c for c in read_json(root, "critique.json", [])}
    rows = []
    for f in findings:
        c = crit.get(f["id"], {})
        rows.append({
            "status": f.get("status", "?"), "id": f["id"], "claim": f["claim"], "kind": f.get("kind", "effect"),
            "value": f["value"], "n": f["n"], "p_value": c.get("p_value"), "min_group_n": c.get("min_group_n"),
            "critic": c.get("reason", ""),
        })
    return pd.DataFrame(rows)


def render_results(root: Path) -> None:
    sm = summarise(root)
    df = findings_frame(root)
    verified = int((df["status"] == "verified").sum()) if len(df) else 0
    cpu = (root / "_no_gpu").exists()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Verified findings", f"{verified} / {len(df)}")
    k2.metric("Run time", f"{int(sm['elapsed'] // 60)}m {int(sm['elapsed'] % 60)}s")
    k3.metric("Tokens generated", f"{sm['tokens']:,}")
    k4.metric("Compute", "CPU" if cpu else "GPU", help="CPU = cuDF was unavailable and agents fell back to pandas" if cpu else "Ran on RAPIDS cuDF")

    t_report, t_find, t_charts, t_details, t_dl = st.tabs(["Report", "Findings", "Charts", "Run details", "Downloads"])

    with t_report:
        rep = root / "report.md"
        render_report(rep.read_text(), root) if rep.exists() else st.info("No report was produced for this run.")

    with t_find:
        if len(df) == 0:
            st.info("No findings.")
        else:
            if (df["status"] == "unverified").any():
                st.warning("Some findings failed the Critic and the statistical gate; they are excluded from the report.")
            st.dataframe(
                df, hide_index=True, use_container_width=True,
                column_config={
                    "claim": st.column_config.TextColumn("claim", width="large"),
                    "critic": st.column_config.TextColumn("critic note", width="medium"),
                    "value": st.column_config.NumberColumn(format="%.4f"),
                    "p_value": st.column_config.NumberColumn("p", format="%.3g"),
                },
            )

    with t_charts:
        figs = read_json(root, "figure_index.json", {})
        claims = {r["id"]: r["claim"] for r in df.to_dict("records")} if len(df) else {}
        shown = [(i, root / p) for i, p in figs.items() if (root / p).exists()]
        if not shown:
            st.info("No charts for this run.")
        for row in range(0, len(shown), 2):
            for col, (fid, path) in zip(st.columns(2), shown[row:row + 2]):
                col.image(str(path), caption=f"{fid}: {claims.get(fid, '')[:110]}")

    with t_details:
        st.subheader("Stages")
        st.dataframe(pd.DataFrame([{"stage": s, **r} for s, r in sm["rows"].items()]), hide_index=True, use_container_width=True)
        plan = read_json(root, "plan.json")
        if plan:
            st.subheader("Plan")
            st.markdown("\n".join(f"- {h}" for h in plan.get("hypotheses", [])))
        for title, name in (("Cleaning steps", "transform_log.json"), ("Data profile", "profile.json"),
                            ("GPU fallbacks", "fallbacks.json")):
            data = read_json(root, name)
            if data:
                with st.expander(title):
                    st.json(data)
        by_agent: dict[str, int] = {}
        for r in load_trace(root):
            if r["event"] == "llm_call":
                by_agent[r["agent"]] = by_agent.get(r["agent"], 0) + (r.get("output_tokens") or 0)
        if by_agent:
            st.subheader("Tokens generated by agent")
            st.bar_chart(pd.Series(by_agent, name="tokens"))

    with t_dl:
        for name, mime in (("report.md", "text/markdown"), ("analysis.py", "text/x-python"), ("findings.json", "application/json")):
            f = root / name
            if f.exists():
                st.download_button(f"Download {name}", f.read_bytes(), file_name=name, mime=mime, key=f"dl_{name}")
        st.download_button("Download full run (.zip)", zip_run(root), file_name=f"{root.name}.zip", key="dl_zip")


# ----------------------------------------------------------------------------- app
def app() -> None:
    st.set_page_config(page_title="RAPIDS Data Analyst", page_icon=None, layout="wide")
    st.title("RAPIDS Data Analyst")
    st.caption("A crew of AI agents profiles, cleans, analyses, audits and reports on your dataset using GPU dataframes (cuDF).")

    ss = st.session_state
    ss.setdefault("job", None)

    # ---- sidebar: model + key + runs ----
    with st.sidebar:
        st.header("Model")
        label = st.selectbox("Provider", list(PROVIDERS))
        provider, env_key = PROVIDERS[label]
        api_key = st.text_input("API key", type="password", help=f"Kept in memory for this session only. Or set {env_key}.")
        if not api_key and os.environ.get(env_key):
            st.caption(f"Using the key from `{env_key}`.")
        model = st.text_input("Model", value=DEFAULT_MODELS[provider], key=f"model_{provider}")
        base_url = st.text_input("Base URL (optional)", value="", key=f"base_{provider}") if provider == "openai" else None
        st.divider()
        st.caption("Compute: " + ("RAPIDS cuDF detected" if has_gpu_stack() else "no RAPIDS found, agents run on pandas"))
        out_dir = Path(st.text_input("Runs folder", value=os.environ.get("RAPIDS_ANALYST_RUNS", "runs"))).resolve()
        running = ss.job is not None and not ss.job.done
        past = list_runs(out_dir)
        view = st.selectbox("Open a previous run", ["-"] + past, key="view_run", disabled=running)

    # ---- inputs ----
    job: Job | None = ss.job
    if view != "-" and not running:
        render_results(out_dir / view)
        return

    with st.container(border=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            up = st.file_uploader("Dataset", type=["csv", "parquet"], disabled=running)
            if up is not None:
                ss["dataset"] = save_upload(up, out_dir)
            elif make_sample(out_dir) is not None and st.button("Use the sample churn dataset", disabled=running):
                ss["dataset"] = make_sample(out_dir)
                ss["question"] = DEFAULT_QUESTION
        with c2:
            question = st.text_area("Question", value=ss.get("question", ""), height=120, disabled=running,
                                    placeholder="e.g. What drives customer churn, and which factors do not?")
        dataset: Path | None = ss.get("dataset") if (up is not None or ss.get("dataset")) else None
        if dataset and dataset.exists():
            st.caption(f"Dataset: **{dataset.name}**")
            with st.expander("Preview (first rows)"):
                st.dataframe(preview(dataset), use_container_width=True)

        have_key = bool(api_key or os.environ.get(env_key))
        missing = [n for n, ok in (("an API key", have_key), ("a dataset", bool(dataset and dataset.exists())),
                                   ("a question", bool(question.strip()))) if not ok]
        if st.button("Run analysis", type="primary", disabled=running or bool(missing)):
            ss.job = start_job(dataset, question.strip(), out_dir, provider, model or None, api_key or None, base_url or None)
            st.rerun()
        if missing:
            st.caption("Needs " + ", ".join(missing) + ".")

    with st.expander("What leaves your machine?"):
        st.markdown(
            "- Sent to the model provider: your question, the first rows of the file, column statistics, and the agents' "
            "findings and cleaning logs.\n"
            "- Stays local: the full dataset. The model writes Python that runs **on this machine** with your user's "
            "permissions, so use providers and models you trust, and not sensitive data you can't share.\n"
            "- Your API key is held in memory for this session and never written to disk or the run trace."
        )

    # ---- live progress / results ----
    if job is None:
        return
    st.divider()
    if not job.done:
        st.subheader("Running")

        @st.fragment(run_every=2)
        def live():
            render_progress(job.ws.root, running=True)
            if job.done:
                st.rerun()  # full rerun: unlocks the form and shows results

        live()
    else:
        sm = summarise(job.ws.root)
        if job.error or sm["failed"]:
            st.error(f"The run failed: {job.error or 'a stage failed after retries'}")
            render_progress(job.ws.root, running=False)
        else:
            st.success(f"Finished in {int(sm['elapsed'] // 60)}m {int(sm['elapsed'] % 60)}s")
            render_results(job.ws.root)


def main() -> None:
    """Console entry point: `rapids-dashboard`."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--theme.primaryColor=#76B900",
                "--browser.gatherUsageStats=false", *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":  # executed by `streamlit run`
    app()
