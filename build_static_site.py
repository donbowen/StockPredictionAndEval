"""Build the static (GitHub Pages) version of the dashboard into docs/.

    python build_static_site.py

The Streamlit app needs a running Python server, so a hosted copy has to wake up before it can show anything.
This script runs the app's own calculations once, here, and writes the results as JSON next to a plain
HTML/JavaScript front end (docs/index.html, docs/app.js). GitHub Pages serves that folder as is, so the page
opens instantly and never sleeps.

What is precomputed here, with the functions in streamlit_app.py: the factor regressions ("Table 1"), Sharpe
ratios, turnover summaries, and, for every model and test year, the pipeline diagram and feature importances
(which need the saved sklearn/torch pipelines). Everything that depends on a widget (trading-cost assumptions,
sample start, moving-average window, tax rates) is computed in the browser from the shipped monthly data.

Re-run this after build_dashboard_bundle.py (or after editing the app's numbers) and commit docs/.
The page text lives in docs/app.js, so edits to the prose in streamlit_app.py need to be copied there by hand.
"""
import json
import logging
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
for name in list(logging.root.manager.loggerDict):
    if name.startswith("streamlit"):
        logging.getLogger(name).setLevel(logging.ERROR)

import streamlit_app as app  # noqa: E402  (runs in Streamlit's "bare" mode: st.* calls are no-ops)

OUT = Path("docs")
DATA = OUT / "data"


def clean(x, digits=7):
    """JSON-ready copy of x: NaN/inf -> None, floats rounded to `digits` significant digits."""
    if isinstance(x, dict):
        return {str(k): clean(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v, digits) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist(), digits)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{digits}g}")
    return x


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), separators=(",", ":")), encoding="utf-8")
    print(f"  {path}  {path.stat().st_size / 1e3:,.0f} kB")


def dates(index):
    return [d.strftime("%Y-%m-%d") for d in index]


def table_json(df):
    """A (Model, Metric) x portfolio table of alphas and t-stats."""
    return {"rows": [list(i) for i in df.index], "cols": list(df.columns), "values": df.values.tolist()}


def build_main():
    returns_wide_df, _, turnover_df = app.load_data()
    ff = app.load_factors()
    models = sorted({c.split("_")[1] for c in returns_wide_df.columns if c.startswith("ret_")})
    index = app.model_index()
    idx = returns_wide_df.index

    returns, table1, sharpe, turnover_summary, hold = {}, {}, {}, {}, {}
    for m in models:
        cols = [f"ret_{m}_bin{b}" for b in ["1", "2", "3", "4", "5", "LS"]]
        returns[m] = {c.rsplit("_bin", 1)[1]: returns_wide_df[c].tolist() for c in cols}
        t = app.calculate_table_1(returns_wide_df, m, ff)
        table1[m] = table_json(t)
        sharpe[m] = app.calculate_sharpe_ratios(returns_wide_df, m).to_dict()
        tm = turnover_df[turnover_df["model"] == m]
        s = tm[["bin", "% buy", "% hold"]].groupby("bin").mean()
        turnover_summary[m] = {"bins": [str(b) for b in s.index], "buy": s["% buy"].tolist(), "hold": s["% hold"].tolist()}
        hold[m] = {str(b): {"dates": dates(g.index), "hold": g["% hold"].tolist()} for b, g in tm.groupby("bin")}

    # Factors (percent per month) on the full monthly range the dashboard uses
    costs = app.load_costs()
    practice = app.load_practice()
    all_dates = idx
    for extra in (costs, practice):
        if extra is not None:
            all_dates = all_dates.union(extra.index.unique())
    f = ff.reindex(all_dates)
    factors = {"dates": dates(all_dates), **{c: f[c].tolist() for c in ["mkt_excess", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]}}

    write(DATA / "main.json", {
        "dates": dates(idx),
        "models": models,
        "modelIndex": index,
        "modelGuide": app.MODEL_GUIDE,
        "modelColors": app.model_colors(models),
        "returns": returns,
        "table1": table1,
        "sharpe": sharpe,
        "turnoverSummary": turnover_summary,
        "hold": hold,
        "factors": factors,
        "realisticApproach": app.REALISTIC_APPROACH,
    })

    if costs is not None:
        out = {"dates": dates(costs.index.unique().sort_values()), "data": {}}
        for (m, u), d in costs.groupby(["model", "universe"], sort=False):
            out["data"].setdefault(m, {})[u] = {"dates": dates(d.index), **{
                c: d[c].tolist() for c in ["ls_gross", "ls_net", "long_gross", "turnover_long", "turnover_short", "short_illiquid"]}}
        write(DATA / "costs.json", out)

    if practice is not None:
        out = {"firstMonth": practice.index.min().strftime("%Y-%m-%d"), "models": list(practice["model"].unique()),
               "approaches": list(practice["approach"].unique()), "data": {}}
        for (m, a), d in practice.groupby(["model", "approach"], sort=False):
            out["data"].setdefault(m, {})[a] = {"dates": dates(d.index), **{
                c: d[c].tolist() for c in ["gross", "net_spread_estimate", "turnover", "trades"]}}
        write(DATA / "practice.json", out)

    # Anomaly zoo, on the months the model portfolios cover (the app merges them on date, inner)
    zoo = app.load_zoo_returns().set_index("date")
    zoo = zoo.loc[zoo.index.isin(idx)]
    write(DATA / "zoo.json", {"dates": dates(zoo.index), "names": list(zoo.columns),
                              "returns": [zoo[c].tolist() for c in zoo.columns]}, )
    return models


def build_model_files(models):
    """Per model and test year: the pipeline diagram, its column groups, and the top feature importances."""
    from sklearn.compose import ColumnTransformer

    manifest = {}
    for m in models:
        files = app.model_files(m)
        manifest[m] = sorted(files)
        for year, path in sorted(files.items()):
            out = {"year": year}
            try:
                pipe = app.load_model_pipeline(path)
            except Exception as e:  # noqa: BLE001
                out["error"] = f"Could not load the pipeline: {e}"
                write(DATA / "models" / m / f"{year}.json", out)
                continue
            try:
                out["pipelineHtml"] = pipe._repr_html_()
            except AttributeError:
                pass
            groups = []
            for step in pipe.named_steps.values():
                if isinstance(step, ColumnTransformer):
                    for name, _, cols in list(step.transformers)[:2]:
                        groups.append({"name": name, "columns": [str(c) for c in cols]})
            out["columnGroups"] = groups
            est = pipe.named_steps[list(pipe.named_steps)[-1]]
            details = app.get_feature_importance_details(est, pipe[:-1].get_feature_names_out())
            if details is not None and details["series"] is not None:
                s = details["series"]
                s = s.sort_values(key=lambda x: x.abs(), ascending=False) if details["sort_by_absolute_value"] \
                    else s.sort_values(ascending=False)
                s = s.head(50)
                out["importance"] = {k: details[k] for k in ("subheader", "x_label", "title", "caption")}
                out["importance"].update(names=[str(i) for i in s.index], values=s.tolist())
            write(DATA / "models" / m / f"{year}.json", out)
    write(DATA / "model_years.json", manifest)


def build_background():
    """Toy tables of the Background page (same random draws as the app)."""
    out = {"timeline": {}}
    for n in range(1, 5):
        _, _, df = app.prepare_combined_data(num_firms=n)
        np.random.seed(42)
        fake = np.random.normal(0.01, 0.05, size=len(df))
        out["timeline"][n] = {"columns": list(df.columns), "rows": df.values.tolist(), "fake": fake.tolist()}

    toy_rets, toy_signals, combined = app.prepare_combined_data(num_firms=1)
    out["toyRets"] = {"columns": list(toy_rets.columns), "rows": toy_rets.values.tolist()}
    out["toySignals"] = {"columns": list(toy_signals.columns), "rows": toy_signals.values.tolist()}
    out["combined"] = {"columns": list(combined.columns), "rows": combined.values.tolist()}

    # Portfolio construction example (mirrors background_page, tab 6; num_firms is 1 there)
    num_firms, train_window = 1, 5
    _, _, p = app.prepare_combined_data(num_firms=2)
    np.random.seed(42)
    fake = np.random.normal(0.01, 0.05, size=len(p))
    keep = list(range(train_window * num_firms - 1, len(p)))
    p["prediction"] = np.where(p.index.isin(keep), fake, np.nan)
    p = p.dropna(subset=["prediction"])
    p["portfolio_assignment"] = p.groupby("yyyymm")["prediction"].transform(
        lambda x: np.ceil(x.rank(method="first") * 2 / len(x)).astype(int))
    p = p[["yyyymm", "firm_id", "ret", "prediction", "portfolio_assignment"]]
    pr = p.groupby(["yyyymm", "portfolio_assignment"])["ret"].mean().unstack(level=-1)
    pr.columns = [f"Portfolio{c}" for c in pr.columns]
    pr["Long-Short"] = pr["Portfolio2"] - pr["Portfolio1"]
    out["portfolios"] = {"columns": ["", *p.columns], "rows": [[i, *r] for i, r in zip(p.index, p.values.tolist())]}
    out["portRets"] = {"columns": ["yyyymm", *pr.columns], "rows": [[i, *r] for i, r in zip(pr.index, pr.values.tolist())]}

    import nbformat
    nb = nbformat.read(Path("stock_prediction_and_eval.ipynb").open(encoding="utf-8"), as_version=4)
    out["notebook"] = [{"type": c.cell_type, "source": c.source} for c in nb.cells if c.cell_type in ("markdown", "code")]
    write(DATA / "background.json", out)


def main():
    if DATA.exists():
        shutil.rmtree(DATA)
    print("Building", OUT)
    models = build_main()
    build_background()
    build_model_files(models)
    (OUT / ".nojekyll").touch()
    print("done")


if __name__ == "__main__":
    main()
