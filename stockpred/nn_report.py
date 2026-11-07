"""Evaluate NN runs next to the corrected linear models (same panel keys, same tie-aware portfolios)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .evaluation import WEALTH_NOTE, factor_regressions, performance_table
from .portfolio import bin_turnover, bin_weights, portfolio_returns, tie_diagnostics
from .report import load_predictions, monthly_rank_similarity

NPORT = 5


def monthly_ic(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    out = {}
    g = pred.groupby("date")
    ry = g["ret"].rank()
    for m in models:
        rp = g[f"pred_{m}"].rank()
        c = pd.DataFrame({"date": pred["date"], "a": rp, "b": ry})
        out[m] = c.groupby("date").apply(lambda x: x["a"].corr(x["b"]), include_groups=False)
    return pd.DataFrame(out)


def evaluate_nn(nn_root: Path, nn_models: list[str], linear_root: Path, linear_models: list[str], years: list[int],
                data_dir, starts=("2000-01", "2002-01", "2005-01", "2010-01"), out_name="results"):
    out = nn_root / out_name
    out.mkdir(exist_ok=True)
    pn = load_predictions(nn_root, nn_models, years)
    pl = load_predictions(linear_root, linear_models, years)
    key = ["permno", "date"]
    assert len(pn) == len(pl) and (pn[key].to_numpy() == pl[key].to_numpy()).all() \
        and np.array_equal(pn["ret"].to_numpy(), pl["ret"].to_numpy()), "NN and linear runs disagree on keys/returns"
    pred = pn.merge(pl[key + [f"pred_{m}" for m in linear_models]], on=key, validate="1:1")
    models = linear_models + nn_models
    checks = {"models": models}
    assert not pred.duplicated(key).any()
    for m in models:
        assert np.isfinite(pred[f"pred_{m}"]).all()
    assert (D.month_index(pred["date"]) - D.month_index(pred["signal_date"]) == 1).all()
    for f in sorted((nn_root / "fits").glob("*.json")):
        j = json.loads(f.read_text())
        if j["model"] in nn_models:
            assert pd.Timestamp(j["train_end"]) < pd.Timestamp(j["val_start"]) and \
                pd.Timestamp(j["val_end"]) < pd.Timestamp(j["test_start"]), f"temporal order violated in {f.name}"
    checks.update(finite_predictions=True, unique_keys=True, signal_precedes_return=True,
                  nn_train_val_test_strictly_ordered=True)
    pred["ret"] = pred["ret"] / 100.0
    wides = {}
    diags, turns = [], []
    for m in models:
        W, k = bin_weights(pred, f"pred_{m}", NPORT)
        wide, tall, wn = portfolio_returns(pred, W, NPORT)
        assert np.allclose(wide["LS"], wide[NPORT] - wide[1], atol=1e-15) and np.allclose(tall["weight_sum"], 1.0)
        assert (tall["nfirms"] >= 1).all()
        wides[m] = wide
        dg = tie_diagnostics(pred, f"pred_{m}", W, k)
        dg["model"] = m
        diags.append(dg.reset_index())
        tr = bin_turnover(pred, wn, NPORT)
        turns.append(pd.DataFrame({"model": m, "turnover_bin1": tr["bin1"], "turnover_bin5": tr[f"bin{NPORT}"], "turnover_LS": tr["LS"]}).reset_index())
    checks.update(ls_equals_bin5_minus_bin1=True, weights_sum_to_one=True, no_empty_portfolios=True)
    diag = pd.concat(diags)
    tie = diag.groupby("model").agg(months=("date", "size"), no_dispersion_months=("no_dispersion", "sum"),
                                    mean_share_in_ties=("share_in_ties", "mean")).reset_index()
    ls = pd.DataFrame({m: wides[m]["LS"] for m in models})
    b5 = pd.DataFrame({m: wides[m][NPORT] for m in models})
    b1 = pd.DataFrame({m: wides[m][1] for m in models})
    perf = performance_table(ls, starts, f"{max(years)}-12")
    ff = D.load_ff_eval(Path(data_dir) / "ff_factors.csv")
    rows = []
    for s in starts:
        msk = ls.index >= pd.Timestamp(s + "-01")
        cols, rf = {}, {}
        for m in models:
            cols[f"{m}|LS"], rf[f"{m}|LS"] = ls.loc[msk, m], False
            cols[f"{m}|bin5"], rf[f"{m}|bin5"] = b5.loc[msk, m], True
        r = factor_regressions(pd.DataFrame(cols), ff, rf)
        r["start"] = s
        rows.append(r)
    reg = pd.concat(rows)
    reg[["model", "portfolio"]] = reg["portfolio"].str.split("|", expand=True)
    ic = monthly_ic(pred, models)
    sim = monthly_rank_similarity(pred, models, ref="OLS")
    ic_y = ic.groupby(ic.index.year).mean()
    ic_summary = pd.DataFrame({"mean_monthly_rank_IC": ic.mean(), "IC_tstat": ic.mean() / (ic.std() / np.sqrt(len(ic)))})
    ic_summary["mean_spearman_vs_OLS"] = sim.mean()
    # returns spread of a top-minus-bottom view that excludes the tails: report extremes separately
    ls.to_csv(out / "ls_returns.csv")
    b5.to_csv(out / "bin5_returns.csv")
    b1.to_csv(out / "bin1_returns.csv")
    perf.to_csv(out / "performance_summary.csv", index=False)
    reg.to_csv(out / "factor_regressions.csv", index=False)
    ic_y.to_csv(out / "rank_ic_by_year.csv")
    ic_summary.to_csv(out / "rank_ic_summary.csv")
    tie.to_csv(out / "tie_summary.csv", index=False)
    pd.concat(turns).to_csv(out / "turnover.csv", index=False)
    (out / "checks.json").write_text(json.dumps({**checks, "wealth_convention": WEALTH_NOTE}, indent=1))
    return perf, reg, ic_summary, ic_y, tie, checks
