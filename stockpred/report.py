"""Assemble predictions -> portfolios -> performance tables -> markdown report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .evaluation import WEALTH_NOTE, factor_regressions, performance_table, perf_stats
from .portfolio import bin_turnover, bin_weights, portfolio_returns, tie_diagnostics
from .runner import MODEL_LABELS, MODEL_ORDER, Config

ORIGINAL_RETURNS = Path("output_portfolios/returns_wide.csv")   # saved results of the ORIGINAL (buggy) pipeline
NPORT = 5


def load_predictions(root: Path, models: list[str], years: list[int]) -> pd.DataFrame:
    wide = None
    for m in models:
        parts = [pd.read_parquet(root / "predictions" / f"{m}_{y}.parquet") for y in years]
        d = pd.concat(parts, ignore_index=True).rename(columns={"pred": f"pred_{m}"})
        if wide is None:
            wide = d
        else:
            key = ["permno", "date"]
            assert len(d) == len(wide) and (d[key].to_numpy() == wide[key].to_numpy()).all() \
                and np.array_equal(d["ret"].to_numpy(), wide["ret"].to_numpy()), "models disagree on keys/returns"
            wide[f"pred_{m}"] = d[f"pred_{m}"].to_numpy()
    return wide.sort_values(["date", "permno"]).reset_index(drop=True)


def load_fits(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "fits").glob("*.json")):
        m = json.loads(f.read_text())
        s = m.get("search") or {}
        rows.append({
            "model": m["model"], "year": m["year"], "train_rows": m["train_rows"], "n_features": m["n_features"],
            "alpha": m["alpha"], "alpha_over_n_train": m.get("alpha_over_n_train"),
            "alpha_ratio_to_alpha_max": s.get("alpha_ratio"), "val_mse_selected": (s.get("val_mse") or [None])[s.get("best_index", 0)] if s else None,
            "hit_lower_endpoint": s.get("hit_lower_endpoint"), "hit_upper_endpoint": s.get("hit_upper_endpoint"),
            "grid_expansions": s.get("expansions"), "inner_val_years": s.get("val_years"),
            "inner_train_rows": s.get("n_inner_train"), "coef_l2": m["coef_l2"], "coef_l1": m["coef_l1"],
            "coef_nonzero": m["coef_nonzero"], "convergence_warnings_final_fit": m["convergence_warnings_final_fit"],
            "convergence_warnings_inner_search": s.get("inner_convergence_warnings"),
            "pred_std": m["pred_std"], "train_end": m["train_end"], "test_start": m["test_start"],
            "seconds_final_fit": m["seconds"]["final_fit"], "seconds_shared_preprocess": m["seconds"]["shared_preprocess_and_transform"],
            "seconds_shared_search": m["seconds"]["shared_search"],
        })
    return pd.DataFrame(rows)


def monthly_rank_similarity(pred: pd.DataFrame, models: list[str], ref="OLS") -> pd.DataFrame:
    """Mean over months (by year) of the cross-sectional Spearman correlation with the OLS prediction."""
    rk = pred[["date"]].copy()
    g = pred.groupby("date")
    for m in models:
        rk[m] = g[f"pred_{m}"].rank(method="average")
    out = {}
    for m in models:
        if m == ref:
            continue
        c = pd.DataFrame({"date": rk["date"], "a": rk[m], "b": rk[ref]})
        mon = c.groupby("date").apply(lambda x: x["a"].corr(x["b"]), include_groups=False)
        out[m] = mon
    s = pd.DataFrame(out)
    s.index.name = "date"
    return s


def evaluate_run(cfg: Config, starts=("2000-01", "2002-01", "2005-01", "2010-01"), smoke=False):
    root = Path(cfg.out_root) / cfg.run_id
    out = root / "results"
    out.mkdir(exist_ok=True)
    years = cfg.years()
    models = [m for m in MODEL_ORDER if all((root / "fits" / f"{m}_{y}.json").exists() for y in years)]
    assert models, "no completed models to evaluate"
    pred = load_predictions(root, models, years)
    checks = {"models": models}

    # ---- structural checks
    assert not pred.duplicated(["permno", "date"]).any(), "duplicate stock-month keys"
    checks["unique_stock_month_keys"] = True
    for m in models:
        assert np.isfinite(pred[f"pred_{m}"]).all()
    checks["finite_predictions"] = True
    assert pred["ret"].notna().all()
    assert (pred["date"].dt.month - pred["signal_date"].dt.month).isin([1, -11]).all()
    assert (D.month_index(pred["date"]) - D.month_index(pred["signal_date"]) == 1).all()
    checks["signal_precedes_return_by_one_month"] = True
    fits = load_fits(root)
    for _, r in fits.iterrows():
        assert pd.Timestamp(r["train_end"]) < pd.Timestamp(r["test_start"])
        if r["inner_val_years"]:
            assert max(r["inner_val_years"]) < r["year"]
    checks["train_ends_before_test_starts"] = True
    checks["inner_validation_years_before_test_year"] = True
    assert pred["date"].dt.year.between(min(years), max(years)).all()
    checks["prediction_years"] = [int(pred.date.dt.year.min()), int(pred.date.dt.year.max())]
    assert (pred["ret"].abs() < 1e5).all(), "implausible returns - unit problem"  # genuine CRSP spikes up to ~2,000% exist; kept unwinsorised
    pred["ret"] = pred["ret"] / 100.0                                  # percent -> decimals for portfolio math
    assert pred["ret"].min() >= -1.0, "returns below -100% - unit problem"
    checks["return_units"] = "panel ret in percent; converted to decimals once here (min=%.3f max=%.3f)" % (
        pred["ret"].min(), pred["ret"].max())

    # ---- portfolios
    wides, talls, diags, turns = [], [], [], []
    for m in models:
        W, key = bin_weights(pred, f"pred_{m}", NPORT)
        wide, tall, wn = portfolio_returns(pred, W, NPORT)
        assert np.allclose(wide["LS"], wide[NPORT] - wide[1], rtol=0, atol=1e-15)
        d = tie_diagnostics(pred, f"pred_{m}", W, key)
        d["eff_n_bin_min"] = tall.groupby("date")["eff_n"].min()
        d["max_weight_any_bin"] = tall.groupby("date")["max_weight"].max()
        d["model"] = m
        diags.append(d.reset_index())
        t = tall.copy()
        t["model"] = m
        talls.append(t)
        w = wide.copy()
        w.columns = [f"ret_{m}_bin{c}" for c in wide.columns]
        wides.append(w)
        tr = bin_turnover(pred, wn, NPORT)
        tr = tr.stack().rename("turnover").reset_index()
        tr.columns = ["date", "bin", "turnover"]
        tr["model"] = m
        turns.append(tr)
    returns_wide = pd.concat(wides, axis=1)
    tall = pd.concat(talls, ignore_index=True)
    # tall/wide consistency + LS identity recomputed from the tall file
    piv = tall.pivot_table(index="date", columns=["model", "bin"], values="ret")
    for m in models:
        for k in range(1, NPORT + 1):
            assert np.allclose(piv[(m, k)].to_numpy(), returns_wide[f"ret_{m}_bin{k}"].to_numpy(), atol=1e-14)
        assert np.allclose((piv[(m, NPORT)] - piv[(m, 1)]).to_numpy(), returns_wide[f"ret_{m}_binLS"].to_numpy(), atol=1e-14)
    checks["tall_equals_wide"] = True
    checks["long_short_equals_bin5_minus_bin1"] = True
    assert np.allclose(tall["weight_sum"], 1.0)
    checks["portfolio_weights_sum_to_one_per_bin_month"] = True
    nm = pred.groupby("date").size()
    assert (tall.groupby(["model", "date"])["nfirms"].min() >= 1).all()
    checks["no_empty_portfolios"] = True

    diag = pd.concat(diags, ignore_index=True)
    diag.to_csv(out / "tie_diagnostics.csv", index=False)
    returns_wide.to_csv(out / "returns_wide.csv")
    tall.to_csv(out / "port_stats_tall.csv", index=False)
    pd.concat(turns).to_csv(out / "turnover.csv", index=False)
    tie_summary = diag.groupby("model").agg(
        months=("date", "size"), no_dispersion_months=("no_dispersion", "sum"),
        mean_share_in_ties=("share_in_ties", "mean"), max_tie_share_any_month=("max_tie_share", "max"),
        mean_share_fractional=("share_fractional", "mean"), min_effective_n_per_bin=("eff_n_bin_min", "min"),
        mean_stocks_per_month=("n_stocks", "mean")).reset_index()
    tie_summary.to_csv(out / "tie_summary.csv", index=False)

    # ---- performance
    ls = pd.DataFrame({m: returns_wide[f"ret_{m}_binLS"] for m in models})
    perf = performance_table(ls, starts, f"{max(years)}-12")
    perf["label"] = perf["model"].map(MODEL_LABELS)
    perf.to_csv(out / "performance_summary.csv", index=False)
    checks["wealth_convention"] = WEALTH_NOTE

    ff_eval = D.load_ff_eval(Path(_ff_path(cfg)))   # contemporaneous factors (not the lagged predictor version)
    reg_rows = []
    for s in starts:
        sub = returns_wide[returns_wide.index >= pd.Timestamp(s + "-01")]
        cols, rf = {}, {}
        for m in models:
            cols[f"{m}|LS"], rf[f"{m}|LS"] = sub[f"ret_{m}_binLS"], False       # spread: no RF
            for k in (1, NPORT):
                cols[f"{m}|bin{k}"], rf[f"{m}|bin{k}"] = sub[f"ret_{m}_bin{k}"], True   # long-only: excess of RF
        reg = factor_regressions(pd.DataFrame(cols), ff_eval, rf)
        reg["start"] = s
        reg_rows.append(reg)
    reg = pd.concat(reg_rows, ignore_index=True)
    reg[["strategy", "portfolio"]] = reg["portfolio"].str.split("|", expand=True)
    reg.to_csv(out / "factor_regressions.csv", index=False)

    # ---- tuning tables
    fits.to_csv(out / "fits_summary.csv", index=False)
    tuned = fits[fits["model"].isin(["RidgeCV", "LassoCV", "Ridge", "Lasso"])]
    tuned.to_csv(out / "selected_alpha_by_year.csv", index=False)
    ols = fits[fits.model == "OLS"].set_index("year")["coef_l2"]
    shr = fits[["model", "year", "coef_l2", "coef_nonzero", "n_features"]].copy()
    shr["coef_l2_ratio_to_OLS"] = shr["coef_l2"] / shr["year"].map(ols)
    sim = monthly_rank_similarity(pred, models)
    simy = sim.groupby(sim.index.year).mean().stack().rename("mean_monthly_spearman_vs_OLS").reset_index()
    simy.columns = ["year", "model", "mean_monthly_spearman_vs_OLS"]
    shr = shr.merge(simy, on=["model", "year"], how="left")
    shr.to_csv(out / "shrinkage_vs_ols.csv", index=False)

    # ---- comparison with the ORIGINAL saved results
    comp = compare_with_original(ls, starts, max(years))
    comp.to_csv(out / "comparison_vs_original.csv", index=False)

    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    write_markdown(cfg, out, perf, tie_summary, tuned, shr, comp, checks, smoke)
    print(f"evaluation written to {out}")
    return perf


def _ff_path(cfg):
    return str(D.resolve_data_dir(cfg.data_dir) / "ff_factors.csv")


def compare_with_original(ls_new: pd.DataFrame, starts, last_year: int) -> pd.DataFrame:
    if not ORIGINAL_RETURNS.exists():
        return pd.DataFrame()
    old = pd.read_csv(ORIGINAL_RETURNS, parse_dates=["date"]).set_index("date")
    old_ls = old.filter(like="_binLS")
    old_ls.columns = [c[len("ret_"):-len("_binLS")] for c in old_ls.columns]
    perf_old = performance_table(old_ls, starts, f"{last_year}-12")
    perf_old["pipeline"] = "ORIGINAL (old timing/imputer/CV) - historical"
    perf_new = performance_table(ls_new, starts, f"{last_year}-12")
    perf_new["pipeline"] = "corrected"
    both = pd.concat([perf_new, perf_old], ignore_index=True)
    corr = {m: ls_new[m].corr(old_ls[m]) for m in ls_new.columns if m in old_ls.columns}
    both["corr_LS_monthly_new_vs_old"] = both["model"].map(corr)
    return both


def _fmt(df, pct=("CAGR", "ann_mean", "ann_vol", "max_drawdown")):
    d = df.copy()
    for c in pct:
        if c in d:
            d[c] = (d[c] * 100).map("{:.1f}%".format)
    for c in ("Sharpe", "ending_wealth", "corr_LS_monthly_new_vs_old"):
        if c in d:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return d


def write_markdown(cfg, out, perf, tie_summary, tuned, shr, comp, checks, smoke):
    L = [f"# Corrected linear-model backtest: `{cfg.run_id}`" + ("  (SMOKE TEST - not results)" if smoke else ""), "",
         f"*Out-of-sample long-short (quintile 5 minus 1) returns, test years {cfg.first_test_year}-{cfg.last_test_year}.*", "",
         WEALTH_NOTE, "", "## Performance by evaluation start (all end 2019-12)", ""]
    p = _fmt(perf.drop(columns=["end", "label"]))
    L.append(p.to_markdown(index=False))
    L += ["", "## Tie / dispersion diagnostics", "", tie_summary.round(4).to_markdown(index=False), ""]
    a = tuned[tuned.model.isin(["RidgeCV", "LassoCV"])].copy()
    if len(a):
        cols = ["model", "year", "alpha", "alpha_over_n_train", "alpha_ratio_to_alpha_max", "train_rows", "coef_nonzero",
                "hit_lower_endpoint", "hit_upper_endpoint", "grid_expansions", "convergence_warnings_final_fit"]
        L += ["## Selected alpha by year", "", a[cols].round(8).to_markdown(index=False), ""]
    L += ["## Shrinkage and ranking similarity vs OLS", "", shr.round(4).to_markdown(index=False), ""]
    if len(comp):
        L += ["## Corrected vs original saved results", "", _fmt(comp.drop(columns=["end", "label"], errors="ignore")).to_markdown(index=False), ""]
    L += ["## Checks", "", "```json", json.dumps(checks, indent=1, default=str), "```"]
    (out / "report.md").write_text("\n".join(L), encoding="utf8")
