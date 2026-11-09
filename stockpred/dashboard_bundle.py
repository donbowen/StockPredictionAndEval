"""Convert the corrected linear, MLP and HGB runs into the file layout the Streamlit dashboard reads
(returns_wide.csv, port_stats_tall.csv, turnover.csv) plus a small model index.

Dashboard model names must not contain underscores (the app parses ``ret_<model>_bin<k>``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio import bin_weights, portfolio_returns
from .report import load_predictions

NPORT = 5
# dashboard name -> (run directory name, model name inside that run, one-line description)
MODELS = {
    "OLS": ("linear_v1", "OLS", "OLS"),
    "Ridge": ("linear_v1", "Ridge", "Ridge, default alpha=1 (baseline)"),
    "RidgeCV": ("linear_v1", "RidgeCV", "Ridge, alpha chosen by chronological validation"),
    "Lasso": ("linear_v1", "Lasso", "Lasso, default alpha=1 (baseline)"),
    "LassoCV": ("linear_v1", "LassoCV", "Lasso, alpha chosen by chronological validation"),
    "MLP-rank-256-128-64": ("nn_v1", "MLP_rank_256x128x64_do0.3",
                            "MLP 256-128-64, dropout 0.3, within-month rank target, 5-seed average"),
    "MLP-rank-128-64": ("nn_v1", "MLP_rank_128x64", "MLP 128-64, within-month rank target, 5-seed average"),
    "MLP-huber-128-64": ("nn_v1", "MLP_huber_128x64", "MLP 128-64, Huber loss on clipped return, 5-seed average"),
    "OLS-rank": ("hgb_v1", "OLS-rank", "OLS on the within-month rank target"),
    "HGB-return": ("hgb_v1", "HGB-return", "Histogram gradient boosting on next month's return"),
    "HGB-rank": ("hgb_v1", "HGB-rank", "Histogram gradient boosting on the within-month rank target"),
    "HGB-quintile": ("hgb_v1", "HGB-quintile", "Histogram gradient boosting classifier on the return quintile, sorted on expected quintile"),
}


def weight_turnover(df: pd.DataFrame, wn: np.ndarray, nport: int = NPORT) -> pd.DataFrame:
    """Dashboard turnover schema from normalised bin weights (each bin's weights sum to 1):
    % buy  = weight of this month's bin that was not held last month,
    % sell = weight of last month's bin that is no longer held,
    % hold = 1 - % buy."""
    base = pd.DataFrame({"permno": df["permno"].to_numpy(), "date": df["date"].to_numpy()})
    rows = []
    for k in range(nport):
        piv = base.assign(w=wn[:, k]).pivot(index="date", columns="permno", values="w").fillna(0.0)
        d = piv.diff().iloc[1:]
        buy, sell = d.clip(lower=0).sum(1), (-d).clip(lower=0).sum(1)
        rows.append(pd.DataFrame({"bin": k + 1, "% buy": buy, "% sell": sell, "% hold": 1.0 - buy}).reset_index())
    return pd.concat(rows, ignore_index=True)


def build_bundle(out_root: Path, out_dir: Path, years: list[int]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted({r for r, _, _ in MODELS.values()})
    preds = {}
    for run in runs:
        names = [m for _, (r, m, _) in MODELS.items() if r == run]
        preds[run] = load_predictions(out_root / run, names, years)
    key = ["permno", "date"]
    ref = preds[runs[0]]
    for run in runs[1:]:
        p = preds[run]
        assert len(p) == len(ref) and (p[key].to_numpy() == ref[key].to_numpy()).all() \
            and np.array_equal(p["ret"].to_numpy(), ref["ret"].to_numpy()), f"{run} disagrees with {runs[0]} on keys/returns"
    pred = ref[key + ["signal_date", "ret"]].copy()
    pred["ret"] = pred["ret"] / 100.0                       # percent -> decimals
    for dash, (run, m, _) in MODELS.items():
        pred[f"pred_{dash}"] = preds[run][f"pred_{m}"].to_numpy()
    wides, talls, turns = [], [], []
    for dash in MODELS:
        W, _ = bin_weights(pred, f"pred_{dash}", NPORT)
        wide, tall, wn = portfolio_returns(pred, W, NPORT)
        assert np.allclose(wide["LS"], wide[NPORT] - wide[1], atol=1e-15)
        w = wide.copy()
        w.columns = [f"ret_{dash}_bin{c}" for c in wide.columns]
        wides.append(w)
        t = tall.copy()
        t["model"] = dash
        talls.append(t.reset_index() if "date" not in t.columns else t)
        tr = weight_turnover(pred, wn)
        tr["model"] = dash
        turns.append(tr)
    returns_wide = pd.concat(wides, axis=1)
    returns_wide.index.name = "date"
    returns_wide.to_csv(out_dir / "returns_wide.csv")
    tall = pd.concat(talls, ignore_index=True)
    tall[["model", "bin", "date", "ret", "std", "nfirms"]].to_csv(out_dir / "port_stats_tall.csv", index=False)
    pd.concat(turns, ignore_index=True)[["model", "bin", "date", "% buy", "% sell", "% hold"]].to_csv(
        out_dir / "turnover.csv", index=False)
    index = {d: {"run": r, "model": m, "description": desc} for d, (r, m, desc) in MODELS.items()}
    (out_dir / "models_index.json").write_text(json.dumps(index, indent=1))
    return index
