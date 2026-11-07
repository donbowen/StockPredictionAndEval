"""Tie-aware equal-weighted quantile portfolios.

Tie policy (fractional allocation, never uses realised returns)
---------------------------------------------------------------
In each month the n stocks are sorted by prediction.  Sorted position i
(0-based) occupies the interval [i, i+1) of the rank axis [0, n); bin k of K
owns [(k-1)n/K, kn/K).  Stocks with identical predictions (after rounding to
``TIE_DECIMALS``) form a tie group occupying the union of their positions
[a, b); the group's rank mass is spread **equally** over its members, so each
member holds ``overlap([a,b), bin k) / (b-a)`` units of bin k.  Consequences:

* tied stocks are treated symmetrically (identical weights);
* every bin has total weight exactly n/K, so portfolios are never empty and
  always equal-sized in effective terms, even with massive ties;
* if every prediction is identical (no cross-sectional dispersion) each stock
  sits 1/K in every bin: all bins earn the equal-weighted universe return and
  the long-short return is exactly 0 (no position taken).  Such months are
  flagged ``no_dispersion`` and their LS return of 0 is *kept* in the series.

Returns are DECIMALS everywhere in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE_DECIMALS = 12  # predictions are in percent; differences below 1e-12 are float noise


def bin_weights(df: pd.DataFrame, pred_col: str, nport: int = 5, date_col: str = "date"):
    """Return (W, tie_stats).  W[i, k] = share of stock i's weight in bin k+1 (rows sum to 1)
    scaled so each (date, bin) column sums to n_date / nport."""
    p = df[pred_col].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError(f"non-finite predictions in {pred_col}")
    key = pd.DataFrame({"d": df[date_col].to_numpy(), "p": np.round(p, TIE_DECIMALS)})
    g = key.groupby("d", sort=False)["p"]
    rmin = g.rank(method="min").to_numpy()
    rmax = g.rank(method="max").to_numpy()
    n = g.transform("size").to_numpy().astype(np.float64)
    a, b = rmin - 1.0, rmax
    W = np.empty((len(df), nport))
    for k in range(nport):
        lo, hi = k * n / nport, (k + 1) * n / nport
        W[:, k] = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None) / (b - a)
    return W, key.assign(tie_size=(b - a), n=n)


def tie_diagnostics(df: pd.DataFrame, pred_col: str, W: np.ndarray, key: pd.DataFrame,
                    date_col: str = "date") -> pd.DataFrame:
    """Per-month tie frequency and dispersion flags."""
    k = key.assign(in_tie=(key["tie_size"] > 1), split=((W > 1e-12).sum(1) > 1), pred=df[pred_col].to_numpy())
    g = k.groupby("d", sort=True)
    out = pd.DataFrame({
        "n_stocks": g.size(),
        "n_unique_pred": g["p"].nunique(),
        "share_in_ties": g["in_tie"].mean(),
        "max_tie_share": g["tie_size"].max() / g.size(),
        "share_fractional": g["split"].mean(),
        "pred_std": g["pred"].std(),
    })
    out["no_dispersion"] = out["n_unique_pred"] == 1
    out.index.name = date_col
    return out


def portfolio_returns(df: pd.DataFrame, W: np.ndarray, nport: int = 5, ret_col="ret",
                      date_col="date"):
    """Equal-weighted bin returns (decimals), bin stats and the long-short series."""
    r = df[ret_col].to_numpy(dtype=np.float64)
    if np.isnan(r).any():
        raise ValueError("missing returns reached portfolio construction")
    d = df[date_col].to_numpy()
    n = pd.Series(d).groupby(d).transform("size").to_numpy().astype(np.float64)
    wn = W / (n / nport)[:, None]                       # normalised: each (date, bin) sums to 1
    rows, tall = [], []
    dates = pd.Index(np.unique(d))
    for k in range(nport):
        w = wn[:, k]
        t = pd.DataFrame({"d": d, "w": w, "wr": w * r, "w2": w * w, "wr2": w * r * r, "pos": (w > 1e-12)})
        g = t.groupby("d")
        mean = g["wr"].sum()
        var = (g["wr2"].sum() - mean ** 2).clip(lower=0)
        stat = pd.DataFrame({
            "bin": k + 1, "ret": mean, "std": np.sqrt(var), "nfirms": g["pos"].sum(),
            "eff_n": 1.0 / g["w2"].sum(), "max_weight": t.groupby("d")["w"].max(),
            "weight_sum": g["w"].sum(),
        })
        stat.index.name = "date"
        tall.append(stat.reset_index())
        rows.append(mean.rename(k + 1))
    wide = pd.concat(rows, axis=1)
    wide.index.name = "date"
    wide["LS"] = wide[nport] - wide[1]
    assert np.allclose(wide["LS"], wide[nport] - wide[1], atol=0, rtol=0)
    assert np.allclose(pd.concat(tall)["weight_sum"], 1.0), "bin weights must sum to 1"
    return wide, pd.concat(tall, ignore_index=True), wn


def bin_turnover(df: pd.DataFrame, wn: np.ndarray, nport: int = 5, date_col="date") -> pd.DataFrame:
    """One-way turnover per bin (and LS) from normalised weights, ignoring drift:
    0.5 * sum_i |w_it - w_i,t-1| with a stock absent from the bin counted as weight 0."""
    out = []
    base = pd.DataFrame({"permno": df["permno"].to_numpy(), "date": df[date_col].to_numpy()})
    for k in range(nport):
        piv = base.assign(w=wn[:, k]).pivot(index="date", columns="permno", values="w").fillna(0.0)
        out.append(0.5 * piv.diff().abs().sum(1).iloc[1:].rename(f"bin{k + 1}"))
    t = pd.concat(out, axis=1)
    long_short = []
    piv5 = base.assign(w=wn[:, nport - 1] - wn[:, 0]).pivot(index="date", columns="permno", values="w").fillna(0.0)
    t["LS"] = 0.5 * piv5.diff().abs().sum(1).iloc[1:]  # of the net long-short book (2 gross)
    return t
