"""Performance statistics and factor regressions (all inputs are DECIMAL monthly returns)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FACTOR_MODELS = {
    "r^e": "1",
    "CAPM": "mkt_excess",
    "FF3": "mkt_excess + SMB + HML",
    "FF4": "mkt_excess + SMB + HML + Mom",
    "FF5": "mkt_excess + SMB + HML + RMW + CMA",
    "FF6": "mkt_excess + SMB + HML + RMW + CMA + Mom",
}
WEALTH_NOTE = ("wealth = cumprod(1 + r_LS), start $1. Exposure convention: $1 long the top quintile and "
               "$1 short the bottom quintile per $1 of strategy equity, rebalanced monthly, before "
               "transaction costs, borrow fees and financing/short-rebate effects.")


def newey_west_lags(T: int) -> int:
    """Newey-West (1994) plug-in rule of thumb: floor(4 (T/100)^(2/9)) (4 for T=240)."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def wealth_index(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def max_drawdown(r: pd.Series) -> float:
    w = np.concatenate([[1.0], wealth_index(r).to_numpy()])
    return float((w / np.maximum.accumulate(w) - 1.0).min())


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    assert r.abs().max() < 10, "returns look like percent, expected decimals"
    n = len(r)
    end = float(wealth_index(r).iloc[-1])
    return {
        "months": n,
        "CAGR": end ** (12.0 / n) - 1.0,
        "ann_mean": 12.0 * r.mean(),
        "ann_vol": np.sqrt(12.0) * r.std(ddof=1),
        "Sharpe": np.sqrt(12.0) * r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan,
        "max_drawdown": max_drawdown(r),
        "ending_wealth": end,
    }


def performance_table(ls_wide: pd.DataFrame, starts=("2000-01", "2002-01", "2005-01", "2010-01"),
                      end="2019-12") -> pd.DataFrame:
    """ls_wide: columns = model name, index = month, values = decimal LS returns."""
    rows = []
    for s in starts:
        sub = ls_wide.loc[(ls_wide.index >= pd.Timestamp(s + "-01")) & (ls_wide.index <= pd.Timestamp(end + "-28"))]
        for m in sub.columns:
            rows.append({"model": m, "start": s, "end": end, **perf_stats(sub[m])})
    return pd.DataFrame(rows)


def factor_regressions(returns: pd.DataFrame, ff_eval: pd.DataFrame, subtract_rf: dict,
                       lags: int | None = None) -> pd.DataFrame:
    """Alphas with HAC (Newey-West) t-stats against CONTEMPORANEOUS factors.

    returns      : decimal monthly returns, columns = portfolios
    subtract_rf  : {column: bool}; True for long-only portfolios, False for long-short spreads
    Regressions are run in percent (factors are in percent).  Alpha in % per month.
    """
    ff = ff_eval.set_index("date")
    rows = []
    for col in returns.columns:
        y = returns[col].dropna() * 100.0
        df = pd.concat([y.rename("y"), ff], axis=1, join="inner")
        if subtract_rf[col]:
            df["y"] = df["y"] - df["RF"]
        L = newey_west_lags(len(df)) if lags is None else lags
        for name, rhs in FACTOR_MODELS.items():
            res = smf.ols(f"y ~ {rhs}", data=df).fit(cov_type="HAC", cov_kwds={"maxlags": L})
            rows.append({"portfolio": col, "factor_model": name, "alpha_pct_per_month": res.params["Intercept"],
                         "t_stat_hac": res.tvalues["Intercept"], "n": len(df), "hac_lags": L})
    return pd.DataFrame(rows)
