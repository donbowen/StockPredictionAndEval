"""Chronologically valid histogram gradient boosting predictors (same protocol as ``nn.py``).

Protocol for test year Y:
  * preprocessor fit on all rows with year < Y (training-only statistics),
  * trees grown on years < Y-V, the number of trees chosen on the V most recent years before Y
    (never sklearn's built-in early stopping, whose validation rows are drawn at random),
  * the test year is touched only by ``predict`` after the number of trees is fixed.

Hyper-parameters are fixed in advance and were not screened on any data, so no model-design choice
can look at a test year.

Targets:
  * ``return``    - next month's return in percent (squared error),
  * ``rank``      - within-month rank of the return in [-0.5, 0.5] (squared error),
  * ``quintile``  - 5-class classifier on the within-month return quintile; the signal is the
                    expected quintile, sum_k k * P(k).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from . import data as D
from .runner import Backtest, Config, code_hash


@dataclass
class HGBSpec:
    name: str
    target: str = "return"           # "return" | "rank" | "quintile"
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 1000
    l2_regularization: float = 1.0
    max_iter: int = 300
    check_every: int = 10
    patience_checks: int = 3         # stop after this many checks without improvement
    clip_q: tuple = (0.005, 0.995)   # training-row quantiles that clip validation returns (same as the MLPs)


def make_target(y: np.ndarray, month: np.ndarray, target: str) -> np.ndarray:
    """Training target, computed from training rows only."""
    if target == "return":
        return y.copy()
    rank = pd.Series(y).groupby(month).rank(pct=True).to_numpy()
    if target == "rank":
        return rank - 0.5
    if target == "quintile":
        return np.clip(np.ceil(rank * 5), 1, 5).astype(np.int8)
    raise ValueError(target)


def monthly_corr(pred: np.ndarray, y: np.ndarray, month: np.ndarray) -> float:
    """Mean over months of the cross-sectional Pearson correlation (the MLPs' early-stopping score)."""
    d = pd.DataFrame({"p": pred, "y": y, "m": month})
    if d["p"].nunique() < 2:
        return -np.inf
    return float(d.groupby("m")[["p", "y"]].corr().iloc[0::2, 1].mean())


def _signal(est, Z) -> np.ndarray:
    """The ranking signal: the prediction, or for the classifier the expected quintile."""
    if isinstance(est, HistGradientBoostingClassifier):
        return est.predict_proba(Z) @ est.classes_.astype(np.float64)
    return est.predict(Z)


def fit_hgb(Z_tr, t_tr, Z_va, y_va, month_va, spec: HGBSpec, seed: int):
    """Grow trees in steps of ``check_every``; keep the count with the best validation score."""
    cls = HistGradientBoostingClassifier if spec.target == "quintile" else HistGradientBoostingRegressor
    est = cls(learning_rate=spec.learning_rate, max_leaf_nodes=spec.max_leaf_nodes,
              min_samples_leaf=spec.min_samples_leaf, l2_regularization=spec.l2_regularization,
              max_iter=spec.check_every, early_stopping=False, warm_start=True, random_state=seed)
    best, best_iter, hist = -np.inf, 0, []
    for n in range(spec.check_every, spec.max_iter + 1, spec.check_every):
        est.set_params(max_iter=n)
        est.fit(Z_tr, t_tr)
        score = monthly_corr(_signal(est, Z_va), y_va, month_va)
        hist.append({"iter": n, "val_corr": score})
        if score > best:
            best, best_iter = score, n
        elif n - best_iter >= spec.check_every * spec.patience_checks:
            break
    return est, best_iter, best, hist


def keep_first_rounds(est, n_iter: int):
    """Drop the rounds after the best one, so the saved model predicts exactly what the backtest reports."""
    if n_iter > len(est._predictors):
        raise ValueError(f"model has fewer than {n_iter} rounds")
    est._predictors = est._predictors[:n_iter]
    return est


class HGBBacktest(Backtest):
    def __init__(self, cfg: Config, specs: list[HGBSpec], rank_ols: bool = True):
        super().__init__(cfg)
        self.specs = {s.name: s for s in specs}
        self.rank_ols = rank_ols

    def run_year_hgb(self, Y: int):
        cfg, run = self.cfg, self.rd
        names = list(self.specs) + (["OLS-rank"] if self.rank_ols else [])
        todo = [m for m in names if not run.is_complete(m, Y, self.chash)]
        if not todo:
            print(f"[{Y}] already complete - skipping", flush=True)
            return
        V = cfg.inner_val_years
        s_te, e_te = self.bounds[Y]
        s_va = self.bounds[Y - V][0]
        assert (self.year[:s_va] < Y - V).all() and (self.year[s_va:s_te] >= Y - V).all() and (self.year[s_va:s_te] < Y).all()
        t0 = time.time()
        pre = self._preprocess(0, s_te)                    # training-only (years < Y) statistics
        Z_all = pre.transform(self.X.iloc[:e_te]).astype(np.float32, copy=False)
        Z_te = pre.transform(self.X.iloc[s_te:e_te]).astype(np.float32, copy=False)   # the test year's own cross-sections
        assert np.isfinite(Z_all).all() and np.isfinite(Z_te).all()
        Z_tr, Z_va = Z_all[:s_va], Z_all[s_va:s_te]
        y_tr, y_va, y_te = self.y[:s_va], self.y[s_va:s_te], self.y[s_te:e_te]
        months = D.month_index(self.panel["date"])
        lo, hi = np.quantile(y_tr, (0.005, 0.995))         # training rows only
        y_va_c = np.clip(y_va, lo, hi)
        t_pre = time.time() - t0

        for m in todo:
            t1 = time.time()
            if m == "OLS-rank":
                # same target swap on the whole training window, so the effect of the target can be seen without trees
                est = LinearRegression().fit(Z_all[:s_te], make_target(self.y[:s_te], months[:s_te], "rank"))
                pred = est.predict(Z_te)
                extra = {"train_rows": int(s_te)}   # trains on the whole window: its train_end and val_* fields below don't apply
            else:
                spec = self.specs[m]
                seed = cfg.seed + 1000 * Y
                est, best_iter, best, hist = fit_hgb(Z_tr, make_target(y_tr, months[:s_va], spec.target), Z_va, y_va_c,
                                                     months[s_va:s_te], spec, seed)
                pred = _signal(keep_first_rounds(est, best_iter), Z_te)
                extra ={"spec": asdict(spec), "seed": seed, "best_iter": best_iter, "best_val_corr": best,
                         "iters_grown": hist[-1]["iter"], "history": hist, "train_rows": int(s_va)}
            assert np.isfinite(pred).all() and len(pred) == e_te - s_te
            joblib.dump(Pipeline([("preprocess", copy.deepcopy(pre)), ("model", est)]), run.p("models", m, Y, "joblib"), compress=3)
            pd.DataFrame({"permno": self.panel.permno.to_numpy()[s_te:e_te], "date": self.panel.date.to_numpy()[s_te:e_te],
                          "signal_date": self.panel.signal_date.to_numpy()[s_te:e_te], "ret": y_te, "pred": pred}
                         ).to_parquet(run.p("predictions", m, Y, "parquet"))
            d = self.panel.date
            meta = {"model": m, "year": Y, "config_hash": self.chash, "panel_key": self.panel_key, "code_hash": code_hash(),
                    "train_start": str(d.iloc[0].date()), "train_end": str(d.iloc[s_va - 1].date()),
                    "val_start": str(d.iloc[s_va].date()), "val_end": str(d.iloc[s_te - 1].date()),
                    "test_start": str(d.iloc[s_te].date()), "test_rows": int(e_te - s_te),
                    "n_features": int(Z_all.shape[1]), "pred_std": float(np.std(pred)),
                    "seconds": {"shared_preprocess_and_transform": round(t_pre, 1), "fit_and_predict": round(time.time() - t1, 1)},
                    "finished_utc": datetime.now(timezone.utc).isoformat(), **extra}
            run.p("fits", m, Y, "json").write_text(json.dumps(meta, indent=1, default=float))   # written last = completion marker
            run.log(event="fit_done", model=m, year=Y, best_iter=extra.get("best_iter"), seconds=meta["seconds"])
            print(f"[{Y}] {m:14s} best_iter={extra.get('best_iter')} val_corr={extra.get('best_val_corr')} "
                  f"pred_std={np.std(pred):.4f} {time.time() - t1:.0f}s (pre {t_pre:.0f}s)", flush=True)
        del Z_all, Z_te

    def run_hgb(self, years: list[int] | None = None):
        for Y in (years or self.cfg.years()):
            self.run_year_hgb(Y)
