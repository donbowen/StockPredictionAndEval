"""Expanding-window annual backtest for linear models with full provenance.

For each test year Y:  train on return months in years < Y, predict year Y.
Hyper-parameters (RidgeCV / LassoCV) are chosen by a chronological search
*inside* the training window (see ``search.py``).  2020 and 2021+ are never
loaded (``max_return_year``), so the holdout cannot touch model selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from . import data as D
from .preprocessing import make_preprocessor
from .search import (chronological_year_folds, make_fold_stats, search_lasso, search_ridge,
                     sufficient_stats)

MODEL_ORDER = ["OLS", "Ridge", "RidgeCV", "Lasso", "LassoCV"]
MODEL_LABELS = {
    "OLS": "OLS",
    "Ridge": "Ridge (default alpha=1, baseline)",
    "RidgeCV": "RidgeCV (chronological validation)",
    "Lasso": "Lasso (default alpha=1, baseline)",
    "LassoCV": "LassoCV (chronological validation)",
}
PKG = Path(__file__).resolve().parent


@dataclass
class Config:
    run_id: str = "linear_v1"
    out_root: str = "output_corrected"
    cache_dir: str = "work_cache"
    data_dir: str | None = None
    first_test_year: int = 2000
    last_test_year: int = 2019
    min_return_year: int = 1975
    max_return_year: int = 2019       # 2020 = unused buffer year, 2021+ = untouched holdout
    inner_val_years: int = 3
    ridge_grid: list = field(default_factory=lambda: np.logspace(-4, 8, 25).tolist())
    lasso_n_alphas: int = 30
    lasso_ratio_min: float = 1e-4
    lasso_tol: float = 1e-4
    lasso_max_iter: int = 10_000
    default_alpha: float = 1.0        # sklearn default for the Ridge / Lasso baselines
    max_missing: float = 0.9
    winsor: tuple = (0.0025, 0.9975)
    seed: int = 0
    blas_threads: int = 12
    smoke_permnos: int = 0            # >0: random subsample of long-lived permnos (smoke test only)

    def years(self):
        return list(range(self.first_test_year, self.last_test_year + 1))

    def hash_payload(self, panel_key: str, code_hash: str) -> dict:
        d = asdict(self)
        for k in ("run_id", "out_root", "cache_dir", "data_dir", "blas_threads"):
            d.pop(k)
        d.update(panel_key=panel_key, code_hash=code_hash)
        return json.loads(json.dumps(d, default=list))


def code_hash() -> str:
    h = hashlib.sha256()
    for f in ("data.py", "preprocessing.py", "search.py", "runner.py"):
        h.update((PKG / f).read_bytes())
    return h.hexdigest()[:16]


def config_hash(cfg: Config, panel_key: str) -> str:
    return hashlib.sha256(json.dumps(cfg.hash_payload(panel_key, code_hash()), sort_keys=True).encode()).hexdigest()[:16]


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def environment_info(cfg: Config) -> dict:
    import scipy, sklearn, statsmodels, pyarrow
    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "pyarrow": pyarrow.__version__,
        "joblib": joblib.__version__, "cpu_logical": __import__("os").cpu_count(),
        "blas_threads_used": cfg.blas_threads, "model_fit_device": "CPU (sklearn); GPU not used",
        "git": git_info(), "code_hash": code_hash(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"], info["ram_available_gb_at_start"] = round(vm.total / 2**30, 1), round(vm.available / 2**30, 1)
    except ImportError:
        pass
    return info


def make_estimator(name: str, cfg: Config, alpha: float | None = None):
    """A brand-new, unfitted estimator on every call (never shared between fits)."""
    if name == "OLS":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=cfg.default_alpha)
    if name == "RidgeCV":
        return Ridge(alpha=alpha)
    if name == "Lasso":
        return Lasso(alpha=cfg.default_alpha, precompute=True, max_iter=cfg.lasso_max_iter, tol=cfg.lasso_tol)
    if name == "LassoCV":
        return Lasso(alpha=alpha, precompute=True, max_iter=cfg.lasso_max_iter, tol=cfg.lasso_tol)
    raise ValueError(name)


# --------------------------------------------------------------- run layout ---
class RunDir:
    def __init__(self, cfg: Config):
        self.root = Path(cfg.out_root) / cfg.run_id
        for s in ("predictions", "models", "fits", "coefs", "features"):
            (self.root / s).mkdir(parents=True, exist_ok=True)

    def p(self, kind, model, year, ext):
        return self.root / kind / (f"{model}_{year}.{ext}" if model else f"{year}.{ext}")

    def is_complete(self, model, year, chash) -> bool:
        m = self.p("fits", model, year, "json")
        if not (m.exists() and self.p("predictions", model, year, "parquet").exists()
                and self.p("models", model, year, "joblib").exists()):
            return False
        return json.loads(m.read_text()).get("config_hash") == chash

    def log(self, **kw):
        with open(self.root / "run_log.jsonl", "a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **kw}, default=str) + "\n")


# --------------------------------------------------------------- the backtest ---
class Backtest:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        t = time.time()
        data_dir = D.resolve_data_dir(cfg.data_dir)
        panel, self.cont, self.disc, self.ff_eval, self.panel_key, self.data_ids = D.get_panel(
            data_dir, Path(cfg.cache_dir), cfg.min_return_year, cfg.max_return_year)
        self.data_dir = data_dir
        if cfg.smoke_permnos:
            rng = np.random.default_rng(cfg.seed)
            early = set(panel.loc[panel.date.dt.year < 1995, "permno"])
            late = set(panel.loc[panel.date.dt.year >= max(cfg.years()), "permno"])
            pool = np.array(sorted(early & late))
            keep = rng.choice(pool, size=min(cfg.smoke_permnos, len(pool)), replace=False)
            panel = panel[panel.permno.isin(keep)].reset_index(drop=True)
        self.timing_examples = D.assert_timing(panel)
        self.factor_check = D.assert_factor_values_lagged(panel, self.ff_eval)
        assert panel.date.dt.year.max() <= cfg.max_return_year
        self.panel = panel
        self.features = self.cont + self.disc
        self.X = panel[self.features].copy()
        self.X.index = pd.MultiIndex.from_arrays([panel["permno"].to_numpy(), panel["date"].to_numpy()],
                                                 names=["permno", "date"])
        self.y = panel["ret"].to_numpy(dtype=np.float64)           # percent
        self.year = panel["date"].dt.year.to_numpy()
        self.bounds = {int(y): (int(np.searchsorted(self.year, y, "left")), int(np.searchsorted(self.year, y, "right")))
                       for y in np.unique(self.year)}
        self.chash = config_hash(cfg, self.panel_key)
        self.rd = RunDir(cfg)
        self.load_seconds = time.time() - t
        self._write_run_files()

    def _write_run_files(self):
        r, cfg = self.rd.root, self.cfg
        (r / "config.json").write_text(json.dumps({**asdict(cfg), "config_hash": self.chash,
                                                    "panel_key": self.panel_key}, indent=1, default=list))
        (r / "environment.json").write_text(json.dumps(environment_info(cfg), indent=1))
        p = self.panel
        (r / "data_manifest.json").write_text(json.dumps({
            "input_files": self.data_ids, "data_dir": str(self.data_dir), "panel_key": self.panel_key,
            "rows": len(p), "permnos": int(p.permno.nunique()),
            "return_months": [str(p.date.min().date()), str(p.date.max().date())],
            "signal_months": [str(p.signal_date.min().date()), str(p.signal_date.max().date())],
            "n_continuous_candidates": len(self.cont), "n_discrete_candidates": len(self.disc),
            "timing_examples": self.timing_examples, "factor_timing_check": self.factor_check,
            "holdout_policy": ("Only return months <= %d are loaded. 2020 is an unused buffer year (neither "
                               "trained on nor evaluated); 2021+ is an untouched holdout." % cfg.max_return_year),
            "units": "ret in percent in the panel; portfolio code converts to decimals",
        }, indent=1))
        (r / "features_candidates.json").write_text(json.dumps({"continuous": self.cont, "discrete": self.disc}))

    # -- helpers
    def _slice(self, a, b):
        return self.X.iloc[a:b], self.y[a:b]

    def _preprocess(self, a, b):
        """Fresh preprocessor fit on rows [a, b) only."""
        pre = make_preprocessor(self.cont, self.disc, self.cfg.max_missing, self.cfg.winsor)
        Xa, ya = self._slice(a, b)
        pre.fit(Xa, ya)
        return pre

    def run_year(self, Y: int, models: list[str]):
        cfg, run = self.cfg, self.rd
        todo = [m for m in models if not run.is_complete(m, Y, self.chash)]
        if not todo:
            print(f"[{Y}] all of {models} already complete (config match) - skipping", flush=True)
            return
        s_test, e_test = self.bounds[Y]
        n_tr = s_test                                           # all rows with year < Y
        assert (self.year[:n_tr] < Y).all() and (self.year[s_test:e_test] == Y).all()
        d_tr_max, d_te_min = self.panel.date.iloc[n_tr - 1], self.panel.date.iloc[s_test]
        assert d_tr_max < d_te_min, "training must end strictly before the test period"
        assert self.panel.signal_date.iloc[n_tr - 1] < d_te_min
        t0 = time.time()
        pre = self._preprocess(0, n_tr)
        Xtr = pre.transform(self.X.iloc[:n_tr]).astype(np.float64, copy=False)
        Xte = pre.transform(self.X.iloc[s_test:e_test]).astype(np.float64, copy=False)
        ytr, yte = self.y[:n_tr], self.y[s_test:e_test]
        fnames = [str(f) for f in pre.get_feature_names_out()]
        assert Xtr.shape[1] == len(fnames) and np.isfinite(Xtr).all() and np.isfinite(Xte).all()
        t_pre = time.time() - t0
        run.p("features", None, Y, "json").write_text(json.dumps(fnames))
        dropped = self._dropped_features(pre)

        searches, t_search = {}, 0.0
        if any(m in ("RidgeCV", "LassoCV") for m in todo):
            t1 = time.time()
            folds = self._inner_folds(Y, n_tr)
            if "RidgeCV" in todo:
                searches["RidgeCV"] = search_ridge(folds, cfg.ridge_grid)
            if "LassoCV" in todo:
                _, _, _, xty, _ = sufficient_stats(Xtr, ytr)
                amax = float(np.max(np.abs(xty)) / n_tr)
                searches["LassoCV"] = search_lasso(folds, amax, cfg.lasso_n_alphas, cfg.lasso_ratio_min,
                                                   tol=cfg.lasso_tol, max_iter=cfg.lasso_max_iter)
            del folds
            t_search = time.time() - t1

        for m in todo:
            sr = searches.get(m)
            est = make_estimator(m, cfg, None if sr is None else sr.alpha)
            t2 = time.time()
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                est.fit(Xtr, ytr)
            t_fit = time.time() - t2
            n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in wl)
            other = sorted({str(w.message)[:200] for w in wl if not issubclass(w.category, ConvergenceWarning)})
            pred = est.predict(Xte)
            assert np.isfinite(pred).all()
            pipe = Pipeline([("preprocess", copy.deepcopy(pre)), ("model", est)])   # per-fit copies, no shared refs
            n_probe = int((self.panel.date.iloc[s_test:e_test] == d_te_min).sum())
            probe = self.X.iloc[s_test:s_test + n_probe]          # whole first month (imputer needs the full cross-section)
            assert np.allclose(pipe.predict(probe), pred[:n_probe], rtol=1e-7, atol=1e-9), "saved pipeline != predictions"
            joblib.dump(pipe, run.p("models", m, Y, "joblib"), compress=3)           # saved immediately
            coef = np.asarray(est.coef_, dtype=np.float64).ravel()
            np.save(run.p("coefs", m, Y, "npy"), np.append(coef, float(est.intercept_)))
            pd.DataFrame({"permno": self.panel.permno.to_numpy()[s_test:e_test],
                          "date": self.panel.date.to_numpy()[s_test:e_test],
                          "signal_date": self.panel.signal_date.to_numpy()[s_test:e_test],
                          "ret": yte, "pred": pred}).to_parquet(run.p("predictions", m, Y, "parquet"))
            meta = {
                "model": m, "label": MODEL_LABELS[m], "year": Y, "config_hash": self.chash,
                "panel_key": self.panel_key, "code_hash": code_hash(), "seed": cfg.seed,
                "baseline_default_alpha": m in ("Ridge", "Lasso"),
                "train_rows": int(n_tr), "test_rows": int(e_test - s_test),
                "train_start": str(self.panel.date.iloc[0].date()), "train_end": str(d_tr_max.date()),
                "test_start": str(d_te_min.date()), "test_end": str(self.panel.date.iloc[e_test - 1].date()),
                "n_features": int(len(fnames)), "dropped_all_missing_or_sparse": dropped,
                "alpha": getattr(est, "alpha", None), "intercept": float(est.intercept_),
                "coef_l2": float(np.linalg.norm(coef)), "coef_l1": float(np.abs(coef).sum()),
                "coef_nonzero": int((np.abs(coef) > 0).sum()),
                "convergence_warnings_final_fit": n_conv, "other_warnings": other,
                "n_iter": (int(np.max(est.n_iter_)) if hasattr(est, "n_iter_") and est.n_iter_ is not None else None),
                "pred_std": float(np.std(pred)),
                "seconds": {"shared_preprocess_and_transform": round(t_pre, 2),
                            "shared_search": round(t_search, 2), "final_fit": round(t_fit, 2)},
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            if m in ("RidgeCV", "Ridge"):
                meta["alpha_over_n_train"] = float(est.alpha) / n_tr
            if sr is not None:
                meta["search"] = {**{k: v for k, v in asdict(sr).items() if k != "extra"}, **sr.extra,
                                  "alpha_over_n_train": sr.alpha / n_tr}
            run.p("fits", m, Y, "json").write_text(json.dumps(meta, indent=1, default=float))   # written last = completion marker
            run.log(event="fit_done", model=m, year=Y, alpha=meta["alpha"], seconds=meta["seconds"])
            print(f"[{Y}] {m:8s} alpha={meta['alpha']!s:>22.22} n_train={n_tr:,} nnz={meta['coef_nonzero']} "
                  f"fit={t_fit:.1f}s (pre {t_pre:.0f}s search {t_search:.0f}s) warn={n_conv}", flush=True)
        del Xtr, Xte

    def _dropped_features(self, pre) -> dict:
        """Features removed by training-only missingness filter / all-missing policy."""
        out = {}
        for name, cols in (("numbers", self.cont), ("categoricals", self.disc)):
            pipe = pre.named_transformers_[name]
            mf = pipe.steps[0][1]
            out[name + "_dropped_by_missingness"] = [c for c, k in zip(cols, mf.keep_mask_) if not k]
        return out

    def _inner_folds(self, Y: int, n_tr: int):
        cfg = self.cfg
        years_avail = sorted(y for y in self.bounds if y < Y)
        specs = chronological_year_folds(years_avail, cfg.inner_val_years)
        folds = []
        for tr_years, v in specs:
            assert max(tr_years) < v
            s_v, e_v = self.bounds[v]
            assert (self.year[:s_v] < v).all()
            pre = self._preprocess(0, s_v)                   # preprocessing fit on THIS inner train block only
            Xt = pre.transform(self.X.iloc[:s_v]).astype(np.float64, copy=False)
            Xv = pre.transform(self.X.iloc[s_v:e_v])
            folds.append(make_fold_stats(Xt, self.y[:s_v], Xv, self.y[s_v:e_v], v))
            del Xt, Xv
        return folds

    def run(self, models: list[str], years: list[int] | None = None):
        from threadpoolctl import threadpool_limits
        years = years or self.cfg.years()
        with threadpool_limits(limits=self.cfg.blas_threads):
            for Y in years:
                self.run_year(Y, models)
