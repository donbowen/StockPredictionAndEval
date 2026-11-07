"""Chronologically valid MLP return predictors (PyTorch, GPU if available).

Protocol for test year Y (identical timing/preprocessing to the linear runs):
  * preprocessor fit on all rows with year < Y (training-only statistics),
  * network trained on years < Y-V, early-stopped on the V most recent years before Y
    (V = cfg.inner_val_years; whole years; the validation block is never used as test),
  * the test year is touched only by ``predict`` after training is finished.
Every fit gets a freshly built network; per-seed state dicts are saved immediately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from torch import nn

from .runner import Backtest, Config, code_hash, config_hash


@dataclass
class MLPSpec:
    name: str
    hidden: tuple = (64, 32)
    dropout: float = 0.0
    weight_decay: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 40
    patience: int = 5
    loss: str = "mse"                 # "mse" | "huber"
    target: str = "clip"              # "raw" | "clip" | "cs_demean" | "rank"
    clip_q: tuple = (0.005, 0.995)    # training-sample quantiles used for "raw"-> clipped targets (percent returns)
    batchnorm: bool = False
    n_seeds: int = 1

    def key(self):
        return json.loads(json.dumps(asdict(self), default=list))


# ---------------------------------------------------------------- torch pieces ---
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_net(n_in: int, spec: MLPSpec) -> nn.Module:
    layers, d = [], n_in
    for h in spec.hidden:
        layers.append(nn.Linear(d, h))
        if spec.batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if spec.dropout > 0:
            layers.append(nn.Dropout(spec.dropout))
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def monthly_corr(p: torch.Tensor, y: torch.Tensor, mc: torch.Tensor, nm: int) -> float:
    """Mean over months of the cross-sectional Pearson correlation of p and y (all float64 on device)."""
    p, y = p.double(), y.double()

    def s(v):
        return torch.zeros(nm, device=p.device, dtype=torch.float64).index_add_(0, mc, v)
    n = s(torch.ones_like(p))
    mp, my = s(p) / n, s(y) / n
    cov = s(p * y) / n - mp * my
    vp, vy = s(p * p) / n - mp ** 2, s(y * y) / n - my ** 2
    c = cov / torch.sqrt((vp * vy).clamp_min(1e-300))
    ok = (n > 2) & (vp > 1e-18) & (vy > 1e-18)
    return float(c[ok].mean()) if ok.any() else float("nan")


def _predict_t(net, X: torch.Tensor, bs=262144) -> torch.Tensor:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(X[i:i + bs]).squeeze(-1))
    return torch.cat(out)


def fit_mlp(Xtr, ytr, Xva, yva, mcv, nmv, spec: MLPSpec, seed: int):
    """Xtr/ytr/Xva/yva are device tensors; ytr is the (scaled) training target, yva the clipped raw return.
    Early stopping maximises the validation mean monthly Pearson correlation. Returns (state, history)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = build_net(Xtr.shape[1], spec).to(Xtr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    lossf = nn.MSELoss() if spec.loss == "mse" else nn.HuberLoss(delta=1.0)
    g = torch.Generator(device=Xtr.device)
    g.manual_seed(seed)
    n = len(Xtr)
    best, best_state, best_ep, hist = -np.inf, None, -1, []
    for ep in range(spec.max_epochs):
        net.train()
        perm = torch.randperm(n, device=Xtr.device, generator=g)
        tot = 0.0
        for i in range(0, n, spec.batch_size):
            idx = perm[i:i + spec.batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        ic = monthly_corr(_predict_t(net, Xva), yva, mcv, nmv)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_ic_pearson": ic})
        if ic > best:
            best, best_ep = ic, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= spec.patience:
            break
    return best_state, {"best_epoch": best_ep, "best_val_ic_pearson": best, "epochs_run": len(hist), "history": hist}


class MLPEnsemble(BaseEstimator, RegressorMixin):
    """Mean of several independently-seeded networks; predictions are in the units of the training target
    (percent for raw/clip/cs_demean targets, centred cross-sectional rank for the rank target)."""

    def __init__(self, n_in=0, spec=None, y_scale=1.0, states=None):
        self.n_in, self.spec, self.y_scale, self.states = n_in, spec, y_scale, states

    def __sklearn_is_fitted__(self):
        return self.states is not None

    def fit(self, X=None, y=None):
        raise NotImplementedError("trained by NNBacktest.train_spec; this object only predicts")

    def predict(self, X):
        dev = _device()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
        acc = None
        for st in self.states:
            net = build_net(self.n_in, self.spec).to(dev)
            net.load_state_dict(st)
            p = _predict_t(net, Xt).double()
            acc = p if acc is None else acc + p
        return (acc / len(self.states) * self.y_scale).cpu().numpy()


# ---------------------------------------------------------------- targets ---
def make_target(ytr: np.ndarray, mc_tr: np.ndarray, spec: MLPSpec):
    lo, hi = np.quantile(ytr, spec.clip_q)
    if spec.target == "raw":
        y = ytr.copy()
    elif spec.target == "clip":
        y = np.clip(ytr, lo, hi)
    elif spec.target == "cs_demean":
        y = np.clip(ytr, lo, hi)
        y = y - pd.Series(y).groupby(mc_tr).transform("mean").to_numpy()
    elif spec.target == "rank":
        y = pd.Series(ytr).groupby(mc_tr).rank(pct=True).to_numpy() - 0.5
    else:
        raise ValueError(spec.target)
    scale = float(np.std(y))
    return (y / scale).astype(np.float32), scale, (float(lo), float(hi))


def spearman_ic(pred: np.ndarray, y: np.ndarray, mc: np.ndarray) -> float:
    d = pd.DataFrame({"p": pred, "y": y, "m": mc})
    d["p"], d["y"] = d.groupby("m")["p"].rank(), d.groupby("m")["y"].rank()
    return float(d.groupby("m").apply(lambda x: x["p"].corr(x["y"]), include_groups=False).mean())


# ---------------------------------------------------------------- backtest ---
class NNBacktest(Backtest):
    def __init__(self, cfg: Config, specs: list[MLPSpec]):
        super().__init__(cfg)
        self.specs = {s.name: s for s in specs}
        payload = json.dumps([s.key() for s in specs], sort_keys=True) + self.chash + "nn_v1"
        self.chash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.nn_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        r = self.rd.root
        cj = json.loads((r / "config.json").read_text())
        cj.update(config_hash=self.chash, model_specs=[s.key() for s in specs])
        (r / "config.json").write_text(json.dumps(cj, indent=1, default=list))
        env = json.loads((r / "environment.json").read_text())
        env["torch"] = torch.__version__
        env["model_fit_device"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU") + " (PyTorch)"
        env["nn_code_hash"] = self.nn_code_hash
        (r / "environment.json").write_text(json.dumps(env, indent=1))

    # ---- shared data prep for one test year
    def prepare(self, Y: int, with_test: bool = True):
        V = self.cfg.inner_val_years
        s_te, e_te = self.bounds[Y]
        s_va = self.bounds[Y - V][0]
        assert (self.year[:s_va] < Y - V).all() and (self.year[s_va:s_te] >= Y - V).all() and (self.year[s_va:s_te] < Y).all()
        pre = self._preprocess(0, s_te)                          # training-only (years < Y) statistics
        dev = _device()

        def tx(a, b):
            return pre.transform(self.X.iloc[a:b]).astype(np.float32, copy=False)
        d = {"Y": Y, "pre": pre, "s_va": s_va, "s_te": s_te, "e_te": e_te, "dev": dev,
             "fnames": [str(f) for f in pre.get_feature_names_out()]}
        d["Xtr"], d["Xva"] = tx(0, s_va), tx(s_va, s_te)
        assert np.isfinite(d["Xtr"]).all() and np.isfinite(d["Xva"]).all()
        dates = self.panel["date"]
        d["ytr"], d["yva"] = self.y[:s_va], self.y[s_va:s_te]
        d["mc_tr"] = pd.factorize(dates.iloc[:s_va])[0]
        mva, _ = pd.factorize(dates.iloc[s_va:s_te])
        d["mc_va"] = mva
        if with_test:
            d["Xte"] = tx(s_te, e_te)
            assert np.isfinite(d["Xte"]).all()
            d["yte"] = self.y[s_te:e_te]
        return d

    def train_spec(self, d, spec: MLPSpec, seeds: list[int]):
        """Train ``len(seeds)`` networks for one spec on prepared data; returns ensemble + diagnostics."""
        dev = d["dev"]
        y_t, scale, (lo, hi) = make_target(d["ytr"], d["mc_tr"], spec)
        Xtr = torch.as_tensor(d["Xtr"], device=dev)
        ytr = torch.as_tensor(y_t, device=dev)
        Xva = torch.as_tensor(d["Xva"], device=dev)
        yva_c = torch.as_tensor(np.clip(d["yva"], lo, hi).astype(np.float32), device=dev)   # common validation target
        mcv = torch.as_tensor(d["mc_va"], device=dev)
        nmv = int(d["mc_va"].max()) + 1
        states, runs = [], []
        for sd in seeds:
            t = time.time()
            st, info = fit_mlp(Xtr, ytr, Xva, yva_c, mcv, nmv, spec, sd)
            info.update(seed=sd, seconds=round(time.time() - t, 1))
            states.append(st)
            runs.append(info)
        ens = MLPEnsemble(n_in=Xtr.shape[1], spec=spec, y_scale=scale, states=states)
        pv = ens.predict(d["Xva"])
        diag = {"val_spearman_ic": spearman_ic(pv, d["yva"], d["mc_va"]),
                "val_pearson_ic_ensemble": monthly_corr(torch.as_tensor(pv, device=dev), yva_c, mcv, nmv),
                "target_scale": scale, "clip_bounds_pct": [lo, hi], "runs": runs,
                "n_params": int(sum(p.numel() for p in build_net(Xtr.shape[1], spec).parameters()))}
        del Xtr, ytr, Xva
        return ens, diag

    # ---- production fit for a test year
    def run_year_nn(self, Y: int, names: list[str], seeds_base: int = 0):
        cfg, run = self.cfg, self.rd
        todo = [m for m in names if not run.is_complete(m, Y, self.chash)]
        if not todo:
            print(f"[{Y}] {names} already complete - skipping", flush=True)
            return
        t0 = time.time()
        d = self.prepare(Y, with_test=True)
        t_pre = time.time() - t0
        s_te, e_te, s_va = d["s_te"], d["e_te"], d["s_va"]
        d_tr_max, d_va_max = self.panel.date.iloc[s_va - 1], self.panel.date.iloc[s_te - 1]
        d_te_min = self.panel.date.iloc[s_te]
        assert d_tr_max < self.panel.date.iloc[s_va] <= d_va_max < d_te_min
        run.p("features", None, Y, "json").write_text(json.dumps(d["fnames"]))
        for m in todo:
            spec = self.specs[m]
            seeds = [cfg.seed + seeds_base + 1000 * Y + i for i in range(spec.n_seeds)]
            t1 = time.time()
            ens, diag = self.train_spec(d, spec, seeds)
            pred = ens.predict(d["Xte"])
            assert np.isfinite(pred).all() and len(pred) == e_te - s_te
            pipe = Pipeline([("preprocess", copy.deepcopy(d["pre"])), ("model", ens)])
            n_probe = int((self.panel.date.iloc[s_te:e_te] == d_te_min).sum())
            assert np.allclose(pipe.predict(self.X.iloc[s_te:s_te + n_probe]), pred[:n_probe], rtol=1e-3, atol=1e-4), \
                "saved pipeline != predictions"
            joblib.dump(pipe, run.p("models", m, Y, "joblib"), compress=3)
            pd.DataFrame({"permno": self.panel.permno.to_numpy()[s_te:e_te],
                          "date": self.panel.date.to_numpy()[s_te:e_te],
                          "signal_date": self.panel.signal_date.to_numpy()[s_te:e_te],
                          "ret": d["yte"], "pred": pred}).to_parquet(run.p("predictions", m, Y, "parquet"))
            meta = {"model": m, "year": Y, "config_hash": self.chash, "panel_key": self.panel_key,
                    "code_hash": code_hash(), "nn_code_hash": self.nn_code_hash, "spec": spec.key(),
                    "seeds": seeds, "train_rows": int(s_va), "val_rows": int(s_te - s_va), "test_rows": int(e_te - s_te),
                    "train_start": str(self.panel.date.iloc[0].date()), "train_end": str(d_tr_max.date()),
                    "val_start": str(self.panel.date.iloc[s_va].date()), "val_end": str(d_va_max.date()),
                    "test_start": str(d_te_min.date()), "n_features": len(d["fnames"]),
                    "pred_std": float(np.std(pred)), **{k: v for k, v in diag.items()},
                    "seconds": {"shared_preprocess_and_transform": round(t_pre, 1), "train_and_predict": round(time.time() - t1, 1)},
                    "finished_utc": datetime.now(timezone.utc).isoformat()}
            run.p("fits", m, Y, "json").write_text(json.dumps(meta, indent=1, default=float))
            run.log(event="fit_done", model=m, year=Y, val_ic=diag["val_spearman_ic"], seconds=meta["seconds"])
            print(f"[{Y}] {m:14s} val_IC(spearman)={diag['val_spearman_ic']:.4f} epochs="
                  f"{[r['best_epoch'] for r in diag['runs']]} train={s_va:,} {time.time() - t1:.0f}s", flush=True)
        del d

    def run_nn(self, names: list[str], years: list[int] | None = None):
        for Y in (years or self.cfg.years()):
            self.run_year_nn(Y, names)
