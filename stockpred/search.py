"""Chronological hyper-parameter search for Ridge / Lasso.

Design
------
* Inner folds are whole calendar years of *return month*; each validation year
  is preceded by a training block made of strictly earlier years (expanding
  window).  Never row-wise / random splits.
* Preprocessing is fit freshly on every inner training block.
* Per fold we keep only sufficient statistics of the preprocessed training
  block (centred Gram matrix X'X, X'y, means, n) plus the preprocessed
  validation block.  All alpha candidates reuse them, so a whole alpha grid
  costs one eigendecomposition (Ridge) or one warm-started Gram coordinate
  descent path (Lasso).  Selection criterion: pooled validation MSE.
* Ridge penalises exactly as ``sklearn.linear_model.Ridge`` (intercept
  unpenalised, objective ||y-Xb||^2 + alpha||b||^2).  Lasso uses sklearn's
  objective (1/2n)||y-Xb||^2 + alpha||b||_1; its alpha grid is expressed as a
  ratio of alpha_max (the smallest alpha giving the all-zero model) computed
  from training data only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path


# ------------------------------------------------------------------ folds ---
def chronological_year_folds(years, n_val_years: int, min_train_years: int = 5):
    """Expanding-window folds over whole years.

    Returns [(train_years, val_year), ...]; every training year < val_year.
    """
    ys = sorted(set(int(y) for y in years))
    folds = []
    for v in ys[-n_val_years:]:
        tr = [y for y in ys if y < v]
        if len(tr) >= min_train_years:
            folds.append((tr, v))
    for tr, v in folds:
        assert max(tr) < v
    return folds


# ----------------------------------------------------------- fold statistics ---
@dataclass
class FoldStats:
    val_year: int
    n_train: int
    n_val: int
    x_mean: np.ndarray
    y_mean: float
    gram: np.ndarray          # centred X'X  (p x p, float64)
    xty: np.ndarray           # centred X'y
    yss: float                # centred y'y
    X_val: np.ndarray         # preprocessed validation features
    y_val: np.ndarray
    n_features: int
    _eig: tuple | None = field(default=None, repr=False)

    @property
    def alpha_max(self) -> float:
        return float(np.max(np.abs(self.xty)) / self.n_train)

    def eig(self):
        if self._eig is None:
            w, V = np.linalg.eigh(self.gram)
            self._eig = (np.clip(w, 0.0, None), V, V.T @ self.xty)
        return self._eig


def sufficient_stats(X: np.ndarray, y: np.ndarray, chunk: int = 200_000):
    """Float64-accumulated centred X'X, X'y, means (X may be float32/64)."""
    n, p = X.shape
    sx = np.zeros(p)
    sxx = np.zeros((p, p))
    sxy = np.zeros(p)
    sy = 0.0
    syy = 0.0
    for i in range(0, n, chunk):
        xb = np.asarray(X[i:i + chunk], dtype=np.float64)
        yb = np.asarray(y[i:i + chunk], dtype=np.float64)
        sx += xb.sum(0)
        sxx += xb.T @ xb
        sxy += xb.T @ yb
        sy += yb.sum()
        syy += float(yb @ yb)
    xm, ym = sx / n, sy / n
    gram = sxx - n * np.outer(xm, xm)
    xty = sxy - n * xm * ym
    return xm, ym, gram, xty, syy - n * ym * ym


def make_fold_stats(X_tr, y_tr, X_val, y_val, val_year) -> FoldStats:
    xm, ym, gram, xty, yss = sufficient_stats(X_tr, y_tr)
    return FoldStats(val_year=int(val_year), n_train=len(X_tr), n_val=len(X_val), x_mean=xm,
                     y_mean=float(ym), gram=gram, xty=xty, yss=float(yss),
                     X_val=np.asarray(X_val, dtype=np.float64), y_val=np.asarray(y_val, dtype=np.float64),
                     n_features=X_tr.shape[1])


# ------------------------------------------------------------------ ridge ---
def ridge_val_sse(fold: FoldStats, alphas: np.ndarray) -> np.ndarray:
    w, V, vty = fold.eig()
    B = V @ (vty[:, None] / (w[:, None] + alphas[None, :]))      # p x A
    b0 = fold.y_mean - fold.x_mean @ B                           # A
    pred = fold.X_val @ B + b0[None, :]
    return ((fold.y_val[:, None] - pred) ** 2).sum(0)


@dataclass
class SearchResult:
    name: str
    alpha: float                  # selected alpha on the outer-training scale
    alpha_grid: list
    val_mse: list                 # pooled validation MSE per grid point
    best_index: int
    n_inner_train: list           # per fold
    n_val: list
    val_years: list
    hit_lower_endpoint: bool = False
    hit_upper_endpoint: bool = False
    expansions: int = 0
    extra: dict = field(default_factory=dict)


def search_ridge(folds: list[FoldStats], grid=None, max_expansions=3, step=10 ** 0.5) -> SearchResult:
    grid = np.logspace(-4, 8, 25) if grid is None else np.asarray(grid, float)
    exp = 0
    while True:
        sse = sum(ridge_val_sse(f, grid) for f in folds)
        mse = sse / sum(f.n_val for f in folds)
        b = int(np.argmin(mse))
        if exp >= max_expansions or 0 < b < len(grid) - 1:
            break
        if b == 0:
            grid = np.concatenate([grid[0] / step ** np.arange(4, 0, -1), grid])
        else:
            grid = np.concatenate([grid, grid[-1] * step ** np.arange(1, 5)])
        exp += 1
    return SearchResult("RidgeCV", float(grid[b]), grid.tolist(), mse.tolist(), b,
                        [f.n_train for f in folds], [f.n_val for f in folds], [f.val_year for f in folds],
                        hit_lower_endpoint=(b == 0), hit_upper_endpoint=(b == len(grid) - 1), expansions=exp)


# ------------------------------------------------------------------ lasso ---
def lasso_path_gram(fold: FoldStats, alphas: np.ndarray, tol=1e-4, max_iter=10_000):
    """Lasso coefficient path on sufficient statistics (no n x p design needed).

    sklearn's Gram coordinate descent only reads X.shape/dtype once the Gram
    matrix and X'y are supplied, so a zero-stride stand-in is passed for X.
    """
    stub = np.broadcast_to(np.float64(0.0), (fold.n_train, fold.n_features))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # y enters the Gram solver only through y'y (tolerance / duality-gap scale)
        a, coefs, gaps = lasso_path(
            stub, np.array([np.sqrt(fold.yss)]), alphas=alphas, precompute=np.ascontiguousarray(fold.gram),
            Xy=fold.xty, tol=tol, max_iter=max_iter, check_input=False,
        )
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return coefs, n_conv


def lasso_val_sse(fold: FoldStats, ratios: np.ndarray, tol=1e-4, max_iter=10_000):
    alphas = fold.alpha_max * ratios
    coefs, n_conv = lasso_path_gram(fold, alphas, tol, max_iter)    # p x A in descending-alpha order
    order = np.argsort(-alphas)
    B = np.empty_like(coefs)
    B[:, order] = coefs
    b0 = fold.y_mean - fold.x_mean @ B
    pred = fold.X_val @ B + b0[None, :]
    return ((fold.y_val[:, None] - pred) ** 2).sum(0), n_conv


def search_lasso(folds: list[FoldStats], alpha_max_outer: float, n_alphas=30, lo=1e-4,
                 max_expansions=2, tol=1e-4, max_iter=10_000) -> SearchResult:
    ratios = np.logspace(0, np.log10(lo), n_alphas)
    exp, n_conv = 0, 0
    while True:
        sse = np.zeros(len(ratios))
        n_conv = 0
        for f in folds:
            s, c = lasso_val_sse(f, ratios, tol, max_iter)
            sse += s
            n_conv += c
        mse = sse / sum(f.n_val for f in folds)
        b = int(np.argmin(mse))
        if exp >= max_expansions or b < len(ratios) - 1:
            break
        extra = ratios[-1] * 10 ** (-np.arange(1, 7) / 3.0)
        ratios = np.concatenate([ratios, extra])
        exp += 1
    return SearchResult("LassoCV", float(ratios[b] * alpha_max_outer), (ratios * alpha_max_outer).tolist(),
                        mse.tolist(), b, [f.n_train for f in folds], [f.n_val for f in folds],
                        [f.val_year for f in folds],
                        hit_lower_endpoint=(b == len(ratios) - 1), hit_upper_endpoint=(b == 0), expansions=exp,
                        extra={"alpha_ratio": float(ratios[b]), "ratio_grid": ratios.tolist(),
                               "inner_convergence_warnings": int(n_conv)})
