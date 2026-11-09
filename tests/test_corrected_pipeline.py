"""Unit tests for the corrected linear-model pipeline (no external data needed)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso, Ridge

from stockpred import data as D
from stockpred.evaluation import max_drawdown, newey_west_lags, perf_stats
from stockpred.portfolio import bin_weights, portfolio_returns, tie_diagnostics
from stockpred.preprocessing import (MissingnessFilter, MonthlyCrossSectionalMeanImputer, Winsorizer,
                                     make_preprocessor)
from stockpred.search import (chronological_year_folds, lasso_val_sse, make_fold_stats, ridge_val_sse,
                              search_ridge)


# ------------------------------------------------------------------ timing ---
def test_factor_lag_moves_forward_one_month_including_december():
    ff = pd.DataFrame({"date": pd.to_datetime(["2009-11-28", "2009-12-28", "2010-01-28"]),
                       "mkt_excess": [1.0, 2.0, 3.0], "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0,
                       "Mom": 0.0, "RF": 0.1})
    lag = D.lag_factors_for_prediction(ff).set_index("date")
    # December 2009 realisation is the predictor for the January 2010 return
    assert lag.loc["2010-01-28", "mkt_excess"] == 2.0
    assert lag.loc["2010-01-28", "ff_source_date"] == pd.Timestamp("2009-12-28")
    assert lag.loc["2009-12-28", "mkt_excess"] == 1.0
    # evaluation factors untouched
    assert ff["date"].iloc[1] == pd.Timestamp("2009-12-28")


def test_signal_dates_dec_to_jan():
    s = D.yyyymm_to_anchor([200912, 201001, 200811])
    nxt = D.next_month(pd.Series(s))
    assert list(nxt.dt.strftime("%Y-%m")) == ["2010-01", "2010-02", "2008-12"]


def test_assert_timing_catches_wrong_direction():
    p = pd.DataFrame({"date": pd.to_datetime(["2010-01-28"]), "signal_date": pd.to_datetime(["2010-02-28"]),
                      "ff_source_date": pd.to_datetime(["2010-02-28"])})
    with pytest.raises(AssertionError):
        D.assert_timing(p)


# ----------------------------------------------------------------- imputer ---
def _frame():
    d = pd.to_datetime(["2000-01-28"] * 3 + ["2000-02-28"] * 3 + ["2001-01-28"] * 2)
    idx = pd.MultiIndex.from_arrays([range(8), d], names=["permno", "date"])
    return pd.DataFrame({"a": [1.0, 3.0, np.nan, np.nan, np.nan, np.nan, 10.0, np.nan],
                         "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                         "allnan": np.nan}, index=idx)


def test_imputer_uses_same_month_only_and_training_fallback():
    X = _frame()
    imp = MonthlyCrossSectionalMeanImputer().fit(X.iloc[:6])      # train = 2000 only
    out = imp.transform(X)
    assert out.iloc[2]["a"] == 2.0                  # Jan-2000 mean(1,3), not a yearly / global mean
    assert out.iloc[7]["a"] == 10.0                 # Jan-2001 uses only Jan-2001's cross-section
    train_mean = np.nanmean([1.0, 3.0])
    assert out.iloc[3]["a"] == train_mean           # Feb-2000 fully missing -> training fallback
    assert "allnan" not in out.columns              # all-missing in training -> dropped (documented)
    assert list(imp.get_feature_names_out()) == ["a", "b"]


def test_imputer_fallback_ignores_later_data():
    X = _frame()
    out_a = MonthlyCrossSectionalMeanImputer().fit(X.iloc[:6]).transform(X.iloc[:6])
    X2 = X.copy()
    X2.iloc[6, 0] = 1e6                              # future observation must not change earlier output
    out_b = MonthlyCrossSectionalMeanImputer().fit(X2.iloc[:6]).transform(X2.iloc[:6])
    pd.testing.assert_frame_equal(out_a, out_b)


def test_preprocessor_fit_on_train_only_and_fresh():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.to_datetime(np.repeat(["2000-01-28", "2000-02-28", "2001-01-28", "2001-02-28"], n // 4))
    X = pd.DataFrame({"c1": rng.normal(size=n), "c2": rng.normal(size=n), "k": rng.integers(0, 3, n).astype(float)},
                     index=pd.MultiIndex.from_arrays([np.arange(n), d], names=["permno", "date"]))
    X.iloc[::7, 0] = np.nan
    pre = make_preprocessor(["c1", "c2"], ["k"])
    pre.fit(X.iloc[:200])
    assert pre is not clone(pre)
    Z = pre.transform(X.iloc[200:])
    assert np.isfinite(Z).all()
    # scaler statistics come from the training block, not the transformed block
    assert list(pre.get_feature_names_out())[:2] == ["numbers__c1", "numbers__c2"]
    a, b = make_preprocessor(["c1"], ["k"]), make_preprocessor(["c1"], ["k"])
    assert a is not b and a.transformers[0][1] is not b.transformers[0][1]


def test_missingness_filter_uses_training_only():
    X = pd.DataFrame({"a": [np.nan] * 9 + [1.0], "b": range(10)}, dtype=float)
    f = MissingnessFilter(0.8).fit(X)
    assert list(f.get_feature_names_out()) == ["b"]


def test_winsorizer_bounds_from_fit():
    tr = np.arange(1000.0).reshape(-1, 1)
    w = Winsorizer(0.01, 0.99).fit(tr)
    out = w.transform(np.array([[-1e9], [500.0], [1e9]]))
    assert out[0, 0] == pytest.approx(9.99) and out[2, 0] == pytest.approx(989.01) and out[1, 0] == 500.0


# ---------------------------------------------------------- chronological CV ---
def test_inner_folds_are_chronological():
    folds = chronological_year_folds(range(1975, 2000), 3)
    assert [v for _, v in folds] == [1997, 1998, 1999]
    for tr, v in folds:
        assert max(tr) < v and tr == sorted(tr)


def _fold(rng, n=2000, p=6, noise=1.0):
    beta = rng.normal(size=p)
    X = rng.normal(size=(n + 500, p)) + 0.3
    y = X @ beta + rng.normal(size=n + 500) * noise + 2.0
    return make_fold_stats(X[:n], y[:n], X[n:], y[n:], 2000), X, y, n


def test_ridge_sufficient_statistics_match_sklearn():
    rng = np.random.default_rng(1)
    f, X, y, n = _fold(rng)
    alphas = np.array([0.1, 10.0, 1000.0])
    sse = ridge_val_sse(f, alphas)
    for a, s in zip(alphas, sse):
        m = Ridge(alpha=a).fit(X[:n], y[:n])
        assert s == pytest.approx(((y[n:] - m.predict(X[n:])) ** 2).sum(), rel=1e-8)


def test_lasso_gram_path_matches_sklearn():
    rng = np.random.default_rng(2)
    f, X, y, n = _fold(rng)
    ratios = np.array([1.0, 0.3, 0.05, 0.005])
    sse, n_conv = lasso_val_sse(f, ratios, tol=1e-10, max_iter=100000)
    for r, s in zip(ratios, sse):
        m = Lasso(alpha=f.alpha_max * r, tol=1e-10, max_iter=100000).fit(X[:n], y[:n])
        assert s == pytest.approx(((y[n:] - m.predict(X[n:])) ** 2).sum(), rel=1e-5)
    assert n_conv == 0


def test_ridge_search_expands_when_endpoint_selected():
    rng = np.random.default_rng(3)
    f, *_ = _fold(rng, noise=0.01)           # almost noiseless -> wants tiny alpha
    res = search_ridge([f], grid=np.logspace(-1, 2, 4), max_expansions=3)
    assert res.expansions >= 1 and min(res.alpha_grid) < 0.1


# ---------------------------------------------------------------- portfolios ---
def _month(preds, rets):
    n = len(preds)
    return pd.DataFrame({"permno": np.arange(n), "date": pd.Timestamp("2010-01-28"),
                         "pred": preds, "ret": rets})


def test_no_ties_equal_quintiles():
    df = _month(np.arange(10.0), np.arange(10.0) / 100)
    W, key = bin_weights(df, "pred")
    assert np.allclose(W.sum(0), 2.0) and np.allclose(W.sum(1), 1.0)
    wide, tall, _ = portfolio_returns(df, W)
    assert wide.loc[df.date[0], 1] == pytest.approx(np.mean([0, 1]) / 100)
    assert wide.loc[df.date[0], 5] == pytest.approx(np.mean([8, 9]) / 100)
    assert wide["LS"].iloc[0] == pytest.approx(wide[5].iloc[0] - wide[1].iloc[0])


def test_ties_symmetric_and_bins_never_empty():
    preds = np.array([0.0] * 7 + [1.0] * 3)               # min-rank would leave bins empty
    df = _month(preds, np.linspace(-0.1, 0.1, 10))
    W, key = bin_weights(df, "pred")
    assert np.allclose(W.sum(0), 2.0)
    assert np.allclose(W[0], W[6])                          # tied stocks identical
    wide, tall, _ = portfolio_returns(df, W)
    assert (tall["nfirms"] > 0).all()
    diag = tie_diagnostics(df, "pred", W, key)
    assert diag["share_in_ties"].iloc[0] == 1.0 and not diag["no_dispersion"].iloc[0]


def test_no_dispersion_month_gives_zero_ls_and_flag():
    df = _month(np.zeros(10), np.random.default_rng(0).normal(size=10) / 10)
    W, key = bin_weights(df, "pred")
    wide, tall, _ = portfolio_returns(df, W)
    assert wide["LS"].iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert tie_diagnostics(df, "pred", W, key)["no_dispersion"].iloc[0]


def test_tie_policy_does_not_use_returns():
    preds = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    a = _month(preds, np.arange(10.0))
    b = _month(preds, np.arange(10.0)[::-1].copy())
    Wa, _ = bin_weights(a, "pred")
    Wb, _ = bin_weights(b, "pred")
    assert np.array_equal(Wa, Wb)


def test_non_finite_predictions_rejected():
    with pytest.raises(ValueError):
        bin_weights(_month(np.array([1.0, np.nan, 2.0, 3.0, 4.0]), np.zeros(5)), "pred")


# ----------------------------------------------------------------- statistics ---
def test_perf_stats_and_drawdown():
    r = pd.Series([0.1, -0.2, 0.1] + [0.0] * 9)
    s = perf_stats(r)
    assert s["ending_wealth"] == pytest.approx(1.1 * 0.8 * 1.1)
    assert s["CAGR"] == pytest.approx(s["ending_wealth"] - 1)
    assert max_drawdown(r) == pytest.approx(-0.2)
    with pytest.raises(AssertionError):
        perf_stats(pd.Series([15.0, -20.0]))                # percent passed by mistake


def test_hgb_targets_and_saved_model_match_the_chosen_round():
    from stockpred.hgb import HGBSpec, _signal, fit_hgb, keep_first_rounds, make_target

    rng = np.random.default_rng(0)
    month = np.repeat(np.arange(24), 200)
    y = rng.normal(size=len(month))
    rank = make_target(y, month, "rank")
    assert rank.min() > -0.5 and rank.max() <= 0.5
    assert np.allclose(pd.Series(rank).groupby(month).mean(), 0.5 / 200)          # each month's ranks, centred
    quintile = make_target(y, month, "quintile")
    assert set(quintile) == {1, 2, 3, 4, 5}
    assert (pd.Series(quintile).groupby(month).value_counts() == 40).all()         # equal-sized quintiles every month

    Z = np.column_stack([y + rng.normal(size=len(y)), rng.normal(size=len(y))])
    tr, va = month < 18, month >= 18
    for target in ("rank", "quintile"):
        spec = HGBSpec("t", target=target, max_iter=30, min_samples_leaf=20, max_leaf_nodes=7)
        est, best_iter, _, _ = fit_hgb(Z[tr], make_target(y[tr], month[tr], target), Z[va], y[va], month[va], spec, seed=0)
        before = (list(est.staged_predict_proba(Z[va]))[best_iter - 1] @ est.classes_ if target == "quintile"
                  else list(est.staged_predict(Z[va]))[best_iter - 1])
        keep_first_rounds(est, best_iter)
        assert len(est._predictors) == best_iter
        assert np.allclose(_signal(est, Z[va]), before)


def test_newey_west_lag_rule():
    assert newey_west_lags(240) == 4


# ------------------------------------------------------------ trading costs ---
def test_leg_trading_charges_half_spread_on_drifted_weights():
    from stockpred.dashboard_bundle import leg_trading
    d = pd.to_datetime(["2001-01-28", "2001-01-28", "2001-02-28", "2001-02-28"])
    pred = pd.DataFrame({"date": d, "permno": [1, 2, 1, 3], "ret": [1.0, 0.0, 0.0, 0.0],
                         "half_spread": [0.01, 0.02, 0.01, 0.03]})
    # January: equal weights in stocks 1 and 2. Stock 1 doubles, so before February's trades it is 2/3 of the leg.
    # February: half in stock 1, half in new stock 3: sell 1/6 of stock 1, all 1/3 of stock 2, buy 1/2 of stock 3.
    turnover, cost = leg_trading(pred, np.array([0.5, 0.5, 0.5, 0.5]))
    assert turnover.iloc[0] == 0 and cost.iloc[0] == 0
    assert turnover.iloc[1] == pytest.approx(0.5)
    # stock 2 left the sample, so it is sold at its last known spread
    assert cost.iloc[1] == pytest.approx(0.01 / 6 + 0.02 / 3 + 0.03 / 2)
