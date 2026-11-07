# Corrected linear-model backtest: `linear_v1_smoke`  (SMOKE TEST - not results)

*Out-of-sample long-short (quintile 5 minus 1) returns, test years 2018-2019.*

wealth = cumprod(1 + r_LS), start $1. Exposure convention: $1 long the top quintile and $1 short the bottom quintile per $1 of strategy equity, rebalanced monthly, before transaction costs, borrow fees and financing/short-rebate effects.

## Performance by evaluation start (all end 2019-12)

| model   | start   |   months | CAGR   | ann_mean   | ann_vol   |   Sharpe | max_drawdown   |   ending_wealth |
|:--------|:--------|---------:|:-------|:-----------|:----------|---------:|:---------------|----------------:|
| OLS     | 2018-01 |       24 | -7.6%  | -7.4%      | 9.4%      |    -0.79 | -17.7%         |            0.85 |
| Ridge   | 2018-01 |       24 | -7.5%  | -7.4%      | 9.4%      |    -0.79 | -17.5%         |            0.85 |
| RidgeCV | 2018-01 |       24 | -13.5% | -13.1%     | 16.3%     |    -0.8  | -33.3%         |            0.75 |
| Lasso   | 2018-01 |       24 | -0.9%  | -0.5%      | 9.5%      |    -0.05 | -10.2%         |            0.98 |
| LassoCV | 2018-01 |       24 | -7.1%  | -6.8%      | 10.6%     |    -0.64 | -18.8%         |            0.86 |

## Tie / dispersion diagnostics

| model   |   months |   no_dispersion_months |   mean_share_in_ties |   max_tie_share_any_month |   mean_share_fractional |   min_effective_n_per_bin |   mean_stocks_per_month |
|:--------|---------:|-----------------------:|---------------------:|--------------------------:|------------------------:|--------------------------:|------------------------:|
| Lasso   |       24 |                      0 |               0.9898 |                    0.4108 |                  0.4383 |                   94      |                 490.625 |
| LassoCV |       24 |                      0 |               0      |                    0.0021 |                  0.0068 |                   93.6406 |                 490.625 |
| OLS     |       24 |                      0 |               0      |                    0.0021 |                  0.0068 |                   93.6406 |                 490.625 |
| Ridge   |       24 |                      0 |               0      |                    0.0021 |                  0.0068 |                   93.6406 |                 490.625 |
| RidgeCV |       24 |                      0 |               0      |                    0.0021 |                  0.0068 |                   93.6406 |                 490.625 |

## Selected alpha by year

| model   |   year |       alpha |   alpha_over_n_train |   alpha_ratio_to_alpha_max |   train_rows |   coef_nonzero | hit_lower_endpoint   | hit_upper_endpoint   |   grid_expansions |   convergence_warnings_final_fit |
|:--------|-------:|------------:|---------------------:|---------------------------:|-------------:|---------------:|:---------------------|:---------------------|------------------:|---------------------------------:|
| LassoCV |   2018 | 0.409962    |              nan     |                   0.385662 |       199299 |              6 | False                | False                |                 0 |                                0 |
| LassoCV |   2019 | 0.400258    |              nan     |                   0.385662 |       205289 |              5 | False                | False                |                 0 |                                0 |
| RidgeCV |   2018 | 3.16228e+06 |               15.867 |                 nan        |       199299 |            258 | False                | False                |                 0 |                                0 |
| RidgeCV |   2019 | 3.16228e+06 |               15.404 |                 nan        |       205289 |            258 | False                | False                |                 0 |                                0 |

## Shrinkage and ranking similarity vs OLS

| model   |   year |   coef_l2 |   coef_nonzero |   n_features |   coef_l2_ratio_to_OLS |   mean_monthly_spearman_vs_OLS |
|:--------|-------:|----------:|---------------:|-------------:|-----------------------:|-------------------------------:|
| Lasso   |   2018 |    0.063  |              1 |          258 |                 0.0145 |                         0.2465 |
| Lasso   |   2019 |    0.0378 |              1 |          258 |                 0.0093 |                         0.2317 |
| LassoCV |   2018 |    0.7564 |              6 |          258 |                 0.1739 |                         0.4334 |
| LassoCV |   2019 |    0.7358 |              5 |          258 |                 0.1801 |                         0.494  |
| OLS     |   2018 |    4.3487 |            258 |          258 |                 1      |                       nan      |
| OLS     |   2019 |    4.0858 |            258 |          258 |                 1      |                       nan      |
| Ridge   |   2018 |    4.3467 |            258 |          258 |                 0.9995 |                         1      |
| Ridge   |   2019 |    4.0843 |            258 |          258 |                 0.9996 |                         1      |
| RidgeCV |   2018 |    0.1731 |            258 |          258 |                 0.0398 |                         0.5794 |
| RidgeCV |   2019 |    0.1748 |            258 |          258 |                 0.0428 |                         0.6455 |

## Corrected vs original saved results

| model                 | start   |   months | CAGR   | ann_mean   | ann_vol   |   Sharpe | max_drawdown   |   ending_wealth | pipeline                                      | corr_LS_monthly_new_vs_old   |
|:----------------------|:--------|---------:|:-------|:-----------|:----------|---------:|:---------------|----------------:|:----------------------------------------------|:-----------------------------|
| OLS                   | 2018-01 |       24 | -7.6%  | -7.4%      | 9.4%      |    -0.79 | -17.7%         |            0.85 | corrected                                     | 0.48                         |
| Ridge                 | 2018-01 |       24 | -7.5%  | -7.4%      | 9.4%      |    -0.79 | -17.5%         |            0.85 | corrected                                     | 0.49                         |
| RidgeCV               | 2018-01 |       24 | -13.5% | -13.1%     | 16.3%     |    -0.8  | -33.3%         |            0.75 | corrected                                     | 0.34                         |
| Lasso                 | 2018-01 |       24 | -0.9%  | -0.5%      | 9.5%      |    -0.05 | -10.2%         |            0.98 | corrected                                     | 0.36                         |
| LassoCV               | 2018-01 |       24 | -7.1%  | -6.8%      | 10.6%     |    -0.64 | -18.8%         |            0.86 | corrected                                     | 0.20                         |
| mlp-256-128-64-32     | 2018-01 |       24 | 8.9%   | 9.0%       | 9.9%      |     0.92 | -7.5%          |            1.19 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-512-256-128-64-32 | 2018-01 |       24 | 14.0%  | 13.9%      | 12.1%     |     1.14 | -11.1%         |            1.3  | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-64-32             | 2018-01 |       24 | 8.1%   | 8.9%       | 15.1%     |     0.59 | -11.7%         |            1.17 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-32-16-8           | 2018-01 |       24 | 1.9%   | 2.5%       | 10.7%     |     0.23 | -11.0%         |            1.04 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-128-64-32         | 2018-01 |       24 | 10.4%  | 10.3%      | 9.0%      |     1.14 | -5.7%          |            1.22 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| OLS                   | 2018-01 |       24 | 7.6%   | 7.5%       | 5.9%      |     1.27 | -4.1%          |            1.16 | ORIGINAL (old timing/imputer/CV) - historical | 0.48                         |
| Lasso                 | 2018-01 |       23 | -4.5%  | -4.4%      | 5.6%      |    -0.78 | -9.8%          |            0.92 | ORIGINAL (old timing/imputer/CV) - historical | 0.36                         |
| Ridge                 | 2018-01 |       24 | 7.6%   | 7.5%       | 5.9%      |     1.27 | -4.1%          |            1.16 | ORIGINAL (old timing/imputer/CV) - historical | 0.49                         |
| LassoCV               | 2018-01 |       24 | 7.3%   | 7.2%       | 5.9%      |     1.21 | -4.0%          |            1.15 | ORIGINAL (old timing/imputer/CV) - historical | 0.20                         |
| RidgeCV               | 2018-01 |       24 | 7.6%   | 7.5%       | 5.9%      |     1.27 | -4.1%          |            1.16 | ORIGINAL (old timing/imputer/CV) - historical | 0.34                         |
| hgbr                  | 2018-01 |       24 | 7.9%   | 8.2%       | 11.1%     |     0.74 | -13.1%         |            1.16 | ORIGINAL (old timing/imputer/CV) - historical |                              |

## Checks

```json
{
 "models": [
  "OLS",
  "Ridge",
  "RidgeCV",
  "Lasso",
  "LassoCV"
 ],
 "unique_stock_month_keys": true,
 "finite_predictions": true,
 "signal_precedes_return_by_one_month": true,
 "train_ends_before_test_starts": true,
 "inner_validation_years_before_test_year": true,
 "prediction_years": [
  2018,
  2019
 ],
 "return_units": "panel ret in percent; converted to decimals once here (min=-0.819 max=3.541)",
 "tall_equals_wide": true,
 "long_short_equals_bin5_minus_bin1": true,
 "portfolio_weights_sum_to_one_per_bin_month": true,
 "no_empty_portfolios": true,
 "wealth_convention": "wealth = cumprod(1 + r_LS), start $1. Exposure convention: $1 long the top quintile and $1 short the bottom quintile per $1 of strategy equity, rebalanced monthly, before transaction costs, borrow fees and financing/short-rebate effects."
}
```