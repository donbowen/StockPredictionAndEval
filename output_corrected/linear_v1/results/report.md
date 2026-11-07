# Corrected linear-model backtest: `linear_v1`

*Out-of-sample long-short (quintile 5 minus 1) returns, test years 2000-2019.*

wealth = cumprod(1 + r_LS), start $1. Exposure convention: $1 long the top quintile and $1 short the bottom quintile per $1 of strategy equity, rebalanced monthly, before transaction costs, borrow fees and financing/short-rebate effects.

## Performance by evaluation start (all end 2019-12)

| model   | start   |   months | CAGR   | ann_mean   | ann_vol   |   Sharpe | max_drawdown   |   ending_wealth |
|:--------|:--------|---------:|:-------|:-----------|:----------|---------:|:---------------|----------------:|
| OLS     | 2000-01 |      240 | 31.3%  | 28.3%      | 13.0%     |     2.17 | -6.9%          |          230.3  |
| Ridge   | 2000-01 |      240 | 31.3%  | 28.3%      | 13.0%     |     2.17 | -6.9%          |          230.32 |
| RidgeCV | 2000-01 |      240 | 27.2%  | 25.2%      | 14.0%     |     1.81 | -9.2%          |          123.42 |
| Lasso   | 2000-01 |      240 | 11.4%  | 12.3%      | 17.3%     |     0.71 | -38.9%         |            8.66 |
| LassoCV | 2000-01 |      240 | 18.8%  | 18.5%      | 15.2%     |     1.22 | -19.5%         |           31.62 |
| OLS     | 2002-01 |      216 | 26.8%  | 24.5%      | 10.8%     |     2.28 | -6.9%          |           71.55 |
| Ridge   | 2002-01 |      216 | 26.8%  | 24.5%      | 10.8%     |     2.28 | -6.9%          |           71.56 |
| RidgeCV | 2002-01 |      216 | 23.5%  | 21.9%      | 11.8%     |     1.86 | -9.2%          |           44.65 |
| Lasso   | 2002-01 |      216 | 10.1%  | 10.3%      | 11.5%     |     0.89 | -15.3%         |            5.63 |
| LassoCV | 2002-01 |      216 | 14.5%  | 14.2%      | 11.9%     |     1.2  | -18.6%         |           11.36 |
| OLS     | 2005-01 |      180 | 23.3%  | 21.5%      | 8.5%      |     2.52 | -3.7%          |           23.22 |
| Ridge   | 2005-01 |      180 | 23.3%  | 21.5%      | 8.5%      |     2.52 | -3.7%          |           23.22 |
| RidgeCV | 2005-01 |      180 | 18.0%  | 17.2%      | 9.9%      |     1.74 | -9.2%          |           12.04 |
| Lasso   | 2005-01 |      180 | 7.8%   | 8.0%       | 9.3%      |     0.86 | -15.3%         |            3.1  |
| LassoCV | 2005-01 |      180 | 9.0%   | 9.1%       | 9.6%      |     0.95 | -18.6%         |            3.64 |
| OLS     | 2010-01 |      120 | 20.6%  | 19.1%      | 6.5%      |     2.96 | -3.7%          |            6.53 |
| Ridge   | 2010-01 |      120 | 20.6%  | 19.1%      | 6.5%      |     2.96 | -3.7%          |            6.53 |
| RidgeCV | 2010-01 |      120 | 15.9%  | 15.2%      | 9.0%      |     1.68 | -8.0%          |            4.36 |
| Lasso   | 2010-01 |      120 | 5.4%   | 5.6%       | 7.4%      |     0.75 | -10.4%         |            1.7  |
| LassoCV | 2010-01 |      120 | 3.8%   | 3.9%       | 6.2%      |     0.63 | -11.8%         |            1.45 |

## Tie / dispersion diagnostics

| model   |   months |   no_dispersion_months |   mean_share_in_ties |   max_tie_share_any_month |   mean_share_fractional |   min_effective_n_per_bin |   mean_stocks_per_month |
|:--------|---------:|-----------------------:|---------------------:|--------------------------:|------------------------:|--------------------------:|------------------------:|
| Lasso   |      240 |                      0 |               1      |                    0.4809 |                  0.4326 |                   802.999 |                 4780.33 |
| LassoCV |      240 |                     48 |               0.3612 |                    1      |                  0.2706 |                   802.999 |                 4780.33 |
| OLS     |      240 |                      0 |               0      |                    0.0002 |                  0.0007 |                   802.96  |                 4780.33 |
| Ridge   |      240 |                      0 |               0      |                    0.0002 |                  0.0007 |                   802.96  |                 4780.33 |
| RidgeCV |      240 |                      0 |               0      |                    0.0002 |                  0.0007 |                   802.96  |                 4780.33 |

## Selected alpha by year

| model   |   year |       alpha |   alpha_over_n_train |   alpha_ratio_to_alpha_max |   train_rows |   coef_nonzero | hit_lower_endpoint   | hit_upper_endpoint   |   grid_expansions |   convergence_warnings_final_fit |
|:--------|-------:|------------:|---------------------:|---------------------------:|-------------:|---------------:|:---------------------|:---------------------|------------------:|---------------------------------:|
| LassoCV |   2000 | 0.187537    |           nan        |                   0.108264 |      1726425 |             40 | False                | False                |                 0 |                                0 |
| LassoCV |   2001 | 0.255297    |           nan        |                   0.148735 |      1808937 |             30 | False                | False                |                 0 |                                0 |
| LassoCV |   2002 | 0.257401    |           nan        |                   0.148735 |      1885046 |             33 | False                | False                |                 0 |                                0 |
| LassoCV |   2003 | 0.187797    |           nan        |                   0.108264 |      1954450 |             46 | False                | False                |                 0 |                                0 |
| LassoCV |   2004 | 0.262677    |           nan        |                   0.148735 |      2018649 |             32 | False                | False                |                 0 |                                0 |
| LassoCV |   2005 | 0.491014    |           nan        |                   0.280722 |      2080710 |              9 | False                | False                |                 0 |                                0 |
| LassoCV |   2006 | 0.481353    |           nan        |                   0.280722 |      2142248 |             10 | False                | False                |                 0 |                                0 |
| LassoCV |   2007 | 0.474419    |           nan        |                   0.280722 |      2203029 |             10 | False                | False                |                 0 |                                0 |
| LassoCV |   2008 | 0.887553    |           nan        |                   0.529832 |      2262822 |              1 | False                | False                |                 0 |                                0 |
| LassoCV |   2009 | 0.269185    |           nan        |                   0.148735 |      2321180 |             34 | False                | False                |                 0 |                                0 |
| LassoCV |   2010 | 0.27654     |           nan        |                   0.148735 |      2375822 |             33 | False                | False                |                 0 |                                0 |
| LassoCV |   2011 | 0.270296    |           nan        |                   0.148735 |      2428309 |             31 | False                | False                |                 0 |                                0 |
| LassoCV |   2012 | 0.265407    |           nan        |                   0.148735 |      2479320 |             31 | False                | False                |                 0 |                                0 |
| LassoCV |   2013 | 1.76091     |           nan        |                   1        |      2528903 |              1 | False                | True                 |                 0 |                                0 |
| LassoCV |   2014 | 1.72877     |           nan        |                   1        |      2577414 |              1 | False                | True                 |                 0 |                                0 |
| LassoCV |   2015 | 1.69759     |           nan        |                   1        |      2627009 |              0 | False                | True                 |                 0 |                                0 |
| LassoCV |   2016 | 1.68012     |           nan        |                   1        |      2677595 |              1 | False                | True                 |                 0 |                                0 |
| LassoCV |   2017 | 1.21411     |           nan        |                   0.727895 |      2727161 |              1 | False                | False                |                 0 |                                0 |
| LassoCV |   2018 | 0.635484    |           nan        |                   0.385662 |      2775771 |              3 | False                | False                |                 0 |                                0 |
| LassoCV |   2019 | 0.865288    |           nan        |                   0.529832 |      2824545 |              1 | False                | False                |                 0 |                                0 |
| RidgeCV |   2000 | 1e+06       |             0.579232 |                 nan        |      1726425 |            227 | False                | False                |                 0 |                                0 |
| RidgeCV |   2001 | 3.16228e+06 |             1.74814  |                 nan        |      1808937 |            238 | False                | False                |                 0 |                                0 |
| RidgeCV |   2002 | 3.16228e+06 |             1.67756  |                 nan        |      1885046 |            238 | False                | False                |                 0 |                                0 |
| RidgeCV |   2003 | 3.16228e+06 |             1.61799  |                 nan        |      1954450 |            238 | False                | False                |                 0 |                                0 |
| RidgeCV |   2004 | 3.16228e+06 |             1.56653  |                 nan        |      2018649 |            238 | False                | False                |                 0 |                                0 |
| RidgeCV |   2005 | 1e+07       |             4.80605  |                 nan        |      2080710 |            250 | False                | False                |                 0 |                                0 |
| RidgeCV |   2006 | 1e+07       |             4.66799  |                 nan        |      2142248 |            251 | False                | False                |                 0 |                                0 |
| RidgeCV |   2007 | 1e+07       |             4.5392   |                 nan        |      2203029 |            255 | False                | False                |                 0 |                                0 |
| RidgeCV |   2008 | 1e+07       |             4.41926  |                 nan        |      2262822 |            256 | False                | False                |                 0 |                                0 |
| RidgeCV |   2009 | 3.16228e+06 |             1.36236  |                 nan        |      2321180 |            256 | False                | False                |                 0 |                                0 |
| RidgeCV |   2010 | 3.16228e+06 |             1.33102  |                 nan        |      2375822 |            256 | False                | False                |                 0 |                                0 |
| RidgeCV |   2011 | 3.16228e+06 |             1.30226  |                 nan        |      2428309 |            256 | False                | False                |                 0 |                                0 |
| RidgeCV |   2012 | 3.16228e+06 |             1.27546  |                 nan        |      2479320 |            256 | False                | False                |                 0 |                                0 |
| RidgeCV |   2013 | 3.16228e+07 |            12.5045   |                 nan        |      2528903 |            256 | False                | False                |                 0 |                                0 |
| RidgeCV |   2014 | 3.16228e+07 |            12.2692   |                 nan        |      2577414 |            257 | False                | False                |                 0 |                                0 |
| RidgeCV |   2015 | 1e+08       |            38.0661   |                 nan        |      2627009 |            258 | False                | False                |                 1 |                                0 |
| RidgeCV |   2016 | 1e+08       |            37.3469   |                 nan        |      2677595 |            258 | False                | False                |                 1 |                                0 |
| RidgeCV |   2017 | 1e+07       |             3.66682  |                 nan        |      2727161 |            263 | False                | False                |                 0 |                                0 |
| RidgeCV |   2018 | 3.16228e+07 |            11.3924   |                 nan        |      2775771 |            263 | False                | False                |                 0 |                                0 |
| RidgeCV |   2019 | 3.16228e+07 |            11.1957   |                 nan        |      2824545 |            263 | False                | False                |                 0 |                                0 |

## Shrinkage and ranking similarity vs OLS

| model   |   year |   coef_l2 |   coef_nonzero |   n_features |   coef_l2_ratio_to_OLS |   mean_monthly_spearman_vs_OLS |
|:--------|-------:|----------:|---------------:|-------------:|-----------------------:|-------------------------------:|
| Lasso   |   2000 |    0.7322 |              1 |          227 |                 0.1264 |                         0.6311 |
| Lasso   |   2001 |    0.7165 |              1 |          238 |                 0.1383 |                         0.5238 |
| Lasso   |   2002 |    0.7306 |              1 |          238 |                 0.1255 |                         0.3387 |
| Lasso   |   2003 |    0.7346 |              1 |          238 |                 0.1321 |                         0.2541 |
| Lasso   |   2004 |    0.7661 |              1 |          238 |                 0.1402 |                         0.2441 |
| Lasso   |   2005 |    0.7491 |              1 |          250 |                 0.1444 |                         0.1926 |
| Lasso   |   2006 |    0.7147 |              1 |          251 |                 0.1395 |                         0.2344 |
| Lasso   |   2007 |    0.69   |              1 |          255 |                 0.137  |                         0.2728 |
| Lasso   |   2008 |    0.6752 |              1 |          256 |                 0.1364 |                         0.3693 |
| Lasso   |   2009 |    0.8098 |              1 |          256 |                 0.1645 |                         0.2492 |
| Lasso   |   2010 |    0.8593 |              1 |          256 |                 0.1763 |                         0.2074 |
| Lasso   |   2011 |    0.8173 |              1 |          256 |                 0.1703 |                         0.2285 |
| Lasso   |   2012 |    0.7844 |              1 |          256 |                 0.1673 |                         0.2386 |
| Lasso   |   2013 |    0.7609 |              1 |          256 |                 0.1648 |                         0.2118 |
| Lasso   |   2014 |    0.7288 |              1 |          257 |                 0.1603 |                         0.2333 |
| Lasso   |   2015 |    0.6976 |              1 |          258 |                 0.1549 |                         0.2643 |
| Lasso   |   2016 |    0.6801 |              1 |          258 |                 0.1514 |                         0.2478 |
| Lasso   |   2017 |    0.668  |              1 |          263 |                 0.1495 |                         0.2175 |
| Lasso   |   2018 |    0.6478 |              1 |          263 |                 0.1459 |                         0.1667 |
| Lasso   |   2019 |    0.6331 |              1 |          263 |                 0.1439 |                         0.198  |
| LassoCV |   2000 |    1.6812 |             40 |          227 |                 0.2902 |                         0.8222 |
| LassoCV |   2001 |    1.52   |             30 |          238 |                 0.2934 |                         0.8254 |
| LassoCV |   2002 |    1.6042 |             33 |          238 |                 0.2755 |                         0.7629 |
| LassoCV |   2003 |    1.7649 |             46 |          238 |                 0.3174 |                         0.7748 |
| LassoCV |   2004 |    1.6382 |             32 |          238 |                 0.2998 |                         0.7749 |
| LassoCV |   2005 |    1.341  |              9 |          250 |                 0.2585 |                         0.3698 |
| LassoCV |   2006 |    1.3019 |             10 |          251 |                 0.2541 |                         0.4005 |
| LassoCV |   2007 |    1.2822 |             10 |          255 |                 0.2545 |                         0.4552 |
| LassoCV |   2008 |    0.7876 |              1 |          256 |                 0.1591 |                         0.3693 |
| LassoCV |   2009 |    1.5794 |             34 |          256 |                 0.3208 |                         0.7395 |
| LassoCV |   2010 |    1.6099 |             33 |          256 |                 0.3303 |                         0.6114 |
| LassoCV |   2011 |    1.6213 |             31 |          256 |                 0.3378 |                         0.6542 |
| LassoCV |   2012 |    1.5977 |             31 |          256 |                 0.3407 |                         0.7177 |
| LassoCV |   2013 |    0      |              1 |          256 |                 0      |                         0.2155 |
| LassoCV |   2014 |    0      |              1 |          257 |                 0      |                         0.233  |
| LassoCV |   2015 |    0      |              0 |          258 |                 0      |                       nan      |
| LassoCV |   2016 |    0      |              1 |          258 |                 0      |                         0.2484 |
| LassoCV |   2017 |    0.4539 |              1 |          263 |                 0.1016 |                         0.2175 |
| LassoCV |   2018 |    1.0241 |              3 |          263 |                 0.2306 |                         0.2807 |
| LassoCV |   2019 |    0.7678 |              1 |          263 |                 0.1745 |                         0.198  |
| OLS     |   2000 |    5.7937 |            227 |          227 |                 1      |                       nan      |
| OLS     |   2001 |    5.1813 |            238 |          238 |                 1      |                       nan      |
| OLS     |   2002 |    5.8228 |            238 |          238 |                 1      |                       nan      |
| OLS     |   2003 |    5.561  |            238 |          238 |                 1      |                       nan      |
| OLS     |   2004 |    5.4636 |            238 |          238 |                 1      |                       nan      |
| OLS     |   2005 |    5.1875 |            250 |          250 |                 1      |                       nan      |
| OLS     |   2006 |    5.1233 |            251 |          251 |                 1      |                       nan      |
| OLS     |   2007 |    5.0376 |            255 |          255 |                 1      |                       nan      |
| OLS     |   2008 |    4.9512 |            256 |          256 |                 1      |                       nan      |
| OLS     |   2009 |    4.9232 |            256 |          256 |                 1      |                       nan      |
| OLS     |   2010 |    4.8742 |            256 |          256 |                 1      |                       nan      |
| OLS     |   2011 |    4.8003 |            256 |          256 |                 1      |                       nan      |
| OLS     |   2012 |    4.6898 |            256 |          256 |                 1      |                       nan      |
| OLS     |   2013 |    4.6162 |            256 |          256 |                 1      |                       nan      |
| OLS     |   2014 |    4.5471 |            257 |          257 |                 1      |                       nan      |
| OLS     |   2015 |    4.5044 |            258 |          258 |                 1      |                       nan      |
| OLS     |   2016 |    4.4935 |            258 |          258 |                 1      |                       nan      |
| OLS     |   2017 |    4.4667 |            263 |          263 |                 1      |                       nan      |
| OLS     |   2018 |    4.4413 |            263 |          263 |                 1      |                       nan      |
| OLS     |   2019 |    4.3998 |            263 |          263 |                 1      |                       nan      |
| Ridge   |   2000 |    5.7935 |            227 |          227 |                 1      |                         1      |
| Ridge   |   2001 |    5.1812 |            238 |          238 |                 1      |                         1      |
| Ridge   |   2002 |    5.8227 |            238 |          238 |                 1      |                         1      |
| Ridge   |   2003 |    5.5609 |            238 |          238 |                 1      |                         1      |
| Ridge   |   2004 |    5.4635 |            238 |          238 |                 1      |                         1      |
| Ridge   |   2005 |    5.1874 |            250 |          250 |                 1      |                         1      |
| Ridge   |   2006 |    5.1233 |            251 |          251 |                 1      |                         1      |
| Ridge   |   2007 |    5.0375 |            255 |          255 |                 1      |                         1      |
| Ridge   |   2008 |    4.9512 |            256 |          256 |                 1      |                         1      |
| Ridge   |   2009 |    4.9232 |            256 |          256 |                 1      |                         1      |
| Ridge   |   2010 |    4.8741 |            256 |          256 |                 1      |                         1      |
| Ridge   |   2011 |    4.8002 |            256 |          256 |                 1      |                         1      |
| Ridge   |   2012 |    4.6898 |            256 |          256 |                 1      |                         1      |
| Ridge   |   2013 |    4.6161 |            256 |          256 |                 1      |                         1      |
| Ridge   |   2014 |    4.5471 |            257 |          257 |                 1      |                         1      |
| Ridge   |   2015 |    4.5044 |            258 |          258 |                 1      |                         1      |
| Ridge   |   2016 |    4.4934 |            258 |          258 |                 1      |                         1      |
| Ridge   |   2017 |    4.4666 |            263 |          263 |                 1      |                         1      |
| Ridge   |   2018 |    4.4412 |            263 |          263 |                 1      |                         1      |
| Ridge   |   2019 |    4.3997 |            263 |          263 |                 1      |                         1      |
| RidgeCV |   2000 |    1.5663 |            227 |          227 |                 0.2703 |                         0.8891 |
| RidgeCV |   2001 |    0.9916 |            238 |          238 |                 0.1914 |                         0.8341 |
| RidgeCV |   2002 |    1.0374 |            238 |          238 |                 0.1782 |                         0.8278 |
| RidgeCV |   2003 |    1.0601 |            238 |          238 |                 0.1906 |                         0.8658 |
| RidgeCV |   2004 |    1.0682 |            238 |          238 |                 0.1955 |                         0.8849 |
| RidgeCV |   2005 |    0.5747 |            250 |          250 |                 0.1108 |                         0.7844 |
| RidgeCV |   2006 |    0.5737 |            251 |          251 |                 0.112  |                         0.7764 |
| RidgeCV |   2007 |    0.5721 |            255 |          255 |                 0.1136 |                         0.7295 |
| RidgeCV |   2008 |    0.5747 |            256 |          256 |                 0.1161 |                         0.7305 |
| RidgeCV |   2009 |    1.0955 |            256 |          256 |                 0.2225 |                         0.8804 |
| RidgeCV |   2010 |    1.1463 |            256 |          256 |                 0.2352 |                         0.8459 |
| RidgeCV |   2011 |    1.1396 |            256 |          256 |                 0.2374 |                         0.8486 |
| RidgeCV |   2012 |    1.1376 |            256 |          256 |                 0.2426 |                         0.8637 |
| RidgeCV |   2013 |    0.2955 |            256 |          256 |                 0.064  |                         0.6737 |
| RidgeCV |   2014 |    0.2963 |            257 |          257 |                 0.0652 |                         0.6603 |
| RidgeCV |   2015 |    0.1145 |            258 |          258 |                 0.0254 |                         0.6303 |
| RidgeCV |   2016 |    0.1153 |            258 |          258 |                 0.0257 |                         0.6294 |
| RidgeCV |   2017 |    0.6328 |            263 |          263 |                 0.1417 |                         0.8162 |
| RidgeCV |   2018 |    0.2973 |            263 |          263 |                 0.0669 |                         0.7602 |
| RidgeCV |   2019 |    0.2986 |            263 |          263 |                 0.0679 |                         0.7482 |

## Corrected vs original saved results

| model                 | start   |   months | CAGR   | ann_mean   | ann_vol   |   Sharpe | max_drawdown   |   ending_wealth | pipeline                                      | corr_LS_monthly_new_vs_old   |
|:----------------------|:--------|---------:|:-------|:-----------|:----------|---------:|:---------------|----------------:|:----------------------------------------------|:-----------------------------|
| OLS                   | 2000-01 |      240 | 31.3%  | 28.3%      | 13.0%     |     2.17 | -6.9%          |          230.3  | corrected                                     | 0.94                         |
| Ridge                 | 2000-01 |      240 | 31.3%  | 28.3%      | 13.0%     |     2.17 | -6.9%          |          230.32 | corrected                                     | 0.94                         |
| RidgeCV               | 2000-01 |      240 | 27.2%  | 25.2%      | 14.0%     |     1.81 | -9.2%          |          123.42 | corrected                                     | 0.84                         |
| Lasso                 | 2000-01 |      240 | 11.4%  | 12.3%      | 17.3%     |     0.71 | -38.9%         |            8.66 | corrected                                     | 0.77                         |
| LassoCV               | 2000-01 |      240 | 18.8%  | 18.5%      | 15.2%     |     1.22 | -19.5%         |           31.62 | corrected                                     | 0.75                         |
| OLS                   | 2002-01 |      216 | 26.8%  | 24.5%      | 10.8%     |     2.28 | -6.9%          |           71.55 | corrected                                     | 0.94                         |
| Ridge                 | 2002-01 |      216 | 26.8%  | 24.5%      | 10.8%     |     2.28 | -6.9%          |           71.56 | corrected                                     | 0.94                         |
| RidgeCV               | 2002-01 |      216 | 23.5%  | 21.9%      | 11.8%     |     1.86 | -9.2%          |           44.65 | corrected                                     | 0.84                         |
| Lasso                 | 2002-01 |      216 | 10.1%  | 10.3%      | 11.5%     |     0.89 | -15.3%         |            5.63 | corrected                                     | 0.77                         |
| LassoCV               | 2002-01 |      216 | 14.5%  | 14.2%      | 11.9%     |     1.2  | -18.6%         |           11.36 | corrected                                     | 0.75                         |
| OLS                   | 2005-01 |      180 | 23.3%  | 21.5%      | 8.5%      |     2.52 | -3.7%          |           23.22 | corrected                                     | 0.94                         |
| Ridge                 | 2005-01 |      180 | 23.3%  | 21.5%      | 8.5%      |     2.52 | -3.7%          |           23.22 | corrected                                     | 0.94                         |
| RidgeCV               | 2005-01 |      180 | 18.0%  | 17.2%      | 9.9%      |     1.74 | -9.2%          |           12.04 | corrected                                     | 0.84                         |
| Lasso                 | 2005-01 |      180 | 7.8%   | 8.0%       | 9.3%      |     0.86 | -15.3%         |            3.1  | corrected                                     | 0.77                         |
| LassoCV               | 2005-01 |      180 | 9.0%   | 9.1%       | 9.6%      |     0.95 | -18.6%         |            3.64 | corrected                                     | 0.75                         |
| OLS                   | 2010-01 |      120 | 20.6%  | 19.1%      | 6.5%      |     2.96 | -3.7%          |            6.53 | corrected                                     | 0.94                         |
| Ridge                 | 2010-01 |      120 | 20.6%  | 19.1%      | 6.5%      |     2.96 | -3.7%          |            6.53 | corrected                                     | 0.94                         |
| RidgeCV               | 2010-01 |      120 | 15.9%  | 15.2%      | 9.0%      |     1.68 | -8.0%          |            4.36 | corrected                                     | 0.84                         |
| Lasso                 | 2010-01 |      120 | 5.4%   | 5.6%       | 7.4%      |     0.75 | -10.4%         |            1.7  | corrected                                     | 0.77                         |
| LassoCV               | 2010-01 |      120 | 3.8%   | 3.9%       | 6.2%      |     0.63 | -11.8%         |            1.45 | corrected                                     | 0.75                         |
| mlp-256-128-64-32     | 2000-01 |      240 | 22.7%  | 22.1%      | 16.9%     |     1.3  | -40.8%         |           60.12 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-512-256-128-64-32 | 2000-01 |      240 | 20.9%  | 20.8%      | 17.6%     |     1.18 | -54.8%         |           44.75 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-64-32             | 2000-01 |      240 | 18.8%  | 19.3%      | 18.8%     |     1.03 | -61.8%         |           31.38 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-32-16-8           | 2000-01 |      240 | 20.2%  | 20.0%      | 17.7%     |     1.13 | -46.2%         |           39.52 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-128-64-32         | 2000-01 |      240 | 23.0%  | 22.8%      | 19.3%     |     1.18 | -61.2%         |           62.42 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| OLS                   | 2000-01 |      240 | 34.7%  | 30.9%      | 13.3%     |     2.33 | -5.5%          |          384.71 | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| Lasso                 | 2000-01 |      232 | 13.5%  | 14.3%      | 18.4%     |     0.78 | -21.2%         |           11.49 | ORIGINAL (old timing/imputer/CV) - historical | 0.77                         |
| Ridge                 | 2000-01 |      240 | 34.7%  | 30.9%      | 13.3%     |     2.33 | -5.5%          |          384.5  | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| LassoCV               | 2000-01 |      240 | 34.7%  | 31.0%      | 13.3%     |     2.32 | -5.2%          |          386.36 | ORIGINAL (old timing/imputer/CV) - historical | 0.75                         |
| RidgeCV               | 2000-01 |      240 | 34.7%  | 30.9%      | 13.3%     |     2.33 | -5.5%          |          384.11 | ORIGINAL (old timing/imputer/CV) - historical | 0.84                         |
| hgbr                  | 2000-01 |      240 | 29.5%  | 27.6%      | 17.3%     |     1.6  | -51.7%         |          175.9  | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-256-128-64-32     | 2002-01 |      216 | 22.4%  | 21.1%      | 12.2%     |     1.73 | -22.8%         |           37.8  | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-512-256-128-64-32 | 2002-01 |      216 | 22.9%  | 21.4%      | 11.3%     |     1.9  | -11.5%         |           41.1  | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-64-32             | 2002-01 |      216 | 21.0%  | 20.0%      | 12.6%     |     1.58 | -11.7%         |           31.05 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-32-16-8           | 2002-01 |      216 | 18.4%  | 17.7%      | 11.6%     |     1.53 | -17.2%         |           21.03 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-128-64-32         | 2002-01 |      216 | 25.7%  | 23.8%      | 12.5%     |     1.91 | -6.8%          |           60.94 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| OLS                   | 2002-01 |      216 | 30.0%  | 27.1%      | 11.2%     |     2.41 | -4.4%          |          111.77 | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| Lasso                 | 2002-01 |      209 | 9.6%   | 10.1%      | 14.0%     |     0.72 | -21.0%         |            4.93 | ORIGINAL (old timing/imputer/CV) - historical | 0.77                         |
| Ridge                 | 2002-01 |      216 | 30.0%  | 27.1%      | 11.2%     |     2.41 | -4.4%          |          111.71 | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| LassoCV               | 2002-01 |      216 | 29.9%  | 27.1%      | 11.3%     |     2.4  | -4.4%          |          111.22 | ORIGINAL (old timing/imputer/CV) - historical | 0.75                         |
| RidgeCV               | 2002-01 |      216 | 30.0%  | 27.1%      | 11.2%     |     2.41 | -4.4%          |          111.69 | ORIGINAL (old timing/imputer/CV) - historical | 0.84                         |
| hgbr                  | 2002-01 |      216 | 32.0%  | 28.9%      | 13.6%     |     2.13 | -13.1%         |          147.22 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-256-128-64-32     | 2005-01 |      180 | 23.1%  | 21.5%      | 11.1%     |     1.95 | -7.5%          |           22.54 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-512-256-128-64-32 | 2005-01 |      180 | 21.6%  | 20.2%      | 10.2%     |     1.98 | -11.1%         |           18.73 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-64-32             | 2005-01 |      180 | 21.5%  | 20.4%      | 12.7%     |     1.61 | -11.7%         |           18.5  | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-32-16-8           | 2005-01 |      180 | 18.9%  | 18.0%      | 10.5%     |     1.7  | -11.0%         |           13.41 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-128-64-32         | 2005-01 |      180 | 24.0%  | 22.3%      | 11.6%     |     1.93 | -6.4%          |           25.11 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| OLS                   | 2005-01 |      180 | 25.6%  | 23.5%      | 9.3%      |     2.53 | -4.1%          |           30.7  | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| Lasso                 | 2005-01 |      174 | 5.1%   | 5.7%       | 11.7%     |     0.48 | -21.0%         |            2.06 | ORIGINAL (old timing/imputer/CV) - historical | 0.77                         |
| Ridge                 | 2005-01 |      180 | 25.6%  | 23.4%      | 9.3%      |     2.53 | -4.1%          |           30.69 | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| LassoCV               | 2005-01 |      180 | 25.5%  | 23.3%      | 9.3%      |     2.5  | -4.0%          |           30.11 | ORIGINAL (old timing/imputer/CV) - historical | 0.75                         |
| RidgeCV               | 2005-01 |      180 | 25.6%  | 23.4%      | 9.3%      |     2.53 | -4.1%          |           30.68 | ORIGINAL (old timing/imputer/CV) - historical | 0.84                         |
| hgbr                  | 2005-01 |      180 | 27.1%  | 25.0%      | 12.6%     |     1.99 | -13.1%         |           36.6  | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-256-128-64-32     | 2010-01 |      120 | 17.6%  | 16.6%      | 8.2%      |     2.03 | -7.5%          |            5.06 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-512-256-128-64-32 | 2010-01 |      120 | 16.7%  | 15.9%      | 8.4%      |     1.89 | -11.1%         |            4.69 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-64-32             | 2010-01 |      120 | 14.6%  | 14.2%      | 10.3%     |     1.38 | -11.7%         |            3.91 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-32-16-8           | 2010-01 |      120 | 15.3%  | 14.8%      | 9.8%      |     1.51 | -11.0%         |            4.14 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| mlp-128-64-32         | 2010-01 |      120 | 17.5%  | 16.6%      | 8.8%      |     1.88 | -6.4%          |            5.02 | ORIGINAL (old timing/imputer/CV) - historical |                              |
| OLS                   | 2010-01 |      120 | 20.5%  | 19.1%      | 6.9%      |     2.78 | -4.1%          |            6.48 | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| Lasso                 | 2010-01 |      118 | 0.9%   | 1.4%       | 9.6%      |     0.14 | -18.6%         |            1.1  | ORIGINAL (old timing/imputer/CV) - historical | 0.77                         |
| Ridge                 | 2010-01 |      120 | 20.5%  | 19.1%      | 6.9%      |     2.78 | -4.1%          |            6.48 | ORIGINAL (old timing/imputer/CV) - historical | 0.94                         |
| LassoCV               | 2010-01 |      120 | 20.2%  | 18.8%      | 6.8%      |     2.75 | -4.0%          |            6.31 | ORIGINAL (old timing/imputer/CV) - historical | 0.75                         |
| RidgeCV               | 2010-01 |      120 | 20.5%  | 19.1%      | 6.9%      |     2.78 | -4.1%          |            6.48 | ORIGINAL (old timing/imputer/CV) - historical | 0.84                         |
| hgbr                  | 2010-01 |      120 | 18.1%  | 17.3%      | 10.0%     |     1.72 | -13.1%         |            5.29 | ORIGINAL (old timing/imputer/CV) - historical |                              |

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
  2000,
  2019
 ],
 "return_units": "panel ret in percent; converted to decimals once here (min=-0.994 max=19.884)",
 "tall_equals_wide": true,
 "long_short_equals_bin5_minus_bin1": true,
 "portfolio_weights_sum_to_one_per_bin_month": true,
 "no_empty_portfolios": true,
 "wealth_convention": "wealth = cumprod(1 + r_LS), start $1. Exposure convention: $1 long the top quintile and $1 short the bottom quintile per $1 of strategy equity, rebalanced monthly, before transaction costs, borrow fees and financing/short-rebate effects."
}
```