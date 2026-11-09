# later, not now: add more models (OLS, MLP) to ipynb data that feeds into this
# I'd love to add in lazy prices!

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import datetime
import statsmodels.formula.api as smf
import plotly.express as px

class _CompatRemainderColsList(list):
    """Fallback for loading sklearn<=1.6 ColumnTransformer pickles in sklearn>=1.8."""


def _ensure_sklearn_joblib_compatibility():
    """Register compatibility shims needed to unpickle the shipped sklearn pipelines."""

    import sys
    from sklearn.compose import _column_transformer

    # Pipelines saved from the notebook pickle its custom classes under __main__; the same
    # classes live in the stockpred package (the notebook's code mirrors it).
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        from stockpred.nn import MLPEnsemble
        from stockpred.preprocessing import MissingnessFilter, MonthlyCrossSectionalMeanImputer, Winsorizer
        for cls in (MissingnessFilter, MonthlyCrossSectionalMeanImputer, Winsorizer, MLPEnsemble):
            if not hasattr(main_module, cls.__name__):
                setattr(main_module, cls.__name__, cls)

    if not hasattr(_column_transformer, "_RemainderColsList"):
        _column_transformer._RemainderColsList = _CompatRemainderColsList


def load_model_pipeline(model_path):
    """Load a saved model pipeline after applying sklearn/joblib compatibility shims."""

    _ensure_sklearn_joblib_compatibility()

    import joblib

    return joblib.load(model_path)


def _get_mlp_feature_importances(estimator, feature_names):
    """Estimate MLP feature importance by propagating absolute weights to the output layer."""

    if not hasattr(estimator, "coefs_") or not estimator.coefs_:
        return None

    downstream = np.ones(estimator.coefs_[-1].shape[1])
    for weights in reversed(estimator.coefs_):
        downstream = np.abs(np.asarray(weights)) @ downstream

    return pd.Series(downstream, index=feature_names)


def _get_torch_ensemble_feature_importances(estimator, feature_names):
    """Same absolute-weight propagation as for sklearn MLPs, averaged over the ensemble's networks."""

    states = getattr(estimator, "states", None)
    if not states:
        return None

    total = None
    for state in states:
        weights = [np.abs(np.asarray(v)) for k, v in state.items() if k.endswith(".weight") and np.ndim(v) == 2]
        downstream = np.ones(weights[-1].shape[0])
        for w in reversed(weights):                      # torch weights are (out, in)
            downstream = w.T @ downstream
        total = downstream if total is None else total + downstream
    return pd.Series(total / len(states), index=feature_names)


def _get_hist_gradient_boosting_feature_importances(estimator, feature_names):
    """Estimate HGBR feature importance from cumulative split gain across all trees."""

    predictors = getattr(estimator, "_predictors", None)
    if predictors is None:
        return None

    importances = np.zeros(len(feature_names), dtype=float)

    for iteration_predictors in predictors:
        for predictor in iteration_predictors:
            nodes = predictor.nodes
            split_mask = nodes["is_leaf"] == 0
            feature_idx = nodes["feature_idx"][split_mask]
            valid_mask = (feature_idx >= 0) & (feature_idx < len(feature_names))
            np.add.at(importances, feature_idx[valid_mask], nodes["gain"][split_mask][valid_mask])

    return pd.Series(importances, index=feature_names)


def get_feature_importance_details(estimator, feature_names):
    """Return feature importance details for supported model types."""

    coef_attr = getattr(estimator, "coef_", None)
    if coef_attr is not None:
        return {
            "subheader": "🔢 Top Coefficients",
            "x_label": "Coefficient Value",
            "title": "Coefficient Magnitude",
            "series": pd.Series(np.asarray(coef_attr).ravel(), index=feature_names),
            "sort_by_absolute_value": True,
            "caption": None,
        }

    imp_attr = getattr(estimator, "feature_importances_", None)
    if imp_attr is not None:
        return {
            "subheader": "🌳 Feature Importances",
            "x_label": "Importance",
            "title": "Feature Importance",
            "series": pd.Series(imp_attr, index=feature_names),
            "sort_by_absolute_value": False,
            "caption": None,
        }

    if estimator.__class__.__name__ == "MLPRegressor":
        return {
            "subheader": "🧠 Estimated Neural Network Importances",
            "x_label": "Estimated Importance",
            "title": "Estimated Neural Network Importance",
            "series": _get_mlp_feature_importances(estimator, feature_names),
            "sort_by_absolute_value": False,
            "caption": "Estimated by propagating each feature's absolute learned weights through the network.",
        }

    if estimator.__class__.__name__ == "MLPEnsemble":
        return {
            "subheader": "🧠 Estimated Neural Network Importances",
            "x_label": "Estimated Importance",
            "title": "Estimated Neural Network Importance",
            "series": _get_torch_ensemble_feature_importances(estimator, feature_names),
            "sort_by_absolute_value": False,
            "caption": "Estimated by propagating absolute learned weights through each network, averaged over the "
                       "ensemble's seeds. A rough guide only: it ignores the ReLU non-linearities.",
        }

    if estimator.__class__.__name__ in ("HistGradientBoostingRegressor", "HistGradientBoostingClassifier"):
        return {
            "subheader": "🌳 Estimated Gradient Boosting Importances",
            "x_label": "Cumulative Split Gain",
            "title": "Estimated Gradient Boosting Importance",
            "series": _get_hist_gradient_boosting_feature_importances(estimator, feature_names),
            "sort_by_absolute_value": False,
            "caption": "Estimated from cumulative split gain across the fitted boosting trees.",
        }

    return None



# Set page configuration
st.set_page_config(
    page_title="ML Stock Prediction Evaluation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Results shown by the dashboard: built by build_dashboard_bundle.py from the v2 pipeline runs.
RESULTS_DIR = "output_corrected/dashboard"

# What the "Models on This Site" tab says about each model.
_RETURN = "Next month's return"
_RANK = "Next month's rank within the month"
MODEL_GUIDE = {
    "OLS": {"Family": "Linear", "Predicts": _RETURN, "Tuning": "None",
            "What it is": "Plain linear regression, the baseline for everything else."},
    "OLS-rank": {"Family": "Linear", "Predicts": _RANK, "Tuning": "None",
                 "What it is": "OLS with the rank target. Comparing it to OLS shows what the target does on its own, without a more flexible model."},
    "Ridge": {"Family": "Linear", "Predicts": _RETURN, "Tuning": "None (alpha = 1)",
              "What it is": "Linear regression with a small fixed penalty on coefficients. An untuned baseline."},
    "RidgeCV": {"Family": "Linear", "Predicts": _RETURN, "Tuning": "Penalty chosen each year",
                "What it is": "Linear regression that shrinks all coefficients toward zero."},
    "Lasso": {"Family": "Linear", "Predicts": _RETURN, "Tuning": "None (alpha = 1)",
              "What it is": "Linear regression that sets coefficients to exactly zero. An untuned baseline. This penalty is so strong that it keeps a single predictor every year (IndRetBig, an industry return), which creates many tied predictions."},
    "LassoCV": {"Family": "Linear", "Predicts": _RETURN, "Tuning": "Penalty chosen each year",
                "What it is": "Linear regression that sets coefficients to exactly zero, so it also picks variables. It typically keeps about 10 of the 200+ predictors."},
    "MLP-rank-256-128-64": {"Family": "Neural net", "Predicts": _RANK, "Tuning": "Training rounds chosen each year",
                            "What it is": "Three hidden layers (256, 128, 64) with dropout of 0.3, averaged over 5 networks. This is the neural net in the Full Code Example."},
    "MLP-rank-128-64": {"Family": "Neural net", "Predicts": _RANK, "Tuning": "Training rounds chosen each year",
                        "What it is": "Smaller network with two hidden layers (128, 64), averaged over 5 networks."},
    "MLP-huber-128-64": {"Family": "Neural net", "Predicts": _RETURN + " (clipped)", "Tuning": "Training rounds chosen each year",
                         "What it is": "Two hidden layers (128, 64), averaged over 5 networks. Its Huber loss reduces the influence of outliers."},
    "HGB-return": {"Family": "Gradient boosted trees", "Predicts": _RETURN, "Tuning": "Number of trees chosen each year",
                   "What it is": "Up to 300 small trees grown in sequence."},
    "HGB-rank": {"Family": "Gradient boosted trees", "Predicts": _RANK, "Tuning": "Number of trees chosen each year",
                 "What it is": "The same trees, predicting the rank target."},
    "HGB-quintile": {"Family": "Gradient boosted trees", "Predicts": "Which quintile next month's return lands in", "Tuning": "Number of trees chosen each year",
                     "What it is": "A classifier. It estimates the probability of each quintile, and stocks are sorted on their expected quintile."},
}


def model_index():
    """{dashboard model name: {run, model, description}} ({} if the bundle is absent)."""
    import json
    p = Path(RESULTS_DIR) / "models_index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def model_files(selected_model):
    """{test year: path} of the per-year saved pipelines of a model."""
    info = model_index().get(selected_model)
    if info is None:
        return {}
    folder = Path("output_corrected") / info["run"] / "models"
    return {int(p.stem.rsplit("_", 1)[1]): p for p in sorted(folder.glob(f"{info['model']}_*.joblib"))}


def select_model_path(selected_model, key):
    """Path of the saved pipeline to inspect. The pipeline is refit every test year, so each model
    gets a year picker."""
    files = model_files(selected_model)
    if not files:
        st.warning("No saved pipelines found for this model. Re-run run_linear_backtest.py / run_nn_backtest.py / "
                   "run_hgb_backtest.py.")
        return f"output_corrected/missing/{selected_model}.joblib"
    year = st.selectbox("Test year (the model was trained only on earlier years)", sorted(files), index=len(files) - 1,
                        key=key)
    return files[year]


# Cache data loading functions. Each cache is keyed on the files' modification times, so a running app picks up
# rebuilt data (build_dashboard_bundle.py) without a restart.
def _modified(*paths):
    return tuple(Path(p).stat().st_mtime if Path(p).exists() else None for p in paths)


def load_data():
    """Portfolio data (returns, portfolio stats, turnover) of the v2 pipeline."""
    files = [f"{RESULTS_DIR}/{f}" for f in ("returns_wide.csv", "port_stats_tall.csv", "turnover.csv")]
    return _load_data(RESULTS_DIR, _modified(*files))


@st.cache_data
def _load_data(source_dir, modified):
    """Load portfolio data from CSV files"""
    returns_wide_df = pd.read_csv(f'{source_dir}/returns_wide.csv')
    port_stats_tall_df = pd.read_csv(f'{source_dir}/port_stats_tall.csv')
    
    # Convert date to datetime
    returns_wide_df['date'] = pd.to_datetime(returns_wide_df['date'])
    port_stats_tall_df['date'] = pd.to_datetime(port_stats_tall_df['date'])
    
    # Set date as index for returns_wide_df
    returns_wide_df = returns_wide_df.set_index('date')
    
    # turnover data: model,bin,date,% buy,% sell,% hold
    turnover_df = pd.read_csv(f'{source_dir}/turnover.csv')
    turnover_df['date'] = pd.to_datetime(turnover_df['date'])
    turnover_df = turnover_df.set_index('date')
    
    return returns_wide_df, port_stats_tall_df, turnover_df

def load_costs():
    """Monthly long-short returns before and after trading costs (built by build_dashboard_bundle.py), or None."""
    p = Path(RESULTS_DIR) / "costs.csv"
    return _load_costs(str(p), _modified(p)) if p.exists() else None


@st.cache_data
def _load_costs(path, modified):
    return pd.read_csv(path, parse_dates=['date']).set_index('date')


@st.cache_data
def load_factors():
    """Load Fama-French factors for regression analysis"""
    start = '1964-01-01'
    start_date = datetime.strptime(start, '%Y-%m-%d')

    try:
        # Get FF5 and momentum
        ff_5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date)[0]
        ff_mom = pdr.get_data_famafrench('F-F_Momentum_Factor', start=start_date)[0]
        ff_mom.columns = ['Mom']  # rename

        # combine
        ff_factors = pd.merge(ff_5, ff_mom, left_index=True, right_index=True, how='outer')
        ff_factors = ff_factors.reset_index().rename(columns={"Mkt-RF": "mkt_excess", "Date": "date"})
        ff_factors["date"] = ff_factors["date"].dt.to_timestamp()
    except Exception:
        ff_factors = pd.read_csv("input_data/ff_factors.csv")
        ff_factors["date"] = pd.to_datetime(ff_factors["date"])
        ff_factors = ff_factors.drop(columns=["_merge"], errors="ignore")

    ff_factors["date"] = ff_factors["date"].apply(lambda x: x.replace(day=28))  # set to 28th of month
    ff_factors = ff_factors.set_index('date')
    
    return ff_factors

@st.cache_data
def load_zoo_returns():
    import os
    zoo_fname = 'input_data/zoo_returns_wide.csv'
    
    if not os.path.exists(zoo_fname):
        import openassetpricing as openap
        openap_obj = openap.OpenAP()  
        port_osap = openap_obj.dl_port('op', 'pandas').query('port== "LS"')
        port_osap['date'] = pd.to_datetime(port_osap['date'])
        # set port datetime to the 28th of the month like the other dataframes
        port_osap['date'] = port_osap['date'].apply(lambda x: x.replace(day=28))  
        
        # format to wide, rows are dates, columns are signal names, values are returns
        port_osap = port_osap[['signalname', 'date', 'ret']]
        port_osap = port_osap.set_index(['signalname','date']).unstack(level=0).droplevel(0, axis=1)
        port_osap.to_csv(zoo_fname, index=True)
    else:
        port_osap = pd.read_csv(zoo_fname)
        port_osap['date'] = pd.to_datetime(port_osap['date'])
        
    return port_osap

# Functions for creating plots and analyses
def create_cumulative_returns_plot(returns_wide_df, selected_model):
    """Create cumulative returns plot for a specific model"""
    plot_df = (1 + returns_wide_df.filter(like=f'{selected_model}_bin')).cumprod()
    
    # Add a month before the first date and set values to one for the first month
    row1 = pd.DataFrame(index=[plot_df.index[0] - pd.DateOffset(months=1)], 
                        data={c: [1] for c in plot_df.columns})
    plot_df = pd.concat([row1, plot_df])
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each bin
    for col in plot_df.columns:
        if 'LS' in col:
            ax.plot(plot_df.index, plot_df[col], linewidth=3, color=plt.cm.viridis(0),
                    label=col.replace(f'ret_{selected_model}_binLS', 'Long-Short'))
        else:
            bin_num = int(col.replace(f'ret_{selected_model}_bin', ''))
            # pattern from less to more dashed as bin_num increases 1 to 5
            linestyle = '-' if bin_num == 1 else '--' if bin_num == 2 else '-.' if bin_num == 3 else ':'
            # increment color for each bin
            color = plt.cm.viridis(1.2-(bin_num / 5)) # top bin is closest to 0, which is the LS color
            ax.plot(plot_df.index, plot_df[col], color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7, label=f'Portfolio {bin_num}')
    
    # If max in the plot_df is greater than 50, log scale the y-axis
    if plot_df.max().max() > 50:
        ax.set_yscale('log')
        # But make the numbers on the y-axis more readable
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    
    ax.set_title(f'Cumulative Returns for {selected_model} Model', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_cumulative_returns_against_zoo(returns_wide_df, selected_model, zoo_returns_df, log_scale=True):
    """Plot cumulative returns of selected model against the zoo returns"""
    
    this_returns_df = returns_wide_df.filter(like=f'{selected_model}_binLS')
    this_returns_df.columns = [selected_model]
    
    plot_df = (this_returns_df
               .merge((zoo_returns_df.set_index('date'))/100, # zoo returns are in percent, convert to decimal before cumprod
                      left_index=True, right_index=True, how='inner')
    )
    
    plot_df = (1 + plot_df).cumprod()
    
    # Add a month before the first date and set values to one for the first month
    row1 = pd.DataFrame(index=[plot_df.index[0] - pd.DateOffset(months=1)], 
                        data={c: [1] for c in plot_df.columns})
    plot_df = pd.concat([row1, plot_df])
    
    # plot this with plotly. all the zoo returns in grey, the selected model in red
    import plotly.express as px
    fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, 
                  title=f'',
                  # color all in grey but first
                  color_discrete_sequence=['red']+['grey']*(len(plot_df.columns)-1),
                  )
    # the selected model should be prominent and "on top" of other lines
    fig.update_traces(opacity=.1, 
                      # select all but selected_model (first column)
                      selector=lambda x: x.name != plot_df.columns[0])
    
    # make the selected_model more prominent (wider line)
    fig.update_traces(line=dict(width=4), 
                      selector=lambda x: x.name == plot_df.columns[0])
    
    fig.update_yaxes(title_text='', title_font=dict(size=14))
    fig.update_xaxes(title_text='', title_font=dict(size=14),
                     dtick="M12", tickformat="%Y")    
       
    # can the legend have a filter?
     
        
    # If max in the plot_df is greater than 50, log scale the y-axis
    if log_scale:
        fig.update_yaxes(type="log")
    # if plot_df.max().max() > 50:
    #     ax.set_yscale('log')
    #     # But make the numbers on the y-axis more readable
    #     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    
    # ax.set_title(f'Cumulative Returns for {
        
    return fig

def create_moving_average_plot(returns_wide_df, selected_model, window=12):
    """Create moving average returns plot for a specific model"""
    plot_df = returns_wide_df.filter(like=f'{selected_model}_bin').fillna(0) * 100
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate and plot moving averages
    for col in plot_df.columns:
        ma_series = plot_df[col].rolling(window=window, center=True).mean()
        if 'LS' in col:
            ax.plot(ma_series.index, ma_series, linewidth=3, color=plt.cm.viridis(0),
                    label=col.replace(f'ret_{selected_model}_binLS', 'Long-Short'))
        else:
            bin_num = int(col.replace(f'ret_{selected_model}_bin', ''))
            # pattern from less to more dashed as bin_num increases 1 to 5
            linestyle = '-' if bin_num == 1 else '--' if bin_num == 2 else '-.' if bin_num == 3 else ':'
            # increment color for each bin
            color = plt.cm.viridis(1.2-(bin_num / 5)) # top bin is closest to 0, which is the LS color
            ax.plot(ma_series.index, ma_series, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7, label=f'Portfolio {bin_num}')
    
    ax.set_title(f'{window}-Month Moving Average Returns for {selected_model} Model', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Monthly Return (%)', fontsize=14)
    ax.legend(loc='lower right')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    fig.tight_layout()
    
    return fig

def calculate_table_1(returns_wide_df, selected_model, ff_factors):
    """Calculate Table 1 regression results for a specific model"""
    portfolios = [col for col in returns_wide_df.columns if '_'+selected_model+'_' in col]
    
    # Convert returns to percentages
    df = returns_wide_df[portfolios] * 100
    
    # Merge with factors
    reg_df = df.merge(
        ff_factors,
        left_index=True, right_index=True, how='left'
    )
    
    # Define factor model formulas
    factor_models = {
        'r^e': '1',
        'CAPM': 'mkt_excess',
        'FF3': 'mkt_excess + SMB + HML',
        'FF4': 'mkt_excess + SMB + HML + Mom',
        'FF5': 'mkt_excess + SMB + HML + RMW + CMA',
        'FF6': 'mkt_excess + SMB + HML + RMW + CMA + Mom'
    }
    
    # Pre-built output table
    index = pd.MultiIndex.from_product([factor_models.keys(), ['alpha', 't-stat']], 
                                      names=['Model', 'Metric'])
    results = pd.DataFrame(index=index, columns=portfolios, dtype=float)
    
    # Run regressions for each portfolio and model
    for portfolio in portfolios:
        reg_df[portfolio] = reg_df[portfolio] - reg_df['RF']  # excess returns
        for model_name, formula in factor_models.items():
            reg = smf.ols(formula=f'Q("{portfolio}") ~ {formula}', data=reg_df).fit()
            # extract the intercept coef and t-stat
            alpha = reg.params['Intercept']
            t_stat = reg.tvalues['Intercept']
            results.at[(model_name, 'alpha'), portfolio] = alpha
            results.at[(model_name, 't-stat'), portfolio] = t_stat
    
    # Clean up column names for display
    results.columns = [col.replace(f'ret_{selected_model}_', '') for col in results.columns]
    
    return results

def calculate_sharpe_ratios(returns_wide_df, selected_model):
    """Calculate annualized Sharpe ratios for portfolios"""
    portfolios = [col for col in returns_wide_df.columns if '_'+selected_model+'_' in col]
    
    mean_returns = returns_wide_df[portfolios].mean()
    std_returns = returns_wide_df[portfolios].std()
    sharpe_ratios = (mean_returns / std_returns) * np.sqrt(12)  # Annualized
    
    # Clean up labels
    sharpe_ratios.index = [col.replace(f'ret_{selected_model}_', '') for col in sharpe_ratios.index]
    
    return sharpe_ratios

# Compare all LS portfolios on a single chart
def create_ls_comparison_chart(returns_wide_df, models):
    """Create a chart comparing Long-Short portfolios across models"""
    ls_columns = [col for col in returns_wide_df.columns if 'binLS' in col and any(model in col for model in models)]
    
    if not ls_columns:
        return None, "No Long-Short portfolios found for the selected models."
    
    plot_df = (1 + returns_wide_df[ls_columns]).cumprod()
    
    # Add a month before the first date and set values to one for the first month
    row1 = pd.DataFrame(index=[plot_df.index[0] - pd.DateOffset(months=1)], 
                        data={c: [1] for c in plot_df.columns})
    plot_df = pd.concat([row1, plot_df])
    
    # Rename columns for better readability
    plot_df.columns = [col.split('_')[1] for col in plot_df.columns]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a color cycle for different models
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(plot_df.columns)))
    
    # Plot each model's LS portfolio
    for i, col in enumerate(plot_df.columns):
        ax.plot(plot_df.index, plot_df[col], linewidth=2, color=colors[i], label=col)
    
    # If max in the plot_df is greater than 50, log scale the y-axis
    if plot_df.max().max() > 50:
        ax.set_yscale('log')
        # But make the numbers on the y-axis more readable
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    
    ax.set_title(f'Comparison of Long-Short Portfolio Returns Across Models', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig, None

def create_ls_comparison_chart_plotly(returns_wide_df, models):
    """Create an interactive Plotly chart comparing Long-Short portfolios across models"""
    
    ls_columns = [col for col in returns_wide_df.columns if 'binLS' in col and any(model in col for model in models)]
    
    if not ls_columns:
        return None, "No Long-Short portfolios found for the selected models."
    
    plot_df = (1 + returns_wide_df[ls_columns]).cumprod()
    
    # Add a month before the first date and set values to one for the first month
    row1 = pd.DataFrame(index=[plot_df.index[0] - pd.DateOffset(months=1)], 
                        data={c: [1] for c in plot_df.columns})
    plot_df = pd.concat([row1, plot_df])
    
    # Rename columns for better readability
    plot_df.columns = [col.split('_')[1] for col in plot_df.columns]
    plot_df.index.name = 'Date'
    log_scale = plot_df.max().max() > 50

    # Convert to long format for Plotly Express
    plot_df_long = plot_df.reset_index().melt(id_vars='Date', var_name='Model', value_name='Value of $1')

    colors = model_colors(plot_df.columns)
    fig = px.line(plot_df_long, x='Date', y='Value of $1', color='Model', color_discrete_map=colors,
                  log_y=log_scale, height=600, template='plotly_white')
    fig.update_traces(line=dict(width=2.2), hovertemplate='%{y:,.1f}')
    fig.update_layout(showlegend=False, hovermode='x unified', margin=dict(l=10, r=190, t=20, b=10),
                      yaxis_title='Value of $1 invested', xaxis_title='')
    if log_scale:
        ticks = [v for v in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] if v <= plot_df.max().max() * 2]
        fig.update_yaxes(tickvals=ticks, ticktext=[f"{v:,}" for v in ticks])

    # Label each line at its end, nudging labels apart so they don't overlap
    end = plot_df.iloc[-1].astype(float)
    pos = np.log10(end) if log_scale else end
    lo, hi = (np.log10(plot_df.min().min()), np.log10(plot_df.max().max())) if log_scale else (plot_df.min().min(), plot_df.max().max())
    gap = 0.045 * (hi - lo)
    pos = pos.sort_values()
    for i in range(1, len(pos)):
        pos.iloc[i] = max(pos.iloc[i], pos.iloc[i - 1] + gap)
    for model, y in pos.items():
        fig.add_annotation(x=plot_df.index[-1], y=y, text=f"<b>{model}</b>", showarrow=False, xanchor='left',
                           xshift=8, font=dict(color=colors[model], size=13))
    return fig, None


# One fixed, easy-to-tell-apart color per model, so a model keeps its color whatever else is selected
MODEL_COLORS = {
    "OLS": "#1F77B4", "LassoCV": "#9467BD", "MLP-rank-256-128-64": "#D62728", "HGB-return": "#2CA02C",
    "HGB-quintile": "#FF7F0E", "Ridge": "#17BECF", "RidgeCV": "#8C564B", "Lasso": "#BCBD22",
    "MLP-rank-128-64": "#E377C2", "MLP-huber-128-64": "#7F7F7F", "OLS-rank": "#222222", "HGB-rank": "#00695C",
}


def model_colors(models):
    spare = iter(px.colors.qualitative.Dark24)
    return {m: MODEL_COLORS.get(m) or next(spare) for m in models}

def calculate_multi_model_table_1(returns_wide_df, models, ff_factors):
    """Calculate Table 1 regression results for multiple models' LS portfolios"""
    ls_columns = [col for col in returns_wide_df.columns if 'binLS' in col and any(model in col for model in models)]
    
    if not ls_columns:
        return None, "No Long-Short portfolios found for the selected models."
    
    # Convert returns to percentages
    df = returns_wide_df[ls_columns] * 100
    
    # Merge with factors
    reg_df = df.merge(
        ff_factors,
        left_index=True, right_index=True, how='left'
    )
    
    # Define factor model formulas
    factor_models = {
        'r^e': '1',
        'CAPM': 'mkt_excess',
        'FF3': 'mkt_excess + SMB + HML',
        'FF4': 'mkt_excess + SMB + HML + Mom',
        'FF5': 'mkt_excess + SMB + HML + RMW + CMA',
        'FF6': 'mkt_excess + SMB + HML + RMW + CMA + Mom'
    }
    
    # Pre-built output table
    index = pd.MultiIndex.from_product([factor_models.keys(), ['alpha', 't-stat']], 
                                      names=['Model', 'Metric'])
    results = pd.DataFrame(index=index, columns=ls_columns, dtype=float)
    
    # Run regressions for each LS portfolio and model
    for portfolio in ls_columns:
        reg_df[portfolio] = reg_df[portfolio] - reg_df['RF']  # excess returns
        for model_name, formula in factor_models.items():
            reg = smf.ols(formula=f'Q("{portfolio}") ~ {formula}', data=reg_df).fit()
            # extract the intercept coef and t-stat
            alpha = reg.params['Intercept']
            t_stat = reg.tvalues['Intercept']
            results.at[(model_name, 'alpha'), portfolio] = alpha
            results.at[(model_name, 't-stat'), portfolio] = t_stat
    
    # Clean up column names for display - extract just the model name
    results.columns = [col.split('_')[1] for col in results.columns]
    
    return results, None

# Introduction page content
def intro_page():
    st.title("📊 Stock Prediction Dashboard (Lehigh FIN377)")
    
    st.markdown("""
    
    This dashboard provides comprehensive analytics for evaluating and comparing machine learning models created for stock return prediction.
    The models shown here were trained in the python notebook in the associated repo, and this code can be easily adjusted to train other models.
    
    This project is inspired by student projects (listed below) from [Lehigh University's FIN377 course.](https://ledatascifi.github.io/ledatascifi-2025/content/about/hall_of_awesomeness.html) 
    
    ## Key Features:
    
    **1. Clear and intuitive explanation of the methodology**
    - Understand the data processing necessary
    - Learn about how the machine learning models are trained 
    - How are the predictions of the models used to create portfolios?
    - See key code snippets for each step, and the full stack code base
    
    **2. Compare the machine learning models we create**
    - View long-short portfolio performance across multiple models
    - Analyze risk-adjusted returns and factor exposures
    - Compare alphas across different factor models
    - Compare Sharpe ratios, turnover, and performance after trading costs
    
    **3. Detailed model analysis**
    - Examine cumulative returns for individual portfolios
    - How does this model's returns compare to returns for 200+ professional models?
    - View moving average returns to analyze trends
    - Estimate alpha for the portfolio to gauge statistical significance and risk-adjusted performance
    - Analyze portfolio turnover over time
    - Model interpretation: Understand the variables used in the model and their significance
    
    ## Navigation:
    
    Use the sidebar to navigate between different pages:
    - **Introduction**: This overview page
    - **Background**: Learn how models are trained, the data used, and how return predictions are turned into portfolios
    - **Compare Models**: Compare performance across multiple models
    - **Model Details**: Deep dive into a specific model
    - **Trading the Model in Practice**: Taxes, account types, a manageable 20-stock version, and how professional funds trade
    
    ## Getting Started:
    
    Select a page from the sidebar to begin exploring the data.
    
    ## Inspiration
    
    - [Neural Network Dashboard](https://lehigh-asset-pricing.streamlit.app/) by Joseph Carruth, Jay Geneve, Michael Jamesley, and Evan Trock
    - [Lazy Prices Replication](https://lazypricesreplication.streamlit.app/) by Hannah Gordon, Akanksha Gavade, Marti Figueres, and Henry Piotrowski
    - [OpenAssetPricing](www.openassetpricing.com) by Andrew Chen and Tom Zimmermann
    - [Assaying Anomalies](https://sites.psu.edu/assayinganomalies/) by Robert Novy-Marx  and Mihail Velikov
    - [FIN377 textbook chapter on Assaying Anomalies](https://ledatascifi.github.io/ledatascifi-2025/content/05/05d_AssetPricingAnomalyTable1.html)
    - [FIN377 textbook chapter on OpenAssetPricing](https://ledatascifi.github.io/ledatascifi-2025/content/05/05e_OpenAP_anomaly_plot.html)
    
    ## Going further?
    
    - You can clone the repo and build your own models. You'll need CRSP return data.
    - What about the role of feature engineering? We only use 200 high quality signals from the OpenAssetPricing dataset here. What if we used more signals?
        - ["Machine Learning from a "Universe" of Signals: 
        The Role of Feature Engineering" (2025, Journal of Financial Economics)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5248179) by Li, Rossi, (Lehigh's own) 
        Sterlin Yan, and Zheng, compare the performance of (A) a ML model based on 18,000 possible accounting variables to (B) a ML model using 
        the smaller set of 200 predictive signals used 
        in this dashboard (which are curated from finance research by Chen and Zimmerman's OpenAssetPricing). They find that using the SMALLER but selected set of signals performs better! Students can replicate this exercise
        by obtaining accounting data from WRDS, creating the features, and then running code within this project with the input signal data replaced.
    - Is implementing this realistic? Is the performance here likely to be replicated in the future? And what about trading costs?
        - The predictive signals used here are curated from finance research but might 
        be overfit to the past and underperform going forward. This dashboard only tests the models performance through Dec 31, 2019 (2020 is left unused and 2021 onward was never loaded, so it is a clean holdout). Students can try these models
        on data from 2021 onward to see if the performance is replicated.
        - ["The Expected Returns on Machine-Learning Strategies" (2024, WP)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406) by Azevedo, Hoegner, and 
    Velikov perform a similar analysis to this dashboard, and goes further to then compute how these portfolios would perform after factoring in trading costs,
    pre-publication overfitting, and post-decimalization era of high liquidity.
    They find that these factors erode 57% of the performance of the machine learning models. Still, they find ML strategies are portfitable and
    have an expected gross Sharpe Ratio of 0.94, which drops to 0.84, net of expenses. For comparison, the historical average Sharpe Ratio 
    of the S&P 500 is about 0.43.    
    The "Additional Metrics" tab on the Compare Models page shows a simple version for the models here: performance after paying
    the bid-ask spread, and when trading only the most liquid stocks.
    """)
    
    st.success(
        "**How the results are built:** predictor factors are lagged one month, missing values are filled with "
        "same-month cross-sectional means, all preprocessing and tuning use only earlier years, and tied predictions "
        "are split evenly across portfolios. Caveats: equal-weighted quintiles, no delisting returns, results are before "
        "transaction costs except where noted, and the strong results lean on small, illiquid stocks (see the cost "
        "tables in Compare Models)."
    )
    index = model_index()
    if index:
        st.markdown("\n".join(f"- **{k}**: {v['description']}" for k, v in index.items()))

    # Load and display some basic statistics
    try:
        returns_wide_df, _, _ = load_data()
        model_names = list(set([col.split('_')[1] for col in returns_wide_df.columns if 'ret_' in col]))
        
        st.info(f"Data loaded successfully. Found {len(model_names)} models with out-of-sample performance for the period from {returns_wide_df.index.min().strftime('%Y-%m-%d')} to {returns_wide_df.index.max().strftime('%Y-%m-%d')}.")
        
        # Show available models
        st.subheader("Available Models")
        st.write(", ".join(sorted(model_names)))
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please ensure the portfolio data files exist in the 'output_corrected/dashboard' directory (run build_dashboard_bundle.py).")

# Compare models page content
def compare_models_page():
    st.title("Compare Model Performance")
    
    try:
        # Load data
        returns_wide_df, _, _ = load_data()
        
        # Extract model names
        model_names = sorted(set([col.split('_')[1] for col in returns_wide_df.columns if 'ret_' in col]))

        # Default selection: one model from each family
        default_models = [m for m in ['OLS', 'LassoCV', 'MLP-rank-256-128-64', 'HGB-return', 'HGB-quintile'] if m in model_names]

        # Model selection
        selected_models = st.multiselect(
            "Select models to compare:",
            options=model_names,
            default=default_models
        )

        if not selected_models:
            st.warning("Please select at least one model to continue.")
            return

        # Create tabs
        tab1, tab2, tab3, tab_list = st.tabs(["Cumulative Returns Comparison", "Risk-Adjusted Returns via Regression Analysis", "Additional Metrics", "Models on This Site"])

        # Tab 1: Cumulative Returns Comparison
        with tab1:
            st.header("Long-Short Portfolio Returns Comparison")
            
            st.markdown("""
            This chart compares the cumulative returns of Long-Short portfolios across the selected models.
            The Long-Short strategy involves buying the highest-ranked stocks (Portfolio 5) and shorting 
            the lowest-ranked stocks (Portfolio 1) for each model. If a model has a high t-stat, then the prediction model 
            is doing a good job of sorting stocks to buy and short compared to the factors in the regression model.
            """)
            

            
            fig, error_msg = create_ls_comparison_chart_plotly(returns_wide_df, selected_models)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(error_msg)
        
        # Tab 2: Factor Analysis Comparison
        with tab2:
            st.header("Factor Analysis Comparison")
            
            st.markdown("""
            This table compares the alpha (excess returns) of Long-Short portfolios across different factor models. 
            
            - The numbers are percent per month. Multiply by 12 to annualize.
            - Higher and statistically significant alphas indicate better model performance that cannot be explained by the risk factors in the given model.
            - The t-statistic indicates the statistical significance of the alpha, and those above 
            the 5% significance level are bolded.
            """)            
            
            # Load factors for analysis
            with st.spinner("Loading Fama-French factors..."):
                ff_factors = load_factors()
            
            results, error_msg = calculate_multi_model_table_1(returns_wide_df, selected_models, ff_factors)
            
            if results is not None:
                # Define custom styling function to highlight significant t-stats
                def highlight_significant(s, threshold=1.96, color='lightgreen'):
                    """Apply highlighting to cells with t-stats above threshold"""
                    styles = pd.DataFrame('', index=s.index, columns=s.columns)
                    
                    for idx in s.index:
                        if idx[1] == 't-stat':  # Check if the second level of MultiIndex is 't-stat'
                            for col in s.columns:
                                value = s.loc[idx, col]
                                if abs(value) > threshold:
                                    styles.loc[idx, col] = f'background-color: {color}'
                    
                    return styles
                
                def bold_significant(s, threshold=1.96):
                    """Apply bold font to significant t-stats"""
                    styles = pd.DataFrame('', index=s.index, columns=s.columns)
                    
                    for idx in s.index:
                        if idx[1] == 't-stat':  # Check if the second level of MultiIndex is 't-stat'
                            for col in s.columns:
                                value = s.loc[idx, col]
                                if abs(value) > threshold:
                                    styles.loc[idx, col] = 'font-weight: bold'
                    
                    return styles
                
                # Apply styling
                styled_table = (results.style
                               .format("{:.3f}", subset=pd.IndexSlice[[(m, 'alpha') for m in results.index.levels[0]], :])
                               .format("({:.2f})", subset=pd.IndexSlice[[(m, 't-stat') for m in results.index.levels[0]], :])
                               .apply(bold_significant, axis=None))
                
                # Create table styling
                table_styles = [
                    # borders 
                    dict(selector='th, thead th', props=[('text-align', 'center'), ('border-left', 'none'), ('border-right', 'none')]),
                    dict(selector='tr, th.row_heading.level1, td', props=[('border', 'none')]),

                    # background colors
                    dict(selector='thead th, th.row_heading.level0', props=[('background-color', '#f2f2f2')]),
                    dict(selector='td, th.row_heading.level1', props=[('background-color', '#f9f9f9')]),

                    # center values
                    dict(selector='td', props=[('text-align', 'center')]),
                    
                    # reduce the height of the table rows
                    dict(selector='td, tr, th.row_heading', props=[('padding-top', '1px'),('padding-bottom', '1px')]),
                    
                    # seperation between models
                    dict(selector='tr:nth-child(even)', props=[('border-bottom', '8px solid white')]),
                ]

                
                html_table = styled_table.set_table_styles(table_styles).to_html()
                
                # Add custom CSS
                st.markdown("""
                <style>
                table thead tr:nth-child(2) {
                    display: none;
                }
                
                table {
                    width: 85%;
                    margin-bottom: 20px;
                }        
                </style>
                """, unsafe_allow_html=True)
                
                # Display the table
                st.write(html_table, unsafe_allow_html=True)
                                
                st.markdown(r"""
                ### Notes:
                - Regression models:
                    - $r^e$ is the raw excess return
                    - CAPM is the Capital Asset Pricing, which means that we estimate a regression of the excess return of the portfolio on the market excess return:
                        - $ret_{i,t}-r_f = \alpha + \beta_{i} \cdot mkt\_excess_t + \epsilon_{i,t}$
                        - where $mkt\_excess_t$ is the market excess return (Mkt-RF)
                        - and $\alpha$ is the intercept and is interpreted as the excess return of the portfolio beyond its exposure to the market
                        - and $\beta_i$ is covariance with the market portfolio
                    - FF# are the Fama-French #-factor models and we estimate alpha for each similarly, but with more factors in the regression:
                        - FF3: Market, SMB (Small Minus Big), HML (High Minus Low)
                        - FF4: FF3 + Mom (Momentum)
                        - FF5: FF3 + RMW (Robust Minus Weak) + CMA (Conservative Minus Aggressive)
                        - FF6: FF5 + Mom
                """, unsafe_allow_html=True)
                
                ## todo - borrow/adapt attribution language from 9.6 on my site 
                        
        with tab3:
            st.header("Additional Performance Metrics about the Long-Short Portfolios")

            costs_df = load_costs()
            if costs_df is None:
                st.error("costs.csv is missing from output_corrected/dashboard (run build_dashboard_bundle.py with the input data).")
            else:
                start = st.radio("Sample starts in:", [2000, 2005, 2010], horizontal=True, key='metrics_start')

                def show(table):
                    st.dataframe(table.style.format('{:.1f}', na_rep='').format('{:.2f}', subset=['Sharpe ratio']))

                def show_ls(universe, returns):
                    show(long_short_metrics(costs_df, selected_models, universe, start, returns))

                st.markdown("""
                Long-short portfolios of all stocks, before trading costs. The sample ends in December 2019.
                - **Turnover** is the share of each leg traded in a month, one way. 100% means the whole leg was replaced.
                  Each month's trades start from last month's weights after they drifted with returns, so even a stock that
                  stays in the portfolio is traded a little to bring it back to equal weight.
                - **Short in illiquid** is the share of the short leg in the 20% of stocks with the lowest dollar
                  trading volume that month. These are the stocks that are hardest and most expensive to short.
                """)
                show_ls("All stocks", lambda d: d['ls_gross'])

                st.subheader("Performance with transaction costs")
                st.markdown("""
                The same portfolios, now paying half the bid-ask spread on every dollar traded, in both legs.
                - The spread for each stock is Open Asset Pricing's estimate from daily high and low prices (Corwin and
                  Schultz, 2012), measured before the month starts. Estimates like this tend to overstate the spreads of
                  liquid stocks, especially after 2005, so these costs are likely too high for those stocks.
                - **This estimate omits other costs**: price impact from trading large amounts, fees for borrowing the stocks
                  that are shorted, and allowance for stocks that can't be borrowed at all.
                """)
                show_ls("All stocks", lambda d: d['ls_net'])

                st.subheader("Performance with cost mitigation")
                st.markdown("""
                One simple way to cut costs: use only the most liquid 20% of stocks each month (about 880 stocks), ranked
                by dollar trading volume. The model's predictions are sorted into quintiles within that group, so the short
                leg has no illiquid stocks.

                The high-low estimate above is a poor guide to what trading these stocks costs. It charges them about
                25-30 bp each way after 2005, but estimates like it run 25-50 bp too high after 2003 compared with spreads
                measured from intraday quotes (Chen and Velikov, 2023). The actual cost of trading large, heavily traded
                stocks is a few basis points. So this table charges an assumed half-spread on every trade, plus a fee for
                borrowing the stocks in the short leg. Change the assumptions to see how the results respond.
                """)
                c1, c2 = st.columns(2)
                half_spread = c1.number_input("Half-spread, each way (basis points)", 0.0, 100.0, 5.0, 1.0,
                                              key='liquid_half_spread') / 10000
                borrow_fee = c2.number_input("Borrowing fee on the short leg (% per year)", 0.0, 10.0, 0.25, 0.25,
                                             key='borrow_fee') / 100
                st.caption("Spreads were wider before decimal prices (2001), so a few basis points understates costs in "
                           "2000-2002. Most stocks cost less than 1% a year to borrow, and large ones typically about "
                           "0.25% (D'Avolio, 2002). Stocks that are heavily shorted can cost much more.")
                show_ls("Most liquid 20%", lambda d: d['ls_gross'] - 2 * half_spread * (d['turnover_long'] + d['turnover_short'])
                        - borrow_fee / 12)

                st.subheader("Long-only performance")
                st.markdown("""
                Many investors can't short at all. This table holds only the long leg: the top quintile of the most liquid
                20% of stocks, paying the half-spread above. It is compared with the stock market (all US stocks,
                value-weighted). The Sharpe ratio here uses returns above the T-bill rate.
                """)
                ff = load_factors()
                rf = ff['RF'] / 100
                market = ff['mkt_excess'] / 100 + rf
                show(long_only_metrics(costs_df, selected_models, start,
                                       lambda d: d['long_gross'] - 2 * half_spread * d['turnover_long'], market, rf))

        with tab_list:
            models_on_this_site()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please make sure the portfolio data files exist in the 'output_corrected/dashboard' directory (run build_dashboard_bundle.py).")


def models_on_this_site():
    st.header("Models on This Site")

    st.markdown("""
    Every model below uses the same data and timeline. It is retrained each year from 2000 to 2019 on all earlier
    years, using the same predictors and preprocessing, and its predictions are sorted into the same 5 portfolios.
    The models differ in what they try to predict and how flexible they are.

    These are all of the models that the site can show. Use the box at the top of this page to choose the ones to compare.
    """)

    header = {"Family": '<a href="#about-the-estimators" target="_self">Estimator Family</a>',
              "Predicts": '<a href="#what-is-the-target" target="_self">Predicts (Target)</a>'}
    rows = [{"Model": name, **MODEL_GUIDE.get(name, {"What it is": info["description"]})} for name, info in model_index().items()]
    table = pd.DataFrame(rows).fillna("").rename(columns=header)
    st.markdown(table.to_html(index=False, escape=False, border=0, classes="model-guide"), unsafe_allow_html=True)
    st.markdown("<style>table.model-guide {width: 100%;} table.model-guide th {text-align: left;}</style>",
                unsafe_allow_html=True)

    st.markdown("""
    **Choosing the designs**
    - The penalties in RidgeCV and LassoCV, and the number of training rounds in the neural nets and HGB models, are chosen each year using only the last three years of the training window.
    - The HGB settings (other than the number of trees) were fixed before seeing any results.
    - The neural net designs were chosen by comparing several designs on rank correlation, validating on 2002-2004. That overlaps the first test years, so 2005 onward is the cleaner test of those designs.
    """)

    st.header("About the Estimators", anchor="about-the-estimators")
    st.markdown("""
            OLS Linear Regression:
            - Ordinary Least Squares (OLS) regression is a statistical method used to estimate the relationships between variables.
            - It is the O.G. and the most basic linear model.
            - It assumes a linear relationship and is easy to interpret.
            - However, it may not capture complex relationships in the data.
            - It is sensitive to outliers, multicollinearity, and may not perform well with high-dimensional data.
            - It is also prone to overfitting if the number of features is large compared to the number of observations.
            - Yet, it is a good starting point for understanding the data and establishing a baseline.
            - And it is _good enough_ for many applications, even when FANCY PANTS models are available.
            
            Ridge Regression:
            - Ridge regression is a type of linear regression that includes a regularization term to prevent overfitting.
            - This term penalizes large coefficients, which causes the OLS coefficients to shrink towards zero.
            - This stabilizes the predictions it makes when it sees new data and improves generalization to unseen data.
            - It is particularly useful when dealing with multicollinearity or when the number of features gets large.
            
            Lasso Regression:
            - Lasso regression is another type of linear regression that includes a regularization term.
            - Unlike ridge regression, lasso will shrink some coefficients to zero, effectively performing variable selection.
            - This means that it can help identify the most important features in the data. 
            - For the purpose of prediction, Lasso and Ridge are often similar, as the underlying idea is the same: OLS, but with smaller coefficients.
            
            RidgeCV and LassoCV:
            - The penalty strength (alpha) is chosen using only the past: the training window's last three years are held out in turn, each time training on the years before it.
            - Plain Ridge and Lasso with the default penalty (alpha = 1) are included as baselines.

            MLP (Multi-Layer Perceptron) aka Neural Network:
            - Two of the MLPs here predict each stock's rank of next-month return within its month, and one predicts a clipped return using a Huber loss. Each averages five randomly initialised networks.
            - MLP is a type of neural network that consists of multiple layers of interconnected nodes (neurons).
            - A punchy one-liner is that it is a "black box" model that can learn complex relationships in the data.
            - NNs are powerful and flexible, but they require careful tuning of hyperparameters and can be prone to overfitting.
            - Some projects have found them to work well in finance, for example, [Gu, Kelly & Xiu (RFS 2020)](https://academic.oup.com/rfs/article/33/5/2223/5758276/) and [OpenAssetPricing](https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/ML_portfolio_example.ipynb).

            Histogram Gradient Boosting (HGB):
            - HGB builds hundreds of small decision trees in sequence, and each new tree tries to fix the errors of the trees before it.
            - Trees pick up non-linear patterns and interactions between variables on their own, and they don't care about the scale of a variable.
            - "Histogram" means that it groups each variable's values into buckets before looking for splits, which makes it fast enough for millions of rows.
            - The main risk is overfitting: more trees always fit the training data better. Here, the number of trees is chosen using the last three years of the training window.
            """)

    st.subheader("What is the target?", anchor="what-is-the-target", divider="gray")
    st.markdown("""
    - A model doesn't have to predict the return itself. Since we only use predictions to sort stocks, we can instead predict each stock's rank within its month, or which quintile it lands in.
    - Those targets ignore how extreme an outlier is. A stock that rises 2,000% counts the same as the month's other top performers.
    - The "Predicts (Target)" column of the table above shows which target each model uses.
    """)


def return_stats(r, rf=None):
    """Annualized mean and volatility, Sharpe ratio (of returns above rf, if given) and max drawdown, in percent."""
    excess = r - rf.reindex(r.index) if rf is not None else r
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])
    return {
        'Mean (%/yr)': 1200 * r.mean(),
        'Std. dev. (%/yr)': 100 * np.sqrt(12) * r.std(),
        'Sharpe ratio': np.sqrt(12) * excess.mean() / excess.std(),
        'Max drawdown (%)': 100 * (1 - wealth / wealth.cummax()).max(),
    }


def cost_rows(costs_df, model, universe, start_year):
    """One model's months in costs.csv, and how many to skip when averaging turnover (nothing is traded in the
    first month of the whole sample)."""
    d = costs_df[(costs_df['model'] == model) & (costs_df['universe'] == universe) & (costs_df.index.year >= start_year)]
    return d, int(not d.empty and d.index.min() == costs_df.index.min())


def long_short_metrics(costs_df, models, universe, start_year, returns):
    """Long-short performance, turnover and illiquid shorts of each model. returns(d) gives the monthly
    long-short return from that model's rows of costs.csv."""
    rows = {}
    for m in models:
        d, skip = cost_rows(costs_df, m, universe, start_year)
        if d.empty:
            continue
        rows[m] = {**return_stats(returns(d)),
                   'Turnover long (%/mo)': 100 * d['turnover_long'].iloc[skip:].mean(),
                   'Turnover short (%/mo)': 100 * d['turnover_short'].iloc[skip:].mean(),
                   'Short in illiquid (%)': 100 * d['short_illiquid'].mean()}
    return pd.DataFrame(rows).T


def long_only_metrics(costs_df, models, start_year, returns, market, rf):
    """Long leg of the most liquid 20% of stocks, next to the market."""
    rows = {}
    for m in models:
        d, skip = cost_rows(costs_df, m, "Most liquid 20%", start_year)
        if d.empty:
            continue
        r = returns(d)
        rows[m] = {**return_stats(r, rf), 'Turnover (%/mo)': 100 * d['turnover_long'].iloc[skip:].mean(),
                   'Mean minus market (%/yr)': 1200 * (r - market.reindex(r.index)).mean()}
    if rows:
        months = r.index
        rows['Stock market'] = {**return_stats(market.reindex(months), rf), 'Mean minus market (%/yr)': 0.0}
    return pd.DataFrame(rows).T

# Model details page content
def model_details_page():
    st.title("Detailed Model Analysis")
    
    # try:
    # Load data
    returns_wide_df, port_stats_tall_df, _ = load_data()
    
    # Extract model names
    model_names = list(set([col.split('_')[1] for col in returns_wide_df.columns if 'ret_' in col]))
    
    # Make the first (default) model LassoCV
    model_names = ['LassoCV'] + sorted([m for m in model_names if m != 'LassoCV'])
    
    # Model selection
    selected_model = st.selectbox(
        "Select a model to analyze:",
        options=model_names
    )
    
    # Load factors for Table 1
    with st.spinner("Loading Fama-French factors..."):
        ff_factors = load_factors()
    
    # Create tabs
    tab1, tab8, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Cumulative Returns", "Against the Factor Zoo", "Moving Average Returns", "Regression Analysis", 
                                            "Model Statistics", "Turnover Patterns", "Model Pipeline", "Variable Interpretation"])
    
    # Tab 1: Cumulative Returns Plot
    with tab1:
        st.header("Cumulative Returns")
        st.markdown("""
        This chart shows how $1 invested at the beginning of the period would have grown over time 
        for each portfolio formed using the selected model. The Long-Short portfolio represents the 
        returns of buying the highest-ranked stocks (Portfolio 5) and shorting the lowest-ranked stocks (Portfolio 1).
        """)
        
        cum_returns_fig = create_cumulative_returns_plot(returns_wide_df, selected_model)
        st.pyplot(cum_returns_fig)
    
    with tab8:
        st.header("Cumulative Returns vs. Anomaly Zoo")
        st.markdown("""
        This chart shows how $1 invested at the beginning of the period would have grown over time 
        using the long-short portfolio for the selected model, compared to returns from the many anomaly
        portfolios proposed in the finance literature.
        
        Details about the zoo portfolios can be found [here](https://drive.google.com/file/d/1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2/view), and you
        can filter the graph below to show only the zoo portfolios you are interested in.
        """)
        
        # Load zoo returns
        zoo_returns_df = load_zoo_returns()
        
        # Create a multiselect widget with "Select All" as the first option
        zoo_input_columns = zoo_returns_df.columns[1:]  # Exclude date column
        zoo_options = ["Select All"] + list(zoo_input_columns) # 1: to exclude date column
        selected_zoo_returns = st.multiselect(
            "Select zoo returns to display:", 
            options=zoo_options, 
            default=["Select All"],
            key='zoo_returns'
        )
        
        # Handle "Select All" option
        if "Select All" in selected_zoo_returns:
            zoo_returns = list(zoo_input_columns)
        else:
            zoo_returns = selected_zoo_returns
        
        # If no options selected, default to all
        if not zoo_returns:
            zoo_returns = list(zoo_input_columns)
            
        zoo_returns_df = zoo_returns_df[['date'] + zoo_returns]
                        
        zoo_fig = plot_cumulative_returns_against_zoo(returns_wide_df, selected_model, zoo_returns_df)
        st.plotly_chart(zoo_fig, use_container_width=True)
    
    # Tab 2: Moving Average Returns Plot
    with tab2:
        st.header("Moving Average Returns")
        window = st.slider("Moving average window (months):", min_value=3, max_value=24, value=12, step=1)
        st.markdown("""
        To visualize the trend in returns over time, this chart shows the smoothed monthly returns 
        for each portfolio formed using the selected model. 
        To smooth monthly variation in the data, we show a moving average.
        The are raw returns, not excess returns. Click over to the regression tab to see excess returns
        and other risk adjustments.
        """)
        
        ma_returns_fig = create_moving_average_plot(returns_wide_df, selected_model, window)
        st.pyplot(ma_returns_fig)
    
    # Tab 3: Table 1 Regression Results
    with tab3:
        st.header("Regression Analysis (a la Table 1)")
        st.markdown("""
        This table shows the results of regressing portfolio returns on various factor models.
        The alpha represents the abnormal return that cannot be explained by the factors.
        The t-statistic indicates the statistical significance of the alpha, and those above 
        the 5% significance level are bolded.
        
        - You should first look at the alpha on the Long-Short portfolio. If this has a high t-stat,
        then the model is doing a good job of sorting stocks to buy and short.
        - We call this "Table 1" because the first table in an asset pricing paper often 
        shows this kind of analysis.
        - Bin1 is the lowest ranked stocks from the prediction model, and Bin5 is the highest ranked stocks.
        - The numbers are percent per month. Multiply by 12 to annualize.
        - Higher and statistically significant alphas indicate better model performance that cannot be explained by the risk factors in the given model.
        - The t-statistic indicates the statistical significance of the alpha, and those above the 5% significance level are bolded.

        """)
        
        table_1 = calculate_table_1(returns_wide_df, selected_model, ff_factors)
        table_1.rename(columns={'binLS': 'Long-Short'}, inplace=True)
        
        # Define custom styling function to highlight significant t-stats
        def highlight_significant(s, threshold=1.96, color='lightgreen'):
            """Apply highlighting to cells with t-stats above threshold"""
            styles = pd.DataFrame('', index=s.index, columns=s.columns)
            
            for idx in s.index:
                if idx[1] == 't-stat':  # Check if the second level of MultiIndex is 't-stat'
                    for col in s.columns:
                        value = s.loc[idx, col]
                        if abs(value) > threshold:
                            styles.loc[idx, col] = f'background-color: {color}'
            
            return styles
                
        def bold_significant(s, threshold=1.96):
            """Apply bold font to significant t-stats"""
            styles = pd.DataFrame('', index=s.index, columns=s.columns)
            
            for idx in s.index:
                if idx[1] == 't-stat':  # Check if the second level of MultiIndex is 't-stat'
                    for col in s.columns:
                        value = s.loc[idx, col]
                        if abs(value) > threshold:
                            styles.loc[idx, col] = 'font-weight: bold'
            
            return styles
                
        # Apply styling
        styled_table = (table_1.style
                        .format("{:.3f}", subset=pd.IndexSlice[[(m, 'alpha') for m in table_1.index.levels[0]], :])
                        .format("({:.2f})", subset=pd.IndexSlice[[(m, 't-stat') for m in table_1.index.levels[0]], :])
                        .apply(bold_significant, axis=None))
        
        # Create table styling
        table_styles = [
            # borders 
            dict(selector='th, thead th', props=[('text-align', 'center'), ('border-left', 'none'), ('border-right', 'none')]),
            dict(selector='tr, th.row_heading.level1, td', props=[('border', 'none')]),

            # background colors
            dict(selector='thead th, th.row_heading.level0', props=[('background-color', '#f2f2f2')]),
            dict(selector='td, th.row_heading.level1', props=[('background-color', '#f9f9f9')]),

            # center values
            dict(selector='td', props=[('text-align', 'center')]),
            
            # reduce the height of the table rows
            dict(selector='td, tr, th.row_heading', props=[('padding-top', '1px'),('padding-bottom', '1px')]),
            
            # seperation between models
            dict(selector='tr:nth-child(even)', props=[('border-bottom', '8px solid white')]),
        ]
                    
        # Generate HTML with styles
        html_table = styled_table.set_table_styles(table_styles).to_html()
        
        # Add custom CSS
        st.markdown("""
        <style>
        table thead tr:nth-child(2) {
            display: none;
        }
        
        table {
            width: 85%;
            margin-bottom: 20px;
        }        
        
        </style>
        """, unsafe_allow_html=True)
        
        # Display the table
        st.write(html_table, unsafe_allow_html=True)
    
        st.markdown(r"""
        ### Notes:
        - Regression models:
            - $r^e$ is the raw excess return
            - CAPM is the Capital Asset Pricing, which means that we estimate a regression of the excess return of the portfolio on the market excess return:
                - $ret_{i,t}-r_f = \alpha + \beta_{i} \cdot mkt\_excess_t + \epsilon_{i,t}$
                - where $mkt\_excess_t$ is the market excess return (Mkt-RF)
                - and $\alpha$ is the intercept and is interpreted as the excess return of the portfolio beyond its exposure to the market
                - and $\beta_i$ is covariance with the market portfolio
            - FF# are the Fama-French #-factor models and we estimate alpha for each similarly, but with more factors in the regression:
                - FF3: Market, SMB (Small Minus Big), HML (High Minus Low)
                - FF4: FF3 + Mom (Momentum)
                - FF5: FF3 + RMW (Robust Minus Weak) + CMA (Conservative Minus Aggressive)
                - FF6: FF5 + Mom
        """, unsafe_allow_html=True)
    
    
    # Tab 4: Model Statistics
    with tab4:
        st.header("Model Statistics")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sharpe Ratios")
            sharpe_ratios = calculate_sharpe_ratios(returns_wide_df, selected_model)
            st.dataframe(pd.DataFrame(sharpe_ratios, columns=['Sharpe Ratio']).style.format("{:.4f}"))
        
        with col2:
            st.subheader("Turnover Analysis")
            
            _, _, turnover_df = load_data()
            
            # Filter turnover data for the selected model
            turnover_model_df = turnover_df[turnover_df['model'] == selected_model]
            
            # Display turnover data
            turnover_model_df = turnover_model_df[['bin', '% buy', '% hold']].groupby('bin').mean()
            turnover_model_df.columns = [c.title() for c in turnover_model_df.columns]
            
            st.dataframe(turnover_model_df.style.format("{:.1%}"))

    with tab5:
        
        st.header("Turnover Patterns")
        
        _, _, turnover_df = load_data()
    
        # Filter turnover data for the selected model
        turnover_model_df = turnover_df[turnover_df['model'] == selected_model]

        # plot turnover_model_df variables over time 
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=turnover_model_df.reset_index(), x='date', y='% hold', hue='bin', ax=ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.set(xlabel='Date', ylabel='Hold Percentage', title='Percentage of Stocks Held Over Time')
        ax.set_ylim(0, 1)
        # Place legend to the right of the plot
        ax.legend(title='Portfolio', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
    with tab6:    
        
        st.header("Visualization of Model Pipeline")
        
        # load the model/model_name joblib pipeline
        
        model_path = select_model_path(selected_model, key="pipeline_year")
        pipeline = None
        try:
            pipeline = load_model_pipeline(model_path)
            # st.write(pipeline)
            
        except FileNotFoundError:
            st.error(f"Model pipeline file not found: {model_path}")
        except Exception as e:
            st.error(f"An error occurred while loading the model pipeline: {e}")
                        
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn import set_config
        import streamlit.components.v1 as components

        if pipeline:
            try:
                html_repr = pipeline._repr_html_()
            except AttributeError:
                print('Problem rendering pipeline.')
            else:
                # get the height of the html object made by html_repr 
                components.html(
                    html_repr,
                    width=1000,
                    height=400,     # adjust as needed
                    scrolling=True
                )
                
            # Print out variables in the different column transformers
            for name, step in pipeline.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    st.markdown(f"## Which columns go through which transformers?")
                    col1, col2 = st.columns(2)
                    
                    transformers_list = list(step.transformers)
                    
                    with col1:
                        transformer_name, transformer, columns = transformers_list[0]
                        st.markdown(f'**Columns in the {transformer_name} transformer**')
                        col_df = pd.DataFrame({f'Scroll to see all': columns})
                        st.dataframe(col_df, hide_index=True)
                    
                    with col2:
                        if len(transformers_list) > 1:
                            transformer_name, transformer, columns = transformers_list[1]
                            st.markdown(f'**Columns in the {transformer_name} transformer**')
                            col_df = pd.DataFrame({'Scroll to see applicable columns': columns})
                            st.dataframe(col_df, hide_index=True)

        
    with tab7:
        st.header("Which variables are important in this model?")

        top_n = st.slider("Top N features", 5, 50, 20, step=5)

        model_path = select_model_path(selected_model, key="importance_year")
        pipeline = None
        try:
            pipeline = load_model_pipeline(model_path)
        except FileNotFoundError:
            st.error(f"Model pipeline file not found: {model_path}")
        except Exception as e:
            st.error(f"An error occurred while loading the model pipeline: {e}")
            
        if pipeline:
            estimator = pipeline.named_steps[list(pipeline.named_steps)[-1]]
            feat_names = pipeline[:-1].get_feature_names_out()
            importance_details = get_feature_importance_details(estimator, feat_names)

            if importance_details is None or importance_details["series"] is None:
                st.info("This model does not expose feature importances in a way the dashboard can display.")
            else:
                st.subheader(importance_details["subheader"])

                importance_series = importance_details["series"]
                if importance_details["sort_by_absolute_value"]:
                    importance_series = importance_series.sort_values(key=lambda x: x.abs(), ascending=False)
                else:
                    importance_series = importance_series.sort_values(ascending=False)

                top = importance_series.head(top_n)

                fig3 = px.bar(
                    x=top.values,
                    y=top.index,
                    orientation='h',
                    labels={'x': importance_details["x_label"], 'y': 'Feature'},
                    title=importance_details["title"],
                    height=max(400, len(top)*25)
                )

                fig3.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    hoverlabel=dict(bgcolor="white", font_size=12),
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig3, use_container_width=True)

                if importance_details["caption"]:
                    st.caption(importance_details["caption"])

                    
    # except Exception as e:
    #     st.error(f"An error occurred: {e}")
    #     st.error("Please make sure the portfolio data files exist in the 'output_corrected/dashboard' directory (run build_dashboard_bundle.py).")

# Navigation sidebar
def sidebar_navigation():
    st.sidebar.title("Pages")
    page = st.sidebar.radio(
        "Go to:",
        ["Introduction", "Background & Methods", "Compare Models", "Dig into a Model", "Trading the Model in Practice"]
    )
    return page


def load_practice():
    """Monthly returns, turnover and trade counts of each model under several trading approaches, or None."""
    p = Path(RESULTS_DIR) / "practice.csv"
    return _load_practice(str(p), _modified(p)) if p.exists() else None


@st.cache_data
def _load_practice(path, modified):
    return pd.read_csv(path, parse_dates=['date']).set_index('date')


PRACTICE_BACKGROUND = r"""
*This page is about how the strategies on this site could be traded. It describes general US federal tax rules
for teaching purposes. It is not tax or investment advice.*

### TL;DR for an individual investor
- **These strategies trade a lot.** Most holdings are replaced within a month or two, so nearly every gain is
  short-term and, in a regular (taxable) brokerage account, is taxed like wages: up to 37%, plus 3.8% investment
  income tax for high earners.
- **In a taxable account, that tax erases the edge.** In a rough simulation for 2005-2019, the 20-stock version
  of the MLP-rank model grew about 14.5% a year before tax, but about 8.7% after tax at top-bracket rates. That is
  no better than simply holding an index fund (also about 8.7%, since an index fund rarely sells and is taxed
  mostly on its dividends). For a middle-bracket investor, about 11% a year was left, versus about 9% for the index fund.
- **Use a tax-advantaged account instead.** Inside a Roth IRA, a traditional IRA, or a 401(k), trades aren't
  taxed. A Roth is the best fit: gains are never taxed. Most 401(k) plans only offer a menu of funds, so you
  would need a plan with a self-directed brokerage window.
- **These accounts are long-only.** IRAs and 401(k) brokerage windows don't allow borrowing or short selling, so
  only the long-only versions on the next tab can be traded there.
- **Keep it manageable.** Holding the top 20 stocks among the most liquid 20% is about 400 trades a year, versus
  more than 3,000 for the full top quintile. A simple rule, holding a stock until its rank falls below 60
  instead of selling as soon as it leaves the top 20, cut trading further and did as well or better in this backtest.
- **A business entity doesn't help.** A single-member LLC is ignored for income taxes, and a partnership passes
  the same short-term gains through to its owners. "Trader tax status" requires near-daily trading, and even
  then gains are taxed as ordinary income.

### TL;DR for trading firms
- **Structure comes first.** Most US hedge funds are partnerships, so gains pass through to investors with their
  short- or long-term character. Much of their money comes from tax-exempt investors (pensions, endowments) or
  from abroad, through offshore funds set up to shield those investors from US tax. Many quantitative long-short
  funds simply don't trade with taxes in mind.
- **Tax-aware long-short funds exist for taxable investors.** Managers such as AQR publish research on these
  strategies. They often aim to produce net capital losses that offset the investor's gains elsewhere.
- **Derivatives have their own rules.** Hedging with stock-index futures is taxed at a blend of 60% long-term and
  40% short-term rates, whatever the holding period. Rules on wash sales, straddles and constructive sales block
  the obvious workarounds. In the best-known abuse case, Renaissance Technologies used bank "basket options" to
  turn short-term gains into long-term ones. A US Senate report criticized this in 2014, and the firm's principals
  reportedly agreed in 2021 to pay the IRS up to about \$7 billion.
- **Implementation is industrial.** Funds borrow stocks through prime brokers, build portfolios with optimizers
  that weigh expected return against risk and trading costs, and execute with algorithms over hours or days.

---

### Tax reduction tools
- **Harvest losses, on both the long and short sides, and hold winners past a year.** Selling a loser realizes a
  loss that offsets gains. Waiting until a winner has been held more than a year turns a short-term gain into a
  long-term one, taxed at a lower rate. The "Tax-Loss Harvesting" tab walks through an example.
- **Sell the highest-cost shares first.** When you sell part of a position bought at different prices, you can
  choose which shares you sell. Selling the most expensive ones realizes the smallest gain (or the largest loss).
- **Swap in similar stocks to avoid wash sales.** A loss doesn't count if you buy the same stock within 30 days
  before or after the sale. Buying a similar stock, for example another company in the same industry, keeps
  the portfolio's exposure while the loss still counts.
- **Trade off expected return against tax cost when choosing trades.** A trade the model only mildly prefers may
  not be worth the tax it triggers. Tax-aware managers put the tax cost of each trade into the optimizer
  alongside the expected return and trading cost.

### Implementation
- **Shorting:** prime brokers lend the stocks and finance the positions. Funds must locate a stock to borrow before
  shorting it, and hard-to-borrow stocks charge high fees.
- **Portfolio construction:** funds use an optimizer with a risk model, a trading-cost model and limits on
  turnover, plus "no-trade" bands like the hold-until-rank-60 rule here, rather than simple quintile sorts.
- **Execution:** trades are split up and spread out by algorithm, often over more than a day, to limit their
  effect on prices.
- **Real costs:** Frazzini, Israel and Moskowitz studied AQR's own trades, over a trillion dollars of them. They
  found real trading costs were far below what spread estimates like the ones on the Compare Models page imply.
- **Disclosure:** large managers file 13F reports, which show only their long positions, once a quarter.
"""


def practice_page():
    st.title("Trading the Model in Practice")
    tab_bg, tab_401k, tab_tlh = st.tabs(["Background", "Trading in Your 401(k)", "Tax-Loss Harvesting"])

    with tab_bg:
        st.markdown(PRACTICE_BACKGROUND)

    with tab_401k:
        practice_401k_tab()

    with tab_tlh:
        tax_loss_harvesting_tab()


def practice_401k_tab():
    st.header("How Each Way of Trading the Model Would Have Done")
    practice = load_practice()
    if practice is None:
        st.error("practice.csv is missing from output_corrected/dashboard (run build_dashboard_bundle.py with the input data).")
        return
    models = list(practice['model'].unique())
    c1, c2 = st.columns(2)
    default = models.index('MLP-rank-256-128-64') if 'MLP-rank-256-128-64' in models else 0
    model = c1.selectbox("Model:", models, index=default, key='practice_model')
    start = c2.radio("Sample starts in:", [2000, 2005, 2010], index=1, horizontal=True, key='practice_start')

    st.markdown("""
    Inside a 401(k) or IRA, trades aren't taxed, so these are the returns the account itself would earn. Each pair of
    rows shows one way to trade the model, before and after trading costs.
    - **Long-short** rows can't be traded in a retirement account, which doesn't allow short selling. They are here
      for comparison.
    - **Most liquid 20%** means the 20% of stocks with the highest dollar trading volume each month, about 880 stocks.
      **Top quintile** holds the best-ranked fifth of those, about 176 stocks. **Top 20** holds just 20 of them.
    - **Trading costs:** the all-stocks rows pay half of each stock's estimated bid-ask spread (as on the Compare
      Models page). The rows that trade only liquid stocks pay the half-spread below, and their short legs pay the
      borrowing fee.
    - **Sharpe ratio** uses returns above the T-bill rate for long-only rows. A long-short return is already the
      difference between two returns, so its Sharpe ratio uses the return itself.
    - **Above market** compares long-only rows with an index fund holding all US stocks.
    - **Trades per year** counts every buy and sell, including the small trades that bring each holding back to
      equal weight.
    """)
    c1, c2 = st.columns(2)
    half_spread = c1.number_input("Half-spread for liquid stocks, each way (basis points)", 0.0, 100.0, 5.0, 1.0,
                                  key='practice_half_spread') / 10000
    borrow_fee = c2.number_input("Borrowing fee on short legs (% per year)", 0.0, 10.0, 0.25, 0.25,
                                 key='practice_borrow_fee') / 100

    ff = load_factors()
    rf = ff['RF'] / 100
    market = ff['mkt_excess'] / 100 + rf
    first_month = practice.index.min()
    rows = {}
    for approach in practice['approach'].unique():
        d = practice[(practice['model'] == model) & (practice['approach'] == approach) & (practice.index.year >= start)]
        long_only = approach.startswith('Long only')
        if approach == 'Long-short, all stocks':
            net = d['net_spread_estimate']
        else:
            net = d['gross'] - 2 * half_spread * d['turnover'] - (0.0 if long_only else borrow_fee / 12)
        trades = 12 * d['trades'][d.index != first_month].mean()     # nothing is traded in the first month
        for label, r in [(approach, d['gross']), (f"{approach}, after costs", net)]:
            s = return_stats(r, rf if long_only else None)
            rows[label] = {'Mean (%/yr)': s['Mean (%/yr)'], 'Sharpe ratio': s['Sharpe ratio'],
                           'Max drawdown (%)': s['Max drawdown (%)'],
                           'Above market (%/yr)': 1200 * (r - market.reindex(r.index)).mean() if long_only else np.nan,
                           'Trades/yr': trades}
    months = d.index
    s = return_stats(market.reindex(months), rf)
    rows['Index fund (stock market)'] = {'Mean (%/yr)': s['Mean (%/yr)'], 'Sharpe ratio': s['Sharpe ratio'],
                                         'Max drawdown (%)': s['Max drawdown (%)'], 'Above market (%/yr)': 0.0}
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format('{:.1f}', na_rep='').format('{:.2f}', subset=['Sharpe ratio'])
                 .format('{:,.0f}', subset=['Trades/yr'], na_rep=''), height=(len(table) + 1) * 35 + 3)
    st.caption("The sample ends in December 2019. The hold-until-rank-60 rule was picked once, not tuned, but with "
               "only 20 stocks, a few lucky picks can move these results a lot.")


def tax_loss_harvesting_tab():
    st.header("Tax-Loss Harvesting: A Small Example")
    st.markdown(r"""
    In January you put \$10,000 into each of three stocks. It is now December, eleven months later:

    | Stock | Bought for | Worth now | Gain or loss | The model says |
    |---|---|---|---|---|
    | A | \$10,000 | \$14,000 | +\$4,000 | sell |
    | B | \$10,000 | \$7,000 | -\$3,000 | keep |
    | C | \$10,000 | \$11,000 | +\$1,000 | sell |

    All three have been held for less than a year, so any gain or loss you realize now is **short-term**: taxed
    at your ordinary income tax rate. Gains held more than a year are **long-term** and taxed at a lower rate.
    """)
    c1, c2 = st.columns(2)
    st_rate = c1.number_input("Short-term rate (your income tax rate, %)", 0.0, 60.0, 37.0, 1.0, key='tlh_st') / 100
    lt_rate = c2.number_input("Long-term rate (%)", 0.0, 40.0, 20.0, 1.0, key='tlh_lt') / 100

    plans = {
        "Plan 1: Follow the model": ("Sell A and C.", 4000 + 1000, 0),
        "Plan 2: Also harvest B's loss": ("Sell A and C, and sell B and put the $7,000 into a similar stock.", 4000 + 1000 - 3000, 0),
        "Plan 3: Harvest, and wait a month on C": ("Sell A, harvest B, and sell C in February, after a year.", 4000 - 3000, 1000),
    }
    rows = []
    for name, (what, short_term, long_term) in plans.items():
        tax = st_rate * max(short_term, 0) + lt_rate * long_term
        rows.append({"Plan": name, "What you do": what, "Net short-term gain": f"${short_term:,.0f}",
                     "Long-term gain": f"${long_term:,.0f}", "Tax": f"${tax:,.0f}"})
    st.table(pd.DataFrame(rows).set_index("Plan"))
    base = st_rate * 5000
    st.markdown(rf"""
    **What happened**
    - **Plan 2** sells B, realizing its \$3,000 loss, which cancels \$3,000 of the gains on A and C. That saves
      \${base - st_rate * 2000:,.0f} in tax this year. You keep the same exposure by buying a *similar* stock,
      for example another company in B's industry.
    - **Why not just buy B back?** The **wash-sale rule** cancels the loss if you buy the same stock within 30 days
      before or after the sale. You could buy B again after 31 days, but you'd be out of it for a month.
    - **Plan 3** also holds C one more month, until it has been held more than a year. Its \$1,000 gain becomes
      long-term and is taxed at the lower rate, saving another \${st_rate * 1000 - lt_rate * 1000:,.0f}. The cost is
      holding a stock the model wants to sell for one more month.

    **The fine print**
    - Harvesting mostly *delays* tax. The replacement stock was bought for \$7,000, so if it later rises back to
      \$10,000, that \$3,000 gain is taxed when you sell it. The saving is real when that later gain is long-term
      (taxed at a lower rate), or when the stock is never sold, for example if it is donated or inherited.
    - If losses exceed gains, up to \$3,000 a year of the extra loss can reduce your other income. The rest
      carries forward to later years.
    - None of this matters inside a 401(k) or IRA, where gains and losses aren't taxed. It also adds work: a
      strategy that trades hundreds of times a year needs careful records of what was bought when, and at what
      price. Brokers report this on Form 1099-B.
    """)

import plotly.graph_objs as go

def make_toy_data(num_months: int = 24, num_firms: int = 10, num_signals: int = 3) -> tuple:
    """Generate a (firm, month) panel with random returns and signals."""
    # Create datetime objects instead of Period objects
    months = pd.date_range("2010-01-01", periods=num_months, freq="ME")
    firms = [f"Firm {i+1}" for i in range(num_firms)]
    
    # Create returns data with slightly correlated random returns
    np.random.seed(42)
    base_returns = np.random.normal(loc=0.01, scale=0.05, size=num_months)
    toy_rets = pd.DataFrame(
        [(firm, m, base_returns[i] + np.random.normal(scale=0.02))
         for i, m in enumerate(months) for firm in firms],
        columns=["firm_id", "yyyymm", "ret"],
    )
    
    # Create signal data with some relationship to returns
    signal_data = []
    for i, m in enumerate(months):
        for firm in firms:
            # Signals partially depend on past returns to simulate real-world correlations
            base_signal = base_returns[i-1] if i > 0 else 0
            row = [firm, m] + [
                base_signal + np.random.normal(loc=0.01, scale=0.03) 
                for _ in range(num_signals)
            ]
            signal_data.append(row)
    
    # Create DataFrame with proper column names
    toy_signals = pd.DataFrame(
        signal_data,
        columns=["firm_id", "yyyymm"] + [f"signal_{i+1}" for i in range(num_signals)]
    )
    
    return toy_rets, toy_signals

@st.cache_data
def prepare_combined_data(num_firms=1):
    """Prepare combined dataset with lagged signals."""
    toy_rets, toy_signals = make_toy_data(num_firms=num_firms)
    
    # Lag the signals by one month compare to returns 
    lag_signals = toy_signals.copy()
    
    # Important: Group by firm_id and shift each signal by one month within each firm
    lag_signals["yyyymm"] = lag_signals.groupby("firm_id")["yyyymm"].shift(-1)
    
    # Merge returns with lagged signals
    combined_toy_df = toy_rets.merge(lag_signals, on=["firm_id", "yyyymm"], how='left')
    
    # Format all dates as yyyy-mm, just for display purposes
    combined_toy_df["yyyymm"] = combined_toy_df["yyyymm"].dt.strftime("%Y-%m")
    toy_rets["yyyymm"] = toy_rets["yyyymm"].dt.strftime("%Y-%m")
    toy_signals["yyyymm"] = toy_signals["yyyymm"].dt.strftime("%Y-%m")
    
    return toy_rets, toy_signals, combined_toy_df

def create_training_timeline_plot(months_all, train_months, pred_month):
    """Create a timeline visualization for walk-forward validation."""
    # Convert any datetime-like objects to strings to avoid serialization issues
    months_all_str = [str(m)[:10] if hasattr(m, 'strftime') else str(m) for m in months_all]
    train_months_str = [str(m)[:10] if hasattr(m, 'strftime') else str(m) for m in train_months]
    pred_month_str = str(pred_month)[:10] if hasattr(pred_month, 'strftime') else str(pred_month)
    
    tl_df = pd.DataFrame({
        "yyyymm": months_all_str,
        "set": np.where(
            np.array(months_all_str) < train_months_str[0],
            "Past",
            np.where(
                np.isin(months_all_str, train_months_str),
                "Train",
                np.where(np.array(months_all_str) == pred_month_str, "Predict", "Future"),
            ),
        ),
    })
    
    # Use Plotly for interactive timeline
    fig = px.bar(
        tl_df,
        x="yyyymm",
        y=[1] * len(tl_df),
        color="set",
        color_discrete_map={
            "Train": "green",
            "Predict": "red",
            "Past": "lightgrey",
            "Future": "grey",
        },
        height=300,
        title="Walk-Forward Validation Timeline",
    )
    fig.update_layout(
        showlegend=True, 
        yaxis=dict(visible=False), 
        xaxis_title="Month",
        title_x=0.5
    )
    return fig

def background_page():
    st.title("Financial Asset Pricing Evaluation: A Step-by-Step Guide")
    
    # Prepare data
    num_firms = 1
    toy_rets, toy_signals, combined_toy_df = prepare_combined_data(num_firms=num_firms)
    
    # Create tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '(1) Overview',
        '(2) Returns Data', 
        '(3) Signals Data', 
        '(4) Combine Return and Signals Data',
        '(5) Training Timeline',
        '(6) Portfolio Construction',
        '(7) Full Code Example'
    ])
    
    with tab0:
        st.subheader("Overview")
        st.markdown("""
        This page provides an walkthrough of the steps of behind training and evaluating stock return prediction models. 
        
        Rather than evaluating how a model performs by looking at a summary stat from the model (like R-squared, MAE, etc.), we 
        evaluate how a model performs by looking at the returns you can achieve by trading on its predictions.
        
        In the other tabs on this page, we walk through the two main data sources--stock returns and predictive signals--and how to combine them,
        along with the steps to train models over time to get an accurate view of how the models would perform in practice.
        
        The basic ideas are:
        1. For firm $i$ in month $t$, we have a return $r_{i,t}$ and a set of "signals" $s_{i,t-1}$ measured in the previous month. 
        2. We can use these signals to predict the return in the current month, $r_{i,t}$. We will consider any functional model 
           $r_{i,t}=f(s_{i,t-1})$ that takes the signals as input and outputs a prediction of the return.
           - Before estimating the model, we can modify the signals to improve the model's performance, for example, by taking the 
             log of a signal or scaling it. There are many things to try: cleaning data, feature creation/engineering, feature selection.
           - Estimators for the function include: OLS, Ridge, Lasso, Gradient Boosting, Neural Networks, etc. This dashboard tries many of them.
        3. Given the predicted returns in a given month, we sort the firms into portfolios (e.g., Low, Middle, High) based on the predicted returns,
           and we determine the returns of each portfolio.
        4. We compute the "long-short" portfolio returns that month, which is simply the return of the High portfolio minus the return of the Low portfolio. 
           - In practice, this portfolio is what you would get if, at the beginning of the month, you: (1) Buy (or go "long") an equal amount of all the stocks in your High portfolio,
           and (2) short an equal amount of all the stocks in your Low portfolio.  
           - This portfolio is a bet that the High portfolio will outperform the Low portfolio.
           - This portfolio is zero-cost, meaning that you do not need to put up any capital to invest in it, because shorting all the stocks
           in the Low portfolio generates enough cash to buy the stocks in the High portfolio. 
        5. We repeat this over time, month by month, to get a time series of returns for the long-short portfolio.
        6. We then evaluate the performance of the long-short portfolio, for example, by looking at its excess returns, alphas, 
        Sharpe ratios, and other measures. 
           
        
        
        """)
    
    with tab1:
        st.subheader("Returns Data")
        st.markdown("""
        The first step is to  obtain stock returns from CRSP (Center for Research in Security Prices), 
        which posts its data on WRDS (Wharton Research Data Services).
        
        The data, in part, might look like the below, where each row contains data about a firm-month:
        - **Firm ID**: Unique identifier for the company (called "permno" in CRSP)
        - **Month**: The specific month of the return
        - **Return**: Return for the stock price for that month, including dividends but not delisting returns
        """)
        st.dataframe(toy_rets)
        
        # Visualization of returns distribution
        # fig_returns = px.box(toy_rets, x='firm_id', y='ret', 
        #                      title='Return Distribution Across Firms')
        # st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab2:
        st.subheader("Data on Signals: Predictive Features")
        st.markdown("""
        The second step is to obtain stock "signals", which are potential predictors of future returns.
        
        Pre-computed signals are available from [OpenAssetPricing](www.openassetpricing.com). 
        You can make your own signals as well, which can be very valuable, but this is outside the scope of this demo.
        
        Signals are features about the firm or economy measured during a given month:
        - **Important Note**: Signals can only be used to predict future returns, not the same month the signal is measured 
        - Each signal represents a different potential predictor of stock performance
        """)
        st.dataframe(toy_signals)
        
        # Correlation heatmap of signals
        # signal_cols = [col for col in toy_signals.columns if col.startswith('signal_')]
        # corr_matrix = toy_signals[signal_cols].corr()
        # fig_corr = px.imshow(corr_matrix, 
        #                       title='Correlation Between Signals',
        #                       color_continuous_scale='RdBu_r')
        # st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("Combined Data: Ready for Prediction")
        st.markdown("""
        In this view, we've shifted signals forward by one month. Each row now contains:
        1. Returns for a specific month
        2. Signals measured in the *previous* month
        
        This setup allows us to predict next month's returns using current month's signals.
        
        The code to get to this combination of the data sources with the lagged signals is _like_:
        ```python
        lag_signals = signals.copy()
        lag_signals["yyyymm"] = lag_signals.groupby("firm_id")["yyyymm"].shift(-1) # groupby is important
        rets_with_signals = returns.merge(lag_signals, on=["firm_id", "yyyymm"], how='left')
        ```
        """)
        st.dataframe(combined_toy_df)
        
        # Simple scatter plot to show relationship between a signal and returns
        # fig_signal_ret = px.scatter(
        #     combined_toy_df, 
        #     x='signal_1', 
        #     y='ret', 
        #     color='firm_id',
        #     title='Signal 1 vs Returns'
        # )
        # st.plotly_chart(fig_signal_ret, use_container_width=True)
    
    with tab4:
        st.subheader("Walk-Forward Validation: How Models Learn")
        st.markdown("""
        Walk-forward validation mimics real-world forecasting:
        1. Train a model using only data *up to* the current point, say May 31, 2010 (the green rows below). 
        1. Predict returns for the next month, June 2010, which is the red row below. Remember, the signals
        in the June 2010 row were measured in May 2010 because 
        we shifted the signals forward/down one row. So on May 31, 2010, we know the values in the signal
        columns for the red row. We put those red values into our model and this is our 
        prediction for the return for the stock during and through June 2010.
        2. We store the predicted returns for later use.
        3. Now, slide the training window forward month by month (either use expanding or rolling windows), 
        and repeat the steps above.
        
        So, as you move the slider below, **notice that the rows our model will use to predict the next 
        month are highlighted in green, and the rows we are predicting is highlighted in red.**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
                                                
            # Allow user to select prediction month and number of firms
            
            num_firms = st.slider(
                "Number of Firms to Display",
                min_value=1,
                max_value=4, step=1)           
            _, _, timeline_toy_df = prepare_combined_data(num_firms=num_firms)
            
            
            # Training window and prediction month selection
            months_all = timeline_toy_df['yyyymm'].unique()
            train_window = 5
            
            pred_idx = st.slider(
                "Select Prediction Month",
                min_value=train_window,
                max_value=len(months_all) - 1,
                value=train_window,
                step=1,
                help="Choose a month to see its training data and prediction window"
            )
            
            # make fake predictions to pull from
            # Set seed for reproducibility
            np.random.seed(42)
            fake_predictions = np.random.normal(0.01, 0.05, size=len(timeline_toy_df))
            
            pred_month = months_all[pred_idx]
            train_months = months_all[0 : pred_idx] # Expanding window, for rolling: pred_idx - train_window
            
            # Put the predictions in the prediction column for from the first training month until the prediction month
            timeline_toy_df['prediction'] = np.nan # reset to blank column
            
            # make timeline_toy_df['prediction'] = fake_predictions, and replace with nan for 
            # rows up to the training window and after the prediction month
            
            we_have_predictions = list(range(train_window*num_firms, pred_idx*num_firms+num_firms))
            timeline_toy_df['prediction'] = np.where(
                (timeline_toy_df.index.isin(we_have_predictions)), 
                fake_predictions,
                np.nan
            )
            
            def highlight_training_and_pred_rows(df, train_months, pred_month):
                """
                Create a styling function to highlight training and prediction rows
                
                Args:
                    df (pd.DataFrame): The full dataframe
                    train_months (array-like): Months to be used for training
                    pred_month (str or Period): Month being predicted
                
                Returns:
                    Styled dataframe with training and prediction rows highlighted
                """
                # Create masks for training and prediction rows
                training_mask = df['yyyymm'].isin(train_months)
                pred_mask = df['yyyymm'] == pred_month
                
                # Define a style function
                def style_special_rows(row):
                    # Highlight training rows with light green, prediction month with light red
                    if row.name in df[training_mask].index:
                        return ['background-color: lightgreen' for _ in row]
                    elif row.name in df[pred_mask].index:
                        return ['background-color: lightsalmon' for _ in row]
                    return ['' for _ in row]
                
                return df.style.apply(style_special_rows, axis=1)

            # In the tab4 section, replace the st.dataframe() with:
            height = 36*(1+len(timeline_toy_df)) - 10
            
            st.dataframe(
                highlight_training_and_pred_rows(timeline_toy_df, train_months, pred_month),
                use_container_width=False,
                height=height,
            )
            
        with col2:
            
            st.markdown("""
        **The code for this is like** (this is highly simplified):
        
        ```python
        # which rows are we using up to this point to train?
        training_rows = df_signals[df_signals['yyyymm'].isin(train_months)]
        prediction_rows = df_signals[df_signals['yyyymm'] == pred_month]
        
        # train the model on the training data and then predict the next month
        model.fit(df_signals[training_rows], df_returns[training_rows])
        predictions = model.predict(df_signals[prediction_rows])
        
        # now we store the predictions for later use...
        ```
        
        
        
        """)
    
    with tab5:
        
        
        col1, col2 = st.columns(2)
        
        with col1:
        
            st.subheader("Portfolio Construction: From Predictions to Portfolio Positions")
            
            st.markdown("""
            After the last step, we have a dataset with return predictions for the next month as though we had done them in real time.
            
            We will now use these predictions to construct portfolios where we rank stocks into bins based on how high we think their returns will be next month.
            It is most common to sort stocks into 5 or 10 bins, but you can use any number of bins you like. In this example, we will sort into 2 portfolios.
                                
            Next, we:
            1. Sort stocks by predicted return each month. In the example, we sort into 2 portfolios. The output of this is to the right.

            ```python
            def port_sort(x, nport):
                return np.ceil(x.rank(method="first") * nport / len(x)).astype(int)

            predictions['portfolio_assignment'] = predictions.groupby('yyyymm')['prediction'].transform(lambda x: port_sort(x, 2))
            ```

            **What about ties?** `method="first"` breaks ties by row order, which keeps this example simple but is arbitrary.
            Real models do produce ties: a Lasso can set every coefficient to zero and give all stocks the same prediction.
            The results on this site split tied stocks evenly across the portfolios they span, and the
            Full Code Example shows how.

            2. Then we compute the monthly portfolio returns. You can see an example of this to the right. Notice that we also compute the 
            long minus short portfolio, which is the difference between the returns of the highest and lowest ranked portfolios.
            
            ```python
            portfolios = (predictions
                          .groupby(['yyyymm', 'portfolio_assignment'])
                          ['ret'].mean())
            
            # reshape to wide, and compute the long minus short portfolio
            ports_wide = portfolios.unstack(level=-1)
            ports_wide.columns = [f"Port{col}" for col in ports_wide.columns]
            ports_wide['LongShort'] = ports_wide['Port2'] - ports_wide['Port1']
            ```
            
            3. Finally, we can subject those returns (focusing on the long-short portfolio) to any kind of analysis we might want to apply to asset returns. 
            The "Compare Models" and "Dig into a Model" pages do this.
            
            - Calculate statistics: Sharpe ratios, drawdowns, turnover, etc.
            - Plot cumulative returns, rolling returns, etc.
            - Compute alphas and factor loadings using regression analysis
            """)
            
        
        with col2:
            
            _, _, portfolio_toy_df = prepare_combined_data(num_firms=2)
            
            # now, let's add predictions to this
            np.random.seed(42)
            fake_predictions = np.random.normal(0.01, 0.05, size=len(portfolio_toy_df))
            
            we_have_predictions = list(range(train_window*num_firms-1, len(portfolio_toy_df)))
            portfolio_toy_df['prediction'] = np.where(
                (portfolio_toy_df.index.isin(we_have_predictions)), 
                fake_predictions,
                np.nan
            )
            
            # reduce to times we have predictions
            portfolio_toy_df.dropna(subset=['prediction'], inplace=True)
            
            # sort firms into portfolios monthly
            def port_sort(x, nport):
                return np.ceil(x.rank(method="first") * nport / len(x)).astype(int)

            portfolio_toy_df['portfolio_assignment'] = portfolio_toy_df[['yyyymm', 'firm_id', 'prediction']].groupby('yyyymm')['prediction'].transform(lambda x: port_sort(x, 2)) 
            portfolio_toy_df = portfolio_toy_df[['yyyymm', 'firm_id', 'ret', 'prediction', 'portfolio_assignment']]
            
            st.markdown("### Predictions sorted into portfolios")
            st.dataframe(portfolio_toy_df, use_container_width=False, )
            
            port_rets = portfolio_toy_df.groupby(['yyyymm', 'portfolio_assignment'])['ret'].mean()
            port_rets = port_rets.unstack(level=-1)
            port_rets.columns = [f"Portfolio{col}" for col in port_rets.columns]
            port_rets['Long-Short'] = port_rets['Portfolio2'] - port_rets['Portfolio1']
            
            st.markdown("### Portfolio returns")
            st.dataframe(port_rets, use_container_width=False)
            
    with tab6:
        
        st.header("Full Code Example")
        st.markdown("""
        Below is the full, self-contained Jupyter Notebook (available in the repo). It takes the ideas in this
        background section and implements them on real data. It is written to be read and changed. The results on the
        other pages come from the `stockpred` package in the repo, which follows the same steps.
        Features that make working with such large data easier:
        - A `DEV` switch that shrinks the data so you can modify and explore quickly
        - Several models in one loop: OLS, Ridge, Lasso, their tuned versions (RidgeCV, LassoCV), and a neural net
        - Every fit is saved as soon as it finishes, and finished fits are not retrained
        - Careful data handling: same-month cross-sectional imputation, winsorizing and scaling that only learn from
          the training years, tuning that only validates on later years, and checks of the one-month timing
        - Portfolios with a fair treatment of ties, and HAC (Newey-West) t-statistics

        **The notebook also shows you how to design and train models to produce your own
        predictive signals: add one line to `model_dict`.**

        A full run takes hours and about 30 GB of RAM. With `DEV = True` it runs in a few minutes. For more details, read the [README.md file in the repo](https://github.com/donbowen/StockPredictionAndEval).
        
        ---
        """)
        
        import nbformat
        from pathlib import Path
        
        # Path to your notebook file
        notebook_path = Path("stock_prediction_and_eval.ipynb")  # Update with actual path
        
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        st.markdown(f"## Outline of Notebook")
        
        # Produce an outline of ## sections in the notebook, and link to them 
        lines = []
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                if cell.source.startswith("## "):
                    # Extract the section title and create a link
                    line_one = cell.source.split('\n')[0]
                    section_title = line_one[3:].strip()
                    lines.append(f"1. [{section_title}](#{section_title.replace(' ', '-').replace('(','').replace(')','').lower()})")
                    
        st.markdown("\n".join(lines))
        
        st.markdown(f"---")
        
        # Display each cell of the notebook
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown':
                st.markdown(cell.source)
            elif cell.cell_type == 'code':
                # with st.expander(f"Code Cell {i+1}"):
                st.code(cell.source, language='python')

        

# Note: This function should be called in your main Streamlit app
# background_page()

# Main app execution
def main():
    page = sidebar_navigation()
    
    if page == "Introduction":
        intro_page()
    elif page == "Background & Methods":
        background_page()
    elif page == "Compare Models":
        compare_models_page()
    elif page == "Dig into a Model":
        model_details_page()
    elif page == "Trading the Model in Practice":
        practice_page()

if __name__ == "__main__":
    main()

# Add footer with information
st.markdown("---")
st.markdown("By Prof. Donald Bowen, Lehigh University (2025). Inspired by several student projects, OpenAssetPricing, and Assaying Anomalies.")
