# later, not now: add more models (OLS, MLP) to ipynb data that feeds into this
# I'd love to add in lazy prices!

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import datetime
import statsmodels.formula.api as smf
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin

class CrossSectionalMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_impute=None):
        self.features_to_impute = features_to_impute 

    def fit(self, X, y=None):
        # Nothing to fit — this imputer calculates means on the fly
        return self
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []

    def transform(self, X):
        X = X.copy()
        
        # Make sure it's a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CrossSectionalMeanImputer only works on pandas DataFrames")
        
        # Check if index is a MultiIndex with 'date' as one of the levels
        if not isinstance(X.index, pd.MultiIndex) or 'date' not in X.index.names:
            try:
                # add date to the index if it's not already there
                X = X.set_index(['permno','date'])
            except:
                raise ValueError("Input DataFrame must have a MultiIndex with 'date' as one of the levels")
        
        # Create a year variable
        X['_year'] = pd.to_datetime(X.index.get_level_values('date').year.values)
        
        # For each numeric column, fill NaNs with cross-sectional mean within each year
        # numeric_cols = X.select_dtypes(include=[np.number]).columns.drop('_year', errors='ignore')
        
        if self.features_to_impute is not None:
            cols_to_impute = [col for col in self.features_to_impute if col in X.columns]
        else:
            cols_to_impute = X.select_dtypes(include=[np.number]).columns.drop('_year', errors='ignore')
        
        for col in cols_to_impute:
            col_global_mean = X[col].mean()
            
            # If the global mean itself is NaN, fallback to 0
            if pd.isna(col_global_mean):
                col_global_mean = 0.0
            
            def safe_fill(x):
                if x.isnull().all():
                    return x.fillna(col_global_mean)  # use fallback global mean (guaranteed non-NaN now)
                else:
                    return x.fillna(x.mean())  # cross-sectional mean
                
            X[col] = X.groupby('_year')[col].transform(safe_fill)
            
        X = X.drop(columns=['_year'])
        
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.0025, upper=0.9975):
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        # Store quantiles for each column
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower)
        self.upper_bounds_ = X_df.quantile(self.upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[col],
                                       upper=self.upper_bounds_[col])
        return X_df.values  # return as numpy array for sklearn compatibility
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []



# Set page configuration
st.set_page_config(
    page_title="ML Stock Prediction Evaluation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading functions
@st.cache_data
def load_data():
    """Load portfolio data from CSV files"""
    returns_wide_df = pd.read_csv('output_portfolios/returns_wide.csv')
    port_stats_tall_df = pd.read_csv('output_portfolios/port_stats_tall.csv')
    
    # Convert date to datetime
    returns_wide_df['date'] = pd.to_datetime(returns_wide_df['date'])
    port_stats_tall_df['date'] = pd.to_datetime(port_stats_tall_df['date'])
    
    # Set date as index for returns_wide_df
    returns_wide_df = returns_wide_df.set_index('date')
    
    # turnover data: model,bin,date,% buy,% sell,% hold
    turnover_df = pd.read_csv('output_portfolios/turnover.csv')
    turnover_df['date'] = pd.to_datetime(turnover_df['date'])
    turnover_df = turnover_df.set_index('date')
    
    return returns_wide_df, port_stats_tall_df, turnover_df

@st.cache_data
def load_factors():
    """Load Fama-French factors for regression analysis"""
    start = '1964-01-01'
    start_date = datetime.strptime(start, '%Y-%m-%d')
    
    # Get FF5 and momentum
    ff_5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date)[0]
    ff_mom = pdr.get_data_famafrench('F-F_Momentum_Factor', start=start_date)[0]
    ff_mom.columns = ['Mom']  # rename
    
    # combine
    ff_factors = pd.merge(ff_5, ff_mom, left_index=True, right_index=True, how='outer')
    ff_factors = ff_factors.reset_index().rename(columns={"Mkt-RF": "mkt_excess", "Date": "date"})
    ff_factors["date"] = ff_factors["date"].dt.to_timestamp()
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
    
    # Convert to long format for Plotly Express
    plot_df_long = plot_df.reset_index().melt(
        id_vars='index',
        var_name='Model',
        value_name='Return'
    )
        
    # Create the figure with Plotly Express
    fig = px.line(
        plot_df_long, 
        x='index', 
        y='Return', 
        color='Model',
        title='Comparison of Long-Short Portfolio Returns Across Models',
        color_discrete_sequence=px.colors.sequential.Viridis,
        height=600
    )
        
    # Improve layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='',
        legend_title='Model',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
        
    # If max in the plot_df is greater than 50, log scale the y-axis
    if plot_df.max().max() > 50:
        fig.update_yaxes(type="log")
        
        # Format y-axis ticks to be more readable in log scale
        fig.update_yaxes(
            tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000"]
        )
            
    return fig, None

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
    - View turnover analysis for each portfolio
    - Compare Sharpe ratios for different models
    
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
    - **Compare Models**: Compare performance across multiple models
    - **Model Details**: Deep dive into a specific model
    
    ## Getting Started:
    
    Select a page from the sidebar to begin exploring the data.
    
    ## Inspiration
    
    - [Neural Network Dashboard](https://lehigh-asset-pricing.streamlit.app/) by Joseph Carruth, Jay Geneve, Michael Jamesley, and Evan Trock
    - [Lazy Prices Replication](https://lazypricesreplication.streamlit.app/) by Hannah Gordon, Akanksha Gavade, Marti Figueres, and Henry Piotrowski
    - [OpenAssetPricing](www.openassetpricing.com) by Andrew Chen and Tom Zimmermann
    - [Assaying Anomalies](https://sites.psu.edu/assayinganomalies/) by Robert Novy-Marx  and Mihail Velikov
    - [FIN377 textbook chapter on Assaying Anomalies](https://ledatascifi.github.io/ledatascifi-2025/content/05/05d_AssetPricingAnomalyTable1.html)
    - [FIN377 textbook chapter on OpenAssetPricing](https://ledatascifi.github.io/ledatascifi-2025/content/05/05e_OpenAP_anomaly_plot.html)
    

    
    """)
    
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
        st.error("Please ensure the portfolio data files exist in the 'output_portfolios' directory.")

# Compare models page content
def compare_models_page():
    st.title("Compare Model Performance")
    
    try:
        # Load data
        returns_wide_df, _, _ = load_data()
        
        # Extract model names
        model_names = list(set([col.split('_')[1] for col in returns_wide_df.columns if 'ret_' in col]))
        
        # Model selection
        selected_models = st.multiselect(
            "Select models to compare:",
            options=model_names,
            default=model_names[:min(5, len(model_names))]  # Default to first 5 or fewer
        )
        
        if not selected_models:
            st.warning("Please select at least one model to continue.")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Cumulative Returns Comparison", "Risk-Adjusted Returns via Regression Analysis", "Additional Metrics", "About the Models"])
        
        # Tab 1: Cumulative Returns Comparison
        with tab1:
            st.header("Long-Short Portfolio Returns Comparison")
            
            st.markdown("""
            This chart compares the cumulative returns of Long-Short portfolios across the selected models.
            The Long-Short strategy involves buying the highest-ranked stocks (Portfolio 5) and shorting 
            the lowest-ranked stocks (Portfolio 1) for each model. If a model has a high t-stat, then the prediction model 
            is doing a good job of sorting stocks to buy and short compared to the factors in the regression model.
            
            These alphas are in percent and are monthly. To annualize, multiply by 12 (close and good enough).
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
                        
            metrics = {}
            for model in selected_models:
                # Find LS column
                ls_col = [col for col in returns_wide_df.columns if f'ret_{model}_binLS' in col]
                if ls_col:
                    ls_data = returns_wide_df[ls_col[0]]
                    metrics[model] = {
                        'Mean Monthly Return (%)': ls_data.mean() * 100,
                        'Std Dev (%)': ls_data.std() * 100,
                        'Sharpe Ratio (Annualized)': (ls_data.mean() / ls_data.std()) * np.sqrt(12),
                        'Max Drawdown (%)': (1 - (1+ls_data).cumprod() / (1+ls_data).cumprod().cummax()).max() * 100
                    }
            
            if metrics:
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df.style.format('{:.4f}'))
            else:
                st.error(error_msg)
                
            st.header("Turnover Analysis")
            
            st.markdown("""
            This table shows the average turnover for each portfolio bin across the selected models.
            Turnover is defined as the percentage of stocks bought, sold, or held in each portfolio.
            """)
            
            _, _, turnover_df = load_data()
            turnover_df = turnover_df[turnover_df['model'].isin(selected_models)]
            turnover_df = turnover_df.set_index(['model', 'bin'])[['% buy', '% sell', '% hold']].groupby(level=['model','bin']).mean()
            # turnover_df.columns = [c.title() for c in turnover_df.columns]  
            
            # user decides if they want to see % buy, % sell, or % hold
            st.selectbox("Select turnover type to display:", options=[ '% hold', '% buy', '% sell'], index=0, key='turnover_type')
            turnover_type = st.session_state['turnover_type']
            
            show_turnover = turnover_df[turnover_type].unstack(level=-1)
            
            # turnover_df = turnover_df.unstack(level=-1)
            
            st.dataframe(show_turnover.style.format("{:.1%}"), use_container_width=False)
            
        with tab4:
            st.header("About the Models")
            
            st.markdown("""
            This section provides a brief overview of the models used in this analysis.
            
            
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
            
            Histogram Gradient Boosting (HGB):
            - HGB is a type of ensemble learning method that builds multiple decision trees in a sequential manner.
            - It is particularly effective for large datasets and can model complex relationships that are non-linear (e.g., interactions between features or non-linear relationships).
            
            MLP (Multi-Layer Perceptron) aka Neural Network:
            - MLP is a type of neural network that consists of multiple layers of interconnected nodes (neurons).
            - A punchy one-liner is that it is a "black box" model that can learn complex relationships in the data.
            - NNs are powerful and flexible, but they require careful tuning of hyperparameters and can be prone to overfitting.
            - Some projects have found them to work well in finance, for example, [Gu, Kelly & Xiu (RFS 2020)](https://academic.oup.com/rfs/article/33/5/2223/5758276/) and [OpenAssetPricing](https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/ML_portfolio_example.ipynb).
            
    
            """)
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please make sure the portfolio data files exist in the 'output_portfolios' directory.")

# Model details page content
def model_details_page():
    st.title("Detailed Model Analysis")
    
    # try:
    # Load data
    returns_wide_df, port_stats_tall_df, _ = load_data()
    
    # Extract model names
    model_names = list(set([col.split('_')[1] for col in returns_wide_df.columns if 'ret_' in col]))
    
    # Make the first model the Ridge model
    model_names = ['Ridge'] + sorted([m for m in model_names if m != 'Ridge'])
    
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
            turnover_model_df = turnover_model_df[['bin', '% buy', '% sell', '% hold']].groupby('bin').mean()
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
        
        model_path = f"models/{selected_model}.joblib"
        pipeline = None
        try:
            import joblib
            pipeline = joblib.load(model_path)
            st.write(pipeline)
            
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
                components.html(
                    html_repr,
                    width=1000,
                    height=500,     # adjust as needed
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
        # todo lasso broken, hbgr shows nada
        
        st.header("Which variables are important in this model?")
        
        st.markdown("_Note: This tab is a work in progress. Some models will not work._")
        
        top_n = st.slider("Top N features", 5, 50, 20, step=5)
        
        # viz_style = st.selectbox("Feature viz style", ["Horizontal bar", "SHAP"])
        viz_style = "Horizontal bar"
        
        # load the model/model_name joblib pipeline
        
        model_path = f"models/{selected_model}.joblib"
        
        pipeline = None
        try:
            import joblib
            pipeline = joblib.load(model_path)
            st.write(pipeline)            
        except FileNotFoundError:
            st.error(f"Model pipeline file not found: {model_path}")
        except Exception as e:
            st.error(f"An error occurred while loading the model pipeline: {e}")
            
        if pipeline:
            # Interpretability
            estimator = pipeline.named_steps[list(pipeline.named_steps)[-1]]
            coef_attr = getattr(estimator, "coef_", None)
            imp_attr  = getattr(estimator, "feature_importances_", None)

            if coef_attr is not None:
                st.subheader("🔢 Top Coefficients")
                # get feature names from preprocessing
                feat_names = pipeline[:-1].get_feature_names_out()
                coefs = pd.Series(coef_attr.flatten(), index=feat_names)
                                
                # sort the coefs by their absolute value 
                coefs = coefs.sort_values(key=lambda x: x.abs(), ascending=False)
                top = coefs.head(top_n)
                
                # Bar chart
                fig3, ax3 = plt.subplots()
                if viz_style=="Horizontal bar":
                    import plotly.express as px
                    
                    # Create Plotly horizontal bar chart
                    fig3 = px.bar(
                        x=top.values,
                        y=top.index,
                        orientation='h',
                        labels={'x': 'Coefficient Value', 'y': 'Feature'},
                        title="Coefficient Magnitude",
                        height=max(400, len(top)*25)  # Dynamic height based on number of features
                    )
                    
                    # Improve layout for readability
                    fig3.update_layout(
                        yaxis={'categoryorder': 'total ascending'},  # Sort bars
                        hoverlabel=dict(bgcolor="white", font_size=12),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    # Display the plotly chart in Streamlit
                    st.plotly_chart(fig3, use_container_width=True)
                # else:
                #     top.plot.bar(ax=ax3)
                #     ax3.set_title("Coefficient magnitude")
                #     st.pyplot(fig3)

            elif imp_attr is not None:
                
                st.subheader("🌳 Feature Importances")
                feat_names = pipeline[:-1].get_feature_names_out()
                imps = pd.Series(imp_attr, index=feat_names).nlargest(top_n)
                st.bar_chart(imps)

            #     if viz_style=="SHAP" and HAS_SHAP:
            #         st.subheader("SHAP Summary")
            #         explainer = shap.Explainer(estimator, pipeline[:-1].transform(X_val))
            #         shap_vals = explainer(pipeline[:-1].transform(X_val))
            #         fig4 = shap.plots.beeswarm(shap_vals, max_display=top_n, show=False)
            #         st.pyplot(fig4)

                    
    # except Exception as e:
    #     st.error(f"An error occurred: {e}")
    #     st.error("Please make sure the portfolio data files exist in the 'output_portfolios' directory.")

# Navigation sidebar
def sidebar_navigation():
    st.sidebar.title("Pages")
    page = st.sidebar.radio(
        "Go to:",
        ["Introduction", "Background & Methods", "Compare Models", "Dig into a Model"]
    )
    return page

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '(1) Returns Data', 
        '(2) Signals Data', 
        '(3) Combine Them',
        '(4) Training Timeline',
        '(5) Portfolio Construction',
        '(6) Full Code Example'
    ])
    
    with tab1:
        st.subheader("Returns Data")
        st.markdown("""
        The first step is to  obtain stock returns from CRSP (Center for Research in Security Prices), 
        which posts its data on WRDS (Wharton Research Data Services).
        
        The data, in part, might look like the below, where each row contains data about a firm-month:
        - **Firm ID**: Unique identifier for the company (called "permno" in CRSP)
        - **Month**: The specific month of the return
        - **Return**: Return for the stock price for that month, including dividends and delisting returns
        """)
        st.dataframe(toy_rets)
        
        # Visualization of returns distribution
        # fig_returns = px.box(toy_rets, x='firm_id', y='ret', 
        #                      title='Return Distribution Across Firms')
        # st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab2:
        st.subheader("Data on Signals: Predictive Features")
        st.markdown("""
        The second step is to obtain stock signals, which are potential predictors of future returns.
        
        Pre-computed signals are available from [OpenAssetPricing](www.openassetpricing.com). 
        You can make your own signals as well, but this is outside the scope of this demo.
        
        Signals are features about the firm or economy measured during the month:
        - **Important Note**: Signals can only be used to predict *next month's* returns
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
        
        The code for this is _like_:
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
        1. Only use data *up to* the current point to train a model, say May 31, 2010. (The green rows below.)
        1. The signals in the June 2010 row were measured in May 2010. Remember, 
        we shifted the signals forward/down one row. So on May 31, 2010, know the values in the signal
        columns for the red row. We put those red values into our model and this is our 
        prediction for the return for the stock during and through June 2010.
        2. We store the predictions for later use.
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
        
            st.subheader("Portfolio Construction: From Predictions to Positions")
            
            st.markdown("""
            After the last step, we have a dataset with predictions for the next month.
                                
            Next, we:
            1. Sort stocks by predicted return each month. In the example, we sort into 2 portfolios. The output of this is to the right.
            
            ```python
            def port_sort(x, nport):
                return np.ceil(x.rank(method="min") * nport / (len(x) + 1)).astype(int)
            
            predictions['portfolio_assignment'] = predictions.groupby('timevar')['prediction'].transform(lambda x: port_sort(x, 2)) 
            ```
            
            2. Then we compute the monthly portfolio returns. You can see an example of this to the right.
            ```python
            portfolios = (predictions
                          .groupby(['yyyymm', 'portfolio_assignment'])
                          ['ret'].mean())
            
            # reshape to wide, and compute the long minus short portfolio
            ports_wide = portfolios.unstack(level=-1)
            ports_wide.columns = [f"Port{col}" for col in ports_wide.columns]
            ports_wide['LongShort'] = ports_wide['Port'] - ports_wide['Port']
            ```
            
            3. Finally, we can subject those returns to any kind of analysis we might want to apply to asset returns. 
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
                return np.ceil(x.rank(method="min") * nport / (len(x) + 1)).astype(int)
            
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
        Below is the full Jupyter Notebook (available in the repo) that is behind creating and evaluating the models 
        in the other pages on the site. This takes the ideas in this background section and 
        implements them on real data. Their are extra features in the code to make working with such large data easier:
        - Reducing the data size to make it easier to modify and explore
        - Ability to train many models
        - Models only train once; when they are in the output dataset, they will not be retrained
        - More sophisticated data handling: Winsorizing and CrossSectionalImputation
        - It is set up for executing a Neural Net model (but I did not run it!)
                
        **The notebook also shows you how to design and train models to produce your own
        predictive signals.**
        
        Running it takes a day or so and a decent amount of RAM. For more details, read the [README.md file in the repo](https://github.com/donbowen/StockPredictionAndEval).
        
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

if __name__ == "__main__":
    main()

# Add footer with information
st.markdown("---")
st.markdown("By Prof. Donald Bowen, Lehigh University (2025). Inspired by several student projects, OpenAssetPricing, and Assaying Anomalies.")
