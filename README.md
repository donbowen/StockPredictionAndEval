# Stock Predictor Project

![GitHub](https://img.shields.io/github/license/yourusername/StockPredictorProject_PROF)

## Overview

Stock Predictor Project is a comprehensive machine learning framework for predicting stock returns and evaluating trading strategies. This project combines state-of-the-art machine learning techniques with traditional financial analysis to create a powerful tool for investment decision-making.

## Motivation

Financial markets generate vast amounts of data, but extracting actionable insights from this data remains challenging. This project addresses this challenge by:

1. **Applying machine learning to financial prediction**: Using various ML algorithms to identify patterns in stock returns that traditional approaches might miss
2. **Creating a systematic evaluation framework**: Moving beyond simple backtest results to apply rigorous statistical tests of performance
3. **Making financial research reproducible**: Providing a complete pipeline from data acquisition to model evaluation

## Key Features

- **State-of-the-art input data**: CRSP for stock returns, OpenAssetPricing for predictive signals
- **Prediction models**: 
  - Included here: OLS, Ridge, Lasso, RidgeCV, LassoCV, PyTorch neural nets (MLPs), and histogram gradient boosting (HGB)
  - Can easily be expanded to any other model that can be integrated into `sklearn` pipelines
- **Out-of-sample validation**: Rolling training and testing to prevent look-ahead bias
- **Portfolio construction**: Creation of long-short portfolios based on model predictions
- **Performance analysis**: Sharpe ratios, factor regression to estimate alpha, and portfolio turnover metrics
- **Interactive dashboard**: Streamlit web app for comparing performance across many models and understanding each

## Repository Structure

```
StockPredictorProject/
├── .gitignore                  # Git ignore file
├── .streamlit/                 # Streamlit configuration
│   └── config.toml             # Streamlit theme configuration
├── input_data/                 # Input data (not tracked by git)
│   └── crsp_data.csv           # CRSP return data
├── output_corrected/           # Results of the package runs (what the dashboard shows)
│   ├── linear_v1/              # OLS, Ridge(CV), Lasso(CV): config, per-fit metadata, results tables
│   ├── nn_v1/                  # MLP ensembles
│   ├── hgb_v1/                 # Histogram gradient boosting, and OLS on the rank target
│   └── dashboard/              # Files the dashboard reads (built by build_dashboard_bundle.py)
├── stockpred/                  # Tested package: data, preprocessing, tuning, portfolios, evaluation, MLPs
├── tests/                      # Unit tests (python -m pytest tests)
├── README.md                   # This file
├── stock_prediction_and_eval.ipynb  # Self-contained walkthrough notebook for students (same steps as the package)
├── run_linear_backtest.py      # Reproduce the linear-model results
├── run_nn_screen.py            # Neural-net design screen
├── run_nn_backtest.py          # Reproduce the neural-net results
├── run_hgb_backtest.py         # Reproduce the gradient boosting results
├── build_dashboard_bundle.py   # Build the dashboard's input files
└── streamlit_app.py            # Streamlit dashboard
```

## Dependencies

This project requires the following Python packages:

- **Core Data Science**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn, statsmodels, PyTorch
- **Finance**: pandas-datareader, openassetpricing
- **Visualization**: matplotlib, seaborn
- **Dashboard**: streamlit
- **Utils**: tqdm, gdown, ydata-profiling

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StockPredictorProject_PROF.git
   cd StockPredictorProject_PROF
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the data:
   - If you don't have the necessary data files, the code will attempt to download them from Google Drive when executed.
   - Exception: CRSP stock return data requires a WRDS login and I will not distribute it.


## Usage

### Running the Analysis

To walk through the full analysis, open and run the Jupyter notebook (`stock_prediction_and_eval.ipynb`). It is
self-contained and:
1. Loads and preprocesses the data (signals measured in month *t* predict month *t+1*)
2. Trains OLS, Ridge, RidgeCV, Lasso, LassoCV and a neural net, expanding the training window year by year
3. Generates out-of-sample predictions (saved per model and year)
4. Creates and evaluates portfolios (tie-aware quintiles, HAC alphas, turnover)
5. Saves results to `output_notebook/`

Set `DEV = True` in its first code cell for a quick end-to-end check on a small sample. A full run takes hours
and a lot of RAM.

### Package runs (v2 pipeline)

The notebook's code mirrors the tested `stockpred/` package, which produced the results in `output_corrected/` that the
dashboard shows (the first version of the pipeline had timing and leakage problems and has been removed).

Reproduce everything (smoke test, OLS, Ridge/RidgeCV/Lasso/LassoCV, evaluation) with one command:

```bash
python run_linear_backtest.py --data-dir input_data --run-id linear_v1
```

Outputs go to `output_corrected/<run-id>/` (config, environment, data manifest, per-fit metadata, predictions,
fitted pipelines and `results/` tables + `report.md`). Fits are resumed only if their config/data/code hashes match.
Timing: signals and predictor factors measured in month *t* predict returns in month *t+1*; 2020 is an unused
buffer year and 2021+ an untouched holdout (never loaded). Tests: `python -m pytest tests`.

### Neural networks and the dashboard data (v2 pipeline)

```bash
python run_nn_screen.py --year 2005      # architecture / target screen (no test data touched)
python run_nn_backtest.py                # three 5-seed MLP ensembles, 2000-2019, then evaluation (needs PyTorch)
python run_hgb_backtest.py               # gradient boosting on three targets and OLS on the rank target, then evaluation (about 7.5 hours)
python build_dashboard_bundle.py --data-dir input_data   # writes output_corrected/dashboard/ for the Streamlit app (input data needed for the trading cost tables)
```

The dashboard's model-pipeline and variable-importance tabs load the per-year saved pipelines in
`output_corrected/<run>/models/`, which are stored in git. Predictions are not. (To browse a notebook run instead, set `RESULTS_DIR` in
`streamlit_app.py` to its folder and write a `models_index.json` like the one in `output_corrected/dashboard/`.)

### Running the Dashboard

To launch the interactive dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard allows you to:
- Explore model performance through interactive visualizations
- Compare different models side-by-side
- View portfolio returns, factor exposures, and other statistics
- Analyze the performance across different time periods

## Methodology

1. **Data Preparation**:
   - Load stock return data and asset pricing signals
   - Handle missing values and outliers with cross-sectional imputation
   - Standardize features for reliable model training

2. **Model Training**:
   - Implement rolling window approach to simulate real-world prediction
   - Train models on historical data up to year t-1
   - Generate predictions for year t
   - Repeat for each year in the sample period

3. **Portfolio Construction**:
   - Sort stocks into quintiles based on model predictions
   - Form zero-cost long-short portfolios (long top quintile, short bottom quintile)
   - Calculate portfolio returns and performance metrics

4. **Performance Evaluation**:
   - Factor regression analysis to assess risk-adjusted returns
   - Portfolio turnover analysis to evaluate trading costs
   - Sharpe ratio calculation for risk-adjusted performance comparison

## Acknowledgments

- Special thanks to all contributors to the Open Asset Pricing project
- Financial data provided by CRSP (Center for Research in Security Prices)
- Factor data provided by Kenneth French's data library
