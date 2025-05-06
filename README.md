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
  - Included here: Ridge, Lasso, RidgeCV, LassoCV, and Histogram-based Gradient Boosting
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
├── models/                     # Saved ML model files
├── output_portfolios/          # Generated portfolio returns and statistics
│   ├── port_stats_tall.csv     # Portfolio statistics in tall format
│   └── returns_wide.csv        # Portfolio returns in wide format
├── predictions/                # Model predictions
│   └── prediction_output.csv   # Out-of-sample predictions
├── README.md                   # This file
├── stock_prediction_and_eval.ipynb  # Main Jupyter notebook for analysis
├── streamlit_app.py            # Streamlit dashboard for single model exploration
└── streamlit_app2.py           # Enhanced multi-page Streamlit dashboard
```

## Dependencies

This project requires the following Python packages:

- **Core Data Science**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn, statsmodels
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

To run the full analysis pipeline, open and run the Jupyter notebook (`stock_prediction_and_eval.ipynb`). This will:
1. Load and preprocess the data
2. Train various machine learning models
3. Generate out-of-sample predictions
4. Create and evaluate portfolios
5. Save results to the output directories

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
