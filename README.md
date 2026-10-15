# Nvidia-stock-price-prediction

Trying different models for predicting the future behavior of Nvidia (NVDA) stock price using hourly historical data. This repository contains simple data acquisition, feature engineering for time-series candlestick data, and a model-training pipeline that compares multiple classifiers and saves trained models.

---

## Quick summary / What this repo does
- Downloads hourly NVDA data (via yfinance), engineers a set of time-series and candlestick features (returns, z-scores, volatility, moving-average positions, lag features), defines a binary target called `volatile`, and trains a variety of classical ML classifiers to predict whether the market will be volatile in the current bar.
- Produces performance metrics (accuracy, F1, MCC, ROC-AUC/PR-AUC where appropriate), tunes a probability threshold (optimize MCC), and saves each fitted model as a `.pkl`.

---

## Architecture & design decisions

High-level components (files):
- Data_acquisition.py
  - DataAcquisition.download_data() — uses yfinance to fetch historical price data (period and interval are parameters).
- Data_preprocessing.py
  - DataPreprocessing.preprocess_data() — canonicalizes time index, computes returns, candle-derived features (range, body, wicks), rolling z-scores, rolling volatility, moving-average positions, creates lag features (Close_lag_i, return_lag_i, etc.), cleans infinities/NaNs, and reports basic dataset info.
  - Visualization helpers for plotting series and confusion matrices.
- pipe_line.py
  - models list — hard-coded collection of classifiers (RandomForest, ExtraTrees, XGBoost, CatBoost, LightGBM, HistGradientBoosting, GradientBoosting, AdaBoost, LogisticRegression pipeline, SVC pipeline).
  - Pipeline class — orchestrates split (time-series split: shuffle=False), model training, threshold tuning (scan 0.30–0.70 for best Matthews Correlation Coefficient), evaluation metrics, and model persistence (joblib).
  - Main routine that loads `data_preprocessed.csv`, separates features vs target (features are columns containing "lag"), iterates over models, trains/evaluates each, prints results, and saves `*_model.pkl`.

Why these decisions were made
- Time-series awareness: train_test_split(... shuffle=False) preserves temporal order and avoids leakage from future data into training.
- Feature engineering focused on candlestick structure and rolling statistics (z-scores, rolling std of returns) to capture short-term volatility patterns.
- Target formulation: `volatile` is derived from the absolute z-score of the current range (threshold 0.5). The pipeline uses lagged features for prediction so labels are based on current bar but predictors are previous bars.
- Class imbalance handling: many models get `class_weight="balanced"` or equivalent settings; CatBoost uses `auto_class_weights="Balanced"`.
- Threshold tuning: instead of raw 0.5 classification for probability-capable models, the pipeline scans thresholds and picks one maximizing MCC (robust to class imbalance).
- Model heterogeneity: trying many tree-based and linear/SVM models helps evaluate which families work best before committing to heavy engineering.

Notable hyperparameters and patterns
- Window size for z-score and win calculation: default 24 (hours).
- Default lag depth: `lage=6` (6 previous observations).
- Several tree ensembles with high `n_estimators` (500–800) for stability but no automated hyperparameter search is present.
- Models are saved as pickled objects (`joblib.dump`) with names like `randomforest_model.pkl`.

---

## Files & data
Top-level important files:
- Data_acquisition.py — data download using yfinance
- Data_preprocessing.py — preprocessing + feature engineering + visualization
- pipe_line.py — training, evaluation, threshold tuning, and model saving
- data_2.csv — raw CSV used by preprocessing in the current main flow
- data_preprocessed.csv — produced output of preprocessing (consumed by pipe_line)
- *.pkl — trained model artifacts (saved by the pipeline)

---

## How to run (minimum steps)
Install dependencies (example):
```bash
python -m pip install -r requirements.txt
# If requirements.txt doesn't exist, install at least:
pip install pandas numpy scikit-learn xgboost lightgbm catboost joblib yfinance matplotlib seaborn
