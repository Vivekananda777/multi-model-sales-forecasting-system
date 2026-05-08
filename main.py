from app.preprocessing import (
    load_data,
    handle_missing_dates
)

from app.feature_engineering import (
    create_features
)

from app.models_training import (
    train_sarima,
    train_prophet,
    train_xgboost,
    train_lstm,
    select_best_model,
    save_model
)

import numpy as np


# ================= Load Data =================

df = load_data(
    "C:/Users/vivekananda/OneDrive/VS_Code/Time series/data/Forecasting Case- Study.xlsx"
)

print(df.head())


# ================= Handle Missing Dates =================

df = handle_missing_dates(df)


# ================= Feature Engineering =================

df = create_features(df)


# ================= Example State =================

state_name = df['State'].unique()[0]

state_df = df[
    df['State'] == state_name
]


# ================= Train Validation Split =================

train_size = int(len(state_df) * 0.8)

train = state_df.iloc[:train_size]

valid = state_df.iloc[train_size:]


# ================= SARIMA =================

sarima_model, sarima_rmse = train_sarima(
    train.set_index('Date')['Total'],
    valid.set_index('Date')['Total']
)

print("SARIMA RMSE:", sarima_rmse)


# ================= Prophet =================

prophet_model, prophet_rmse = train_prophet(
    train[['Date', 'Total']],
    valid[['Date', 'Total']]
)

print("Prophet RMSE:", prophet_rmse)


# ================= XGBoost =================

xgb_model, xgb_rmse = train_xgboost(
    train,
    valid
)

print("XGBoost RMSE:", xgb_rmse)


# ================= LSTM =================

series = state_df['Total'].values

lstm_model, lstm_rmse = train_lstm(series)

print("LSTM RMSE:", lstm_rmse)


# ================= Model Selection =================

results = {
    "SARIMA": sarima_rmse,
    "Prophet": prophet_rmse,
    "XGBoost": xgb_rmse,
    "LSTM": lstm_rmse
}

print(results)

# FORCE XGBoost FOR DEPLOYMENT
best_model = "XGBoost"

print("Selected Deployment Model:", best_model)


# ================= Save Best Model =================

save_model(
    xgb_model,
    r"models/saved_models/best_model.pkl"
)

print("Best model saved successfully")