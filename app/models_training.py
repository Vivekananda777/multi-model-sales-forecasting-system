import numpy as np
import joblib

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from prophet import Prophet

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


FEATURES = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'month',
    'week',
    'year',
    'holiday_flag'
]


# ================= SARIMA =================

def train_sarima(train, valid):

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    fitted = model.fit(disp=False)

    preds = fitted.forecast(len(valid))

    rmse = np.sqrt(mean_squared_error(valid, preds))

    return fitted, rmse


# ================= Prophet =================

def train_prophet(train_df, valid_df):

    prophet_train = train_df.rename(
        columns={
            'Date': 'ds',
            'Total': 'y'
        }
    )

    model = Prophet()

    model.fit(prophet_train)

    future = model.make_future_dataframe(
        periods=len(valid_df),
        freq='W'
    )

    forecast = model.predict(future)

    preds = forecast['yhat'].tail(len(valid_df)).values

    rmse = np.sqrt(
        mean_squared_error(
            valid_df['Total'],
            preds
        )
    )

    return model, rmse


# ================= XGBoost =================

def train_xgboost(train_df, valid_df):

    X_train = train_df[FEATURES]
    y_train = train_df['Total']

    X_valid = valid_df[FEATURES]
    y_valid = valid_df['Total']

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    rmse = np.sqrt(
        mean_squared_error(y_valid, preds)
    )

    return model, rmse


def create_sequences(data, seq_length=8):

    X = []
    y = []

    for i in range(len(data) - seq_length):

        X.append(
            data[i:i + seq_length]
        )

        y.append(
            data[i + seq_length]
        )

    return np.array(X), np.array(y)

# ================= LSTM =================

def train_lstm(series):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        series.reshape(-1, 1)
    )

    X, y = create_sequences(scaled)

    X = X.reshape(
        (X.shape[0], X.shape[1], 1)
    )

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = Sequential()

    model.add(
        LSTM(
            64,
            activation='relu',
            input_shape=(X.shape[1], 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        verbose=0
    )

    preds = model.predict(X_test)

    # Convert back to original scale
    preds = scaler.inverse_transform(preds)

    y_test_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_actual,
            preds
        )
    )

    return model, rmse


# ================= Model Selection =================

def select_best_model(results):

    best_model = min(
        results,
        key=results.get
    )

    return best_model


def save_model(model, path):

    joblib.dump(model, path)