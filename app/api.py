from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import timedelta

from app.feature_engineering import create_features


# ================= CREATE APP =================

app = FastAPI()


# ================= LOAD MODEL =================

model = joblib.load(
    'models/saved_models/best_model.pkl'
)

DATA_PATH = 'data/Forecasting Case- Study.xlsx'


# ================= PREPARE DATA =================

def prepare_state_data(state_name):

    df = pd.read_excel(DATA_PATH)

    df['Date'] = pd.to_datetime(df['Date'])

    df = (
        df.groupby(['State', 'Date'])['Total']
        .sum()
        .reset_index()
    )

    state_df = df[
        df['State'] == state_name
    ]

    state_df = state_df.sort_values('Date')

    return state_df


# ================= HOME ROUTE =================

@app.get('/')
def home():

    return {
        'message': 'Forecasting API Running Successfully'
    }


# ================= FORECAST ROUTE =================

@app.get('/forecast/{state}')
def forecast(state: str):

    state_df = prepare_state_data(state)

    if len(state_df) < 35:

        return {
            'error': 'Not enough historical data'
        }

    # Feature Engineering
    state_df = create_features(state_df)

    last_row = state_df.iloc[-1:].copy()

    forecasts = []

    current_date = state_df['Date'].max()

    for i in range(8):

        future_date = current_date + timedelta(weeks=1)

        future_df = pd.DataFrame({

            'lag_1': [last_row['Total'].values[0]],

            'lag_7': [last_row['lag_7'].values[0]],

            'lag_30': [last_row['lag_30'].values[0]],

            'rolling_mean_7': [
                last_row['rolling_mean_7'].values[0]
            ],

            'rolling_std_7': [
                last_row['rolling_std_7'].values[0]
            ],

            'month': [future_date.month],

            'week': [
                future_date.isocalendar()[1]
            ],

            'year': [future_date.year],

            'holiday_flag': [
                1 if future_date.month in [11, 12]
                else 0
            ]
        })

        prediction = float(
            model.predict(future_df)[0]
        )

        forecasts.append({

            'date': str(future_date.date()),

            'prediction': round(prediction, 2)
        })

        current_date = future_date

        last_row['Total'] = prediction

    return {

        'state': state,

        'next_8_weeks_forecast': forecasts
    }