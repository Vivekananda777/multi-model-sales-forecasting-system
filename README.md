# Multi-Model Sales Forecasting System with FastAPI

## Project Overview

This project is an end-to-end time series forecasting system developed to predict the next 8 weeks of sales for different states using historical sales data.

The system:

* preprocesses and cleans time-series data
* handles missing dates and missing values
* performs feature engineering
* trains multiple forecasting models
* compares model performance automatically
* selects the best model using RMSE
* deploys predictions using FastAPI REST API

---

# Problem Statement

Forecast the next 8 weeks of sales for each state using historical sales data while:

* handling missing values
* handling seasonality and trends
* performing feature engineering
* comparing multiple forecasting algorithms
* exposing predictions through a REST API

---

# Models Implemented

1. SARIMA
2. Prophet
3. XGBoost
4. LSTM

---

# Feature Engineering

Implemented features:

* Lag Features (t-1, t-7, t-30)
* Rolling Mean
* Rolling Standard Deviation
* Month Feature
* Week Feature
* Year Feature
* Holiday Flag

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Statsmodels
* Prophet
* XGBoost
* TensorFlow/Keras
* FastAPI
* Uvicorn

---

# Project Structure
```
forecasting-system/

│

├── app/

│   ├── api.py

│   ├── preprocessing.py

│   ├── feature_engineering.py

│   ├── models_training.py

│   └── predict.py

│

├── data/

│   └── Forecasting Case- Study.xlsx

│

├── models/

│   └── saved_models/

│

├── main.py

├── requirements.txt

└── README.md

---
```
# Installation

Install dependencies:

pip install -r requirements.txt

---

# Run Training Pipeline

python main.py

---

# Run API

python -m uvicorn app.api:app --reload

---

# Swagger API Documentation

Open:

http://127.0.0.1:8000/docs


# Model Evaluation

Models are evaluated using RMSE (Root Mean Squared Error).

The system automatically selects the best-performing model for deployment.

---

# Key Highlights

* End-to-end forecasting pipeline
* Multiple forecasting algorithms
* Automated model comparison
* REST API deployment
* Production-style architecture
* Real-time prediction support

---

# Future Improvements

* Docker deployment
* Cloud deployment (AWS/GCP)
* CI/CD pipeline
* MLflow integration
* Real-time monitoring

---

