import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

ARTIFACTS_DIR = 'model_deployment_artifacts'
MODEL_FILE = 'lgbm_multioutput_regressor_model.pkl'
ENCODERS = {
    'state_name': 'state_encoder.pkl',
    'district_name': 'district_encoder.pkl',
    'market_name': 'market_encoder.pkl',
    'variety': 'variety_encoder.pkl',
    'commodity_name': 'commodity_encoder.pkl',
}
SCALER_FILE = 'feature_scaler.pkl'
GROUP_COLS = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety']
FEATURE_COLS = [
    'year', 'month', 'day',
    'commodity_encoded', 'state_encoded', 'market_encoded',
    'modal_price_lag_1', 'modal_price_rolling_7'
]

class PredictRequest(BaseModel):
    state_name: str
    district_name: str
    market_name: str
    commodity_name: str
    variety: str
    date: str  # 'YYYY-MM-DD'

class PredictResponse(BaseModel):
    min_price: float
    max_price: float
    modal_price: float

app = FastAPI(
    title="Mandi Price Prediction API",
    description="Predicts min, max, and modal prices for mandi commodities with incremental encoder updates.",
    version="1.0"
)

def load_artifact(filename):
    return joblib.load(os.path.join(ARTIFACTS_DIR, filename))

def save_artifact(obj, filename):
    joblib.dump(obj, os.path.join(ARTIFACTS_DIR, filename))

def update_label_encoder(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.concatenate([encoder.classes_, [value]])
    return encoder

def preprocess_input(data: dict, encoders, scaler, history_df=None):
    df = pd.DataFrame([data])
    cat_map = {
        'state_name': 'state_encoded',
        'district_name': 'district_encoded',
        'market_name': 'market_encoded',
        'commodity_name': 'commodity_encoded',
        'variety': 'variety_encoded',
    }
    for col, enc_col in cat_map.items():
        df[col] = df[col].astype(str)
        encoders[col] = update_label_encoder(encoders[col], df[col].iloc[0])
        save_artifact(encoders[col], ENCODERS[col])
        df[enc_col] = encoders[col].transform(df[col])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    # Lag and rolling features
    if history_df is not None and not history_df.empty:
        # Filter history for the same group
        mask = (
            (history_df['state_name'] == data['state_name']) &
            (history_df['district_name'] == data['district_name']) &
            (history_df['market_name'] == data['market_name']) &
            (history_df['commodity_name'] == data['commodity_name']) &
            (history_df['variety'] == data['variety'])
        )
        group_hist = history_df[mask].copy()
        group_hist['date'] = pd.to_datetime(group_hist['date'], errors='coerce', dayfirst=True)
        group_hist = group_hist[group_hist['date'] < df['date'].iloc[0]]
        group_hist = group_hist.sort_values('date')
        group_hist['modal_price'] = pd.to_numeric(group_hist['modal_price'], errors='coerce')
        # Compute lag and rolling
        if not group_hist.empty:
            df['modal_price_lag_1'] = group_hist['modal_price'].iloc[-1]
            df['modal_price_rolling_7'] = group_hist['modal_price'].tail(7).mean()
        else:
            df['modal_price_lag_1'] = 0
            df['modal_price_rolling_7'] = 0
    else:
        df['modal_price_lag_1'] = 0
        df['modal_price_rolling_7'] = 0
    X = df[FEATURE_COLS].copy()
    X_scaled = scaler.transform(X)
    # Return as DataFrame with feature names to avoid LightGBM warning
    return pd.DataFrame(X_scaled, columns=FEATURE_COLS)

model = load_artifact(MODEL_FILE)
encoders = {col: load_artifact(fname) for col, fname in ENCODERS.items()}
scaler = load_artifact(SCALER_FILE)
try:
    history_df = pd.read_csv('master_data_2019_2025.csv', dtype=str)
except Exception:
    history_df = None

@app.post("/predict", response_model=PredictResponse)
def predict_price(request: PredictRequest):
    try:
        X_scaled = preprocess_input(request.dict(), encoders, scaler, history_df)
        y_pred = model.predict(X_scaled)[0]
        print(f"Request: {request.dict()} | Prediction: {y_pred}")
        return PredictResponse(
            min_price=float(y_pred[0]),
            max_price=float(y_pred[1]),
            modal_price=float(y_pred[2])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
