import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

ARTIFACTS_DIR = 'model_deployment_artifacts'
MODEL_FILE = 'lgbm_multioutput_regressor_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'
ENCODERS = {
    'state_name': 'state_encoder.pkl',
    'district_name': 'district_encoder.pkl',
    'market_name': 'market_encoder.pkl',
    'variety': 'variety_encoder.pkl',
    'commodity_name': 'commodity_encoder.pkl',
}

def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, MODEL_FILE))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, SCALER_FILE))
    encs = {}
    for k, v in ENCODERS.items():
        encs[k] = joblib.load(os.path.join(ARTIFACTS_DIR, v))
    return model, scaler, encs

def build_sample_row(encs):
    # pick first known class for each encoder if available
    sample = {}
    for k, enc in encs.items():
        try:
            val = enc.classes_[0]
        except Exception:
            val = 'Unknown'
        sample[k] = val
    # numeric lags
    today = datetime.now()
    sample['date'] = today.strftime('%Y-%m-%d')
    sample['modal_price_lag_1'] = 0.0
    sample['modal_price_rolling_7'] = 0.0
    return sample

def run_smoke():
    try:
        model, scaler, encs = load_artifacts()
    except Exception as e:
        print('Failed to load artifacts:', e)
        return 1
    sample = build_sample_row(encs)
    # create features in same order as training
    df = pd.DataFrame([sample])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    # encode
    df['commodity_encoded'] = encs['commodity_name'].transform(df['commodity_name'].astype(str))
    df['state_encoded'] = encs['state_name'].transform(df['state_name'].astype(str))
    df['market_encoded'] = encs['market_name'].transform(df['market_name'].astype(str))
    feature_cols = ['year', 'month', 'day', 'commodity_encoded', 'state_encoded', 'market_encoded', 'modal_price_lag_1', 'modal_price_rolling_7']
    X = df[feature_cols].copy()
    # Ensure column names match those used during training to avoid sklearn warnings
    X.columns = feature_cols
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print('Scaler transform failed:', e)
        return 1
    try:
        preds = model.predict(X_scaled)
        print('Prediction (min, max, modal):', preds)
    except Exception as e:
        print('Model prediction failed:', e)
        return 1
    return 0

if __name__ == '__main__':
    exit(run_smoke())
