import logging


def full_retrain(master_file, model_file):
    """
    Retrain the model from scratch using the entire master dataset and the new feature engineering pipeline.
    """
    logging.info("Starting full retrain on all data...")
    logging.info("Loading master file: %s", master_file)
    df_master = pd.read_csv(master_file, dtype=str)
    # Use the same feature engineering as preprocess_new_data
    group_cols = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety']
    df_master = df_master.sort_values(group_cols + ['date'])
    df_master['modal_price'] = pd.to_numeric(df_master['modal_price'], errors='coerce')
    df_master['modal_price_lag_1'] = df_master.groupby(group_cols)['modal_price'].shift(1)
    df_master['modal_price_rolling_7'] = df_master.groupby(group_cols)['modal_price'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    df_master['modal_price_lag_1'] = df_master['modal_price_lag_1'].fillna(df_master['modal_price'].median())
    df_master['modal_price_rolling_7'] = df_master['modal_price_rolling_7'].fillna(df_master['modal_price'].median())
    # Encode categorical columns
    encoders = {}
    cat_map = {
        'state_name': 'state_encoded',
        'district_name': 'district_encoded',
        'market_name': 'market_encoded',
        'commodity_name': 'commodity_encoded',
        'variety': 'variety_encoded',
    }
    for col, fname in ENCODERS.items():
        encoders[col] = load_encoder(os.path.join(ARTIFACTS_DIR, fname)) if os.path.exists(os.path.join(ARTIFACTS_DIR, fname)) else LabelEncoder().fit(df_master[col].astype(str))
    for col, enc_col in cat_map.items():
        df_master[col] = df_master[col].astype(str)
        encoders[col] = update_label_encoder(encoders[col], df_master[col])
        save_encoder(encoders[col], os.path.join(ARTIFACTS_DIR, ENCODERS[col]))
        df_master[enc_col] = encoders[col].transform(df_master[col])
    # Extract year, month, day from date
    df_master['date'] = pd.to_datetime(df_master['date'], errors='coerce', dayfirst=True)
    df_master['year'] = df_master['date'].dt.year
    df_master['month'] = df_master['date'].dt.month
    df_master['day'] = df_master['date'].dt.day
    # Build feature matrix in correct order
    feature_cols = ['year', 'month', 'day', 'commodity_encoded', 'state_encoded', 'market_encoded', 'modal_price_lag_1', 'modal_price_rolling_7']
    # Drop rows with missing target values
    df_master = df_master.dropna(subset=TARGETS)
    X = df_master[feature_cols].copy()
    logging.info("Preprocessing complete. Feature matrix shape: %s", X.shape)
    scaler = StandardScaler().fit(X)
    save_encoder(scaler, os.path.join(ARTIFACTS_DIR, SCALER_FILE))
    logging.info("Feature scaler fitted and saved: %s", SCALER_FILE)
    X_scaled = scaler.transform(X)
    y = df_master[TARGETS].astype(float)
    # Train model
    model = MultiOutputRegressor(LGBMRegressor())
    try:
        logging.info("Starting model.fit on %d samples and %d targets", X_scaled.shape[0], y.shape[1] if len(y.shape) > 1 else 1)
        model.fit(X_scaled, y)
        logging.info("Model.fit complete.")
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, model_file))
        logging.info("Full retrain complete and model saved: %s", model_file)
    except Exception as e:
        logging.exception("Full retrain failed during model.fit: %s", e)
        raise
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename="automation_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def log_status(message):
    print(message)
    logging.info(message)

def log_error(message):
    print("ERROR:", message)
    logging.error(message)

# Fetch daily data before processing
def fetch_daily_data():
    try:
        subprocess.run(["python", "fetch_agmarknet_daily.py"], check=True)
        log_status("Daily data fetched successfully.")
    except Exception as e:
        log_error(f"Data fetching failed: {e}")
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename="automation_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def log_status(message):
    print(message)
    logging.info(message)

def log_error(message):
    print("ERROR:", message)
    logging.error(message)
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# Paths
ARTIFACTS_DIR = 'model_deployment_artifacts'
ENCODERS = {
    'state_name': 'state_encoder.pkl',
    'district_name': 'district_encoder.pkl',
    'market_name': 'market_encoder.pkl',
    'variety': 'variety_encoder.pkl',
    'commodity_name': 'commodity_encoder.pkl',
}
SCALER_FILE = 'feature_scaler.pkl'
MODEL_FILE = 'lgbm_multioutput_regressor_model.pkl'

TARGETS = ['min_price', 'max_price', 'modal_price']
FEATURES = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety', 'date']

COMMODITY_LIST = [
    'Onion', 'Potato', 'Banana', 'Bajra(Pearl Millet/Cumbu)', 'Grapes',
    'Green Chilli', 'Cummin Seed(Jeera)', 'Coconut', 'Tomato',
    'Arecanut(Betelnut/Supari)', 'Jute', 'Rice', 'Mustard', 'Cotton',
    'Wheat', 'Soyabean', 'Jowar(Sorghum)', 'Maize', 'Groundnut',
    'Ragi (Finger Millet)'
]

MASTER_COLUMNS = FEATURES[:5] + TARGETS + ['date']
KEY_COLS = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety', 'date']

def load_encoder(path):
    return joblib.load(path)

def save_encoder(encoder, path):
    joblib.dump(encoder, path)

def update_label_encoder(encoder, series):
    # Add new classes if found
    new_classes = set(series.unique()) - set(encoder.classes_)
    if new_classes:
        encoder.classes_ = np.concatenate([encoder.classes_, list(new_classes)])
    return encoder

def update_master(master_file, daily_file):
    df_master = pd.read_csv(master_file, dtype=str)
    df_daily = pd.read_csv(daily_file, dtype=str)
    # Filter commodities
    df_daily = df_daily[df_daily['commodity_name'].isin(COMMODITY_LIST)]
    # Keep only master columns, in order
    df_daily = df_daily[MASTER_COLUMNS]
    # Handle missing values
    for col in TARGETS:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        median = df_daily[col].median()
        df_daily[col] = df_daily[col].fillna(median)
    for col in FEATURES:
        mode = df_daily[col].mode()[0] if not df_daily[col].mode().empty else ''
        df_daily[col] = df_daily[col].fillna(mode)
    # Find truly new rows
    merged = df_daily.merge(df_master[KEY_COLS], on=KEY_COLS, how='left', indicator=True)
    new_daily_rows = merged[merged['_merge'] == 'left_only']
    num_new_rows = len(new_daily_rows)
    # Append only new rows
    if num_new_rows > 0:
        df_master = pd.concat([df_master, df_daily.loc[new_daily_rows.index]], ignore_index=True)
        df_master = df_master.drop_duplicates(subset=KEY_COLS, keep='last').reset_index(drop=True)
        df_master.to_csv(master_file, index=False)
    else:
        df_master = df_master.drop_duplicates(subset=KEY_COLS, keep='last').reset_index(drop=True)
        df_master.to_csv(master_file, index=False)
    print(f"Number of new rows added: {num_new_rows}")
    print(f"Final total number of rows in {master_file}: {len(df_master)}")
    return df_daily.loc[new_daily_rows.index]  # Return only the new rows for retraining

def preprocess_new_data(new_data):
    # --- Feature engineering to match evaluate_model.py ---
    import numpy as np
    group_cols = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety']
    new_data = new_data.sort_values(group_cols + ['date'])
    new_data['modal_price'] = pd.to_numeric(new_data['modal_price'], errors='coerce')
    new_data['modal_price_lag_1'] = new_data.groupby(group_cols)['modal_price'].shift(1)
    new_data['modal_price_rolling_7'] = new_data.groupby(group_cols)['modal_price'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    new_data['modal_price_lag_1'] = new_data['modal_price_lag_1'].fillna(new_data['modal_price'].median())
    new_data['modal_price_rolling_7'] = new_data['modal_price_rolling_7'].fillna(new_data['modal_price'].median())
    # Encode categorical columns
    encoders = {}
    cat_map = {
        'state_name': 'state_encoded',
        'district_name': 'district_encoded',
        'market_name': 'market_encoded',
        'commodity_name': 'commodity_encoded',
        'variety': 'variety_encoded',
    }
    for col, fname in ENCODERS.items():
        encoders[col] = load_encoder(os.path.join(ARTIFACTS_DIR, fname))
    for col, enc_col in cat_map.items():
        new_data[col] = new_data[col].astype(str)
        encoders[col] = update_label_encoder(encoders[col], new_data[col])
        save_encoder(encoders[col], os.path.join(ARTIFACTS_DIR, ENCODERS[col]))
        new_data[enc_col] = encoders[col].transform(new_data[col])
    # Extract year, month, day from date
    new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce', dayfirst=True)
    new_data['year'] = new_data['date'].dt.year
    new_data['month'] = new_data['date'].dt.month
    new_data['day'] = new_data['date'].dt.day
    # Build feature matrix in correct order
    feature_cols = ['year', 'month', 'day', 'commodity_encoded', 'state_encoded', 'market_encoded', 'modal_price_lag_1', 'modal_price_rolling_7']
    X = new_data[feature_cols].copy()
    # Scale features
    scaler = load_encoder(os.path.join(ARTIFACTS_DIR, SCALER_FILE))
    scaler.fit(X)
    save_encoder(scaler, os.path.join(ARTIFACTS_DIR, SCALER_FILE))
    X_scaled = scaler.transform(X)
    y = new_data[TARGETS].astype(float)
    return X_scaled, y

def incremental_retrain(model_file, new_data):
    if new_data.empty:
        print("No new data for retraining. Skipping model update.")
        return
    X_new, y_new = preprocess_new_data(new_data)
    # Load model
    model = joblib.load(os.path.join(ARTIFACTS_DIR, model_file))
    # Retrain (fit on new data)
    model.fit(X_new, y_new)
    # Save updated model
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, model_file))
    print("Incremental retraining complete and model saved.")

if __name__ == "__main__":
    master_file = 'master_data_2019_2025.csv'
    daily_file = 'daily_new.csv'
    model_file = MODEL_FILE
    import sys
    try:
        log_status("=== Pipeline run started ===")
        # Always generate fresh daily_new.csv first
        import subprocess
        try:
            subprocess.run(["python", "combine_filter_daily.py"], check=True)
            log_status("daily_new.csv generated successfully.")
        except Exception as e:
            log_error(f"combine_filter_daily.py failed: {e}")
            raise
        if len(sys.argv) > 1 and sys.argv[1] == "full_retrain":
            # Perform a full retrain on all data
            full_retrain(master_file, model_file)
        else:
            fetch_daily_data()
            new_rows = update_master(master_file, daily_file)
            if new_rows.empty:
                log_status("No new data for retraining. Skipping model update.")
            else:
                incremental_retrain(model_file, new_rows)
        # Always run evaluation and log results
        try:
            from evaluate_model import evaluate_model
            results = evaluate_model(master_file)
            log_status(f"Evaluation results: {results}")
        except Exception as eval_err:
            log_error(f"Evaluation failed: {eval_err}")
        log_status("=== Pipeline run completed successfully ===")
    except Exception as e:
        log_error(f"Pipeline failed: {e}")
