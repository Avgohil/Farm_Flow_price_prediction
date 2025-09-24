import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*, but LGBMRegressor was fitted with feature names*")
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
TARGETS = ['min_price', 'max_price', 'modal_price']
FEATURES = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety', 'date']

def load_artifact(filename):
    return joblib.load(os.path.join(ARTIFACTS_DIR, filename))

def update_label_encoder(encoder, series):
    # Add new classes if found
    new_classes = set(series.unique()) - set(encoder.classes_)
    if new_classes:
        import numpy as np
        encoder.classes_ = np.concatenate([encoder.classes_, list(new_classes)])
    return encoder

def preprocess(df, encoders, scaler):
    # 0. Generate lag and rolling features (grouped by all key columns, sorted by date)
    group_cols = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety']
    df = df.sort_values(group_cols + ['date'])
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df['modal_price_lag_1'] = df.groupby(group_cols)['modal_price'].shift(1)
    df['modal_price_rolling_7'] = df.groupby(group_cols)['modal_price'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    # Fill missing lag/rolling with median
    df['modal_price_lag_1'] = df['modal_price_lag_1'].fillna(df['modal_price'].median())
    df['modal_price_rolling_7'] = df['modal_price_rolling_7'].fillna(df['modal_price'].median())
    # 1. Encode categorical columns and create encoded columns
    df = df.copy()
    cat_map = {
        'state_name': 'state_encoded',
        'district_name': 'district_encoded',
        'market_name': 'market_encoded',
        'commodity_name': 'commodity_encoded',
        'variety': 'variety_encoded',
    }
    for col, enc_col in cat_map.items():
        df[col] = df[col].astype(str)
        encoders[col] = update_label_encoder(encoders[col], df[col])
        joblib.dump(encoders[col], os.path.join(ARTIFACTS_DIR, ENCODERS[col]))
        df[enc_col] = encoders[col].transform(df[col])

    # 2. Extract year, month, day from date
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # 3. Build feature matrix in correct order
    feature_cols = ['year', 'month', 'day', 'commodity_encoded', 'state_encoded', 'market_encoded', 'modal_price_lag_1', 'modal_price_rolling_7']
    X = df[feature_cols].copy()

    # 4. Scale features
    X_scaled = scaler.transform(X)
    return X_scaled

def load_all_encoders():
    encoders = {}
    for col, fname in ENCODERS.items():
        encoders[col] = load_artifact(fname)
    return encoders

def evaluate_model(test_file, results_file='evaluation_results.csv'):
    model = load_artifact(MODEL_FILE)
    encoders = load_all_encoders()
    scaler = load_artifact(SCALER_FILE)
    df_test = pd.read_csv(test_file, dtype=str)
    for col in TARGETS:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        # Fill NaN in targets with median
        df_test[col] = df_test[col].fillna(df_test[col].median())
    X_test = preprocess(df_test, encoders, scaler)
    y_test = df_test[TARGETS].astype(float).values
    y_pred = model.predict(X_test)
    results = {}
    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = mean_squared_error(y_test[:, i], y_pred[:, i]) ** 0.5
        results[target] = {'MAE': mae, 'RMSE': rmse}
        print(f"{target}: MAE={mae:.2f}, RMSE={rmse:.2f}")
    pd.DataFrame(results).T.to_csv(results_file)
    print(f"Evaluation results saved to {results_file}")

    # Save actual, predicted, and residuals for each target
    pred_df = pd.DataFrame(y_test, columns=[f'actual_{t}' for t in TARGETS])
    for i, t in enumerate(TARGETS):
        pred_df[f'predicted_{t}'] = y_pred[:, i]
        pred_df[f'residual_{t}'] = pred_df[f'actual_{t}'] - pred_df[f'predicted_{t}']
    pred_df.to_csv('prediction_results.csv', index=False)
    print("Detailed prediction results saved to prediction_results.csv")
    return results

if __name__ == "__main__":
    test_file = "master_data_2019_2025.csv"  # Changed to use available file
    evaluate_model(test_file)
