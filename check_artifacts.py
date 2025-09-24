import os

REQUIRED_FILES = [
    'commodity_encoder(1).pkl',
    'district_encoder.pkl',
    'feature_scaler.pkl',
    'lgbm_multioutput_regressor_model.pkl',
    'market_encoder.pkl',
    'state_encoder.pkl',
    'variety_encoder.pkl',
]

ARTIFACTS_DIR = 'model_deployment_artifacts'

def check_artifacts():
    missing = []
    for fname in REQUIRED_FILES:
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, fname)):
            missing.append(fname)
    if missing:
        print('Missing files in model_deployment_artifacts:')
        for f in missing:
            print(' -', f)
    else:
        print('All required model artifacts are present.')

if __name__ == "__main__":
    check_artifacts()
