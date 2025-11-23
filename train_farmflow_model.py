"""Wrapper to run full model training using update_and_retrain.py utilities.
This script calls the `full_retrain` function from `update_and_retrain` to
retrain the LightGBM multioutput model on the entire master dataset.

Usage:
    python train_farmflow_model.py
"""
import os
import sys
import logging

try:
    import update_and_retrain as uar
except Exception as e:
    print(f"Failed to import update_and_retrain: {e}")
    sys.exit(1)

# Ensure logging is configured and writes to automation_log.txt
logging.basicConfig(
    filename="automation_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)


def main():
    master_file = 'master_data_2019_2025.csv'
    model_file = uar.MODEL_FILE if hasattr(uar, 'MODEL_FILE') else 'lgbm_multioutput_regressor_model.pkl'
    logging.info(f"Starting full retrain: master={master_file}, model_out={model_file}")
    try:
        uar.full_retrain(master_file, model_file)
        logging.info("Full retrain finished successfully.")
    except Exception as e:
        logging.exception(f"Full retrain failed: {e}")
        raise


if __name__ == '__main__':
    main()
