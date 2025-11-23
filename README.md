# 🌾 FarmFlow Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-orange.svg)
![License](https://img.shields.io/badge/License-Internal-red.svg)

*An intelligent agricultural commodity price prediction system using machine learning*

</div>

## 📖 Overview

FarmFlow Price Prediction is a comprehensive machine learning system designed to predict agricultural commodity prices using historical market data. The system automates data collection, preprocessing, model training, and prediction serving through a REST API.

### ✨ Key Features

- 🔄 **Automated Data Pipeline**: Daily data fetching from Agmarknet
- 🤖 **ML-Powered Predictions**: LightGBM multioutput regression model
- 🚀 **FastAPI Server**: RESTful API for real-time predictions
- 📊 **Model Evaluation**: Comprehensive performance metrics
- 🔧 **Automated Workflows**: Batch scripts for scheduled operations
- 📈 **Interactive Analysis**: Jupyter notebooks for data exploration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd farmflow-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the prediction server**
   ```bash
   python mandi_fastapi.py
   ```

## 📁 Project Structure

```
farmflow-price-prediction/
├── 📊 Data Processing
│   ├── fetch_agmarknet_daily.py      # Daily data fetching from Agmarknet (robust parser)
│   ├── combine_filter_daily.py       # Data combination and filtering
│   ├── update_master.py              # Master dataset updates
│   └── check_commodities.py          # Commodity data validation
├── 🤖 Machine Learning
│   ├── train_farmflow_model.py       # Model training pipeline
│   ├── evaluate_model.py             # Model performance evaluation
│   └── update_and_retrain.py         # Automated retraining
├── 🚀 API & Deployment
│   ├── mandi_fastapi.py              # FastAPI prediction server
│   ├── check_artifacts.py            # Model artifact validation
│   └── model_deployment_artifacts/   # Trained models & encoders
├── 📈 Analysis
│   └── Price_prediction.ipynb        # Interactive analysis notebook
├── 🗂️ Data
│   ├── daily_*.csv                   # Daily market data files
│   ├── master_data_2019_2025.csv     # Historical master dataset
│   ├── prediction_results.csv        # Model predictions
│   └── evaluation_results.csv        # Performance metrics
└── ⚙️ Automation
    └── run_farmflow_daily.bat         # Daily automation script
```

## 🖼️ Screenshots



What to include :
- **Pipeline logs / run output:** screenshots of `run_farmflow_daily` or trimmed views of `automation_log.txt` showing successful steps and timestamps. Remove sensitive data and avoid capturing very long logs — crop or trim them.
- **FastAPI UI & prediction output:** screenshots of the `/docs` interface, example request/response payloads, and the server output for a sample prediction.
- **Serve/demo screens:** terminal or browser screenshots showing the server running, health checks, and example client usage.




## 🔧 Usage

### Data Pipeline

```bash
# Fetch daily market data (robust fetcher)
python fetch_agmarknet_daily.py

# Process and combine data
python combine_filter_daily.py

# Update master dataset
python update_master.py
```

### Model Training & Evaluation

```bash
# Train the prediction model
python train_farmflow_model.py

# Evaluate model performance
python evaluate_model.py

# Update and retrain model
python update_and_retrain.py
```

### API Server

```bash
# Start the FastAPI server (recommended: use uvicorn)
uvicorn mandi_fastapi:app --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`

### Automation

For Windows users, run the daily automation:
```cmd
run_farmflow_daily.bat
```


  workflow summary:
- The fetcher reads last date from `master_data_2019_2025.csv` and fetches missing days from Agmarknet into `daily_YYYY-MM-DD.csv` files.
- `combine_filter_daily.py` merges daily CSVs into `daily_new.csv` and applies basic filtering.
- `update_master.py` appends genuinely new rows to the master dataset and returns new rows for retraining.
- `update_and_retrain.py` performs either incremental retrain (on new rows) or a full retrain; artifacts are written to `model_deployment_artifacts/`.
- `mandi_fastapi.py` loads artifacts and serves predictions via `/predict`.



## ✅ Additional Notes

- A robust fetcher implementation is installed as `fetch_agmarknet_daily.py`. It reads the last date from `master_data_2019_2025.csv` and fetches missing days through yesterday.
- A smoke-test script `smoke_predict.py` is included to quickly verify model artifacts load and make a prediction.
- A pinned environment file was saved as `requirements_pinned.txt` to help with reproducible installs. For a minimal runtime set, consider trimming to required packages only.
- Pipeline logs are written to `automation_log.txt` — check this file for retrain/evaluation history and errors.

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get price predictions for commodities |
| `/health` | GET | Check API health status |
| `/docs` | GET | Interactive API documentation |

## 🧠 Model Architecture

The system uses **LightGBM MultiOutput Regressor** with the following features:

- **Input Features**: State, District, Market, Commodity, Variety, Date features
- **Output Targets**: Minimum, Maximum, and Modal prices
- **Preprocessing**: Label encoding for categorical variables, feature scaling
- **Evaluation Metrics**: RMSE, MAE, R² score

## 📊 Model Artifacts

The `model_deployment_artifacts/` directory contains:

- `lgbm_multioutput_regressor_model.pkl` - Trained LightGBM model
- `*_encoder.pkl` - Label encoders for categorical features
- `feature_scaler.pkl` - Feature scaling transformer

## 🔍 Monitoring & Evaluation

- **Performance Tracking**: Automated evaluation with metrics logging
- **Data Validation**: Commodity and artifact integrity checks
- **Automated Retraining**: Scheduled model updates with new data

## 🛠️ Development

### Interactive Development

Use the Jupyter notebook for interactive analysis:
```bash
jupyter notebook Price_prediction.ipynb
```

### Adding New Features

1. Update data processing scripts for new features
2. Retrain the model with `train_farmflow_model.py`
3. Update API endpoints in `mandi_fastapi.py`
4. Test with `evaluate_model.py`

## 📈 Performance

Current model performance metrics are logged in `evaluation_results.csv` with continuous monitoring and improvement.

## 🤝 Contributing

This project is part of our final-year major project, and I lead the complete Machine Learning pipeline.


<div align="center">
Made with ❤️ by the FarmFlow Team
</div>
