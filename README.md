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
│   ├── fetch_agmarknet_daily.py      # Daily data fetching from Agmarknet
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

## 🔧 Usage

### Data Pipeline

```bash
# Fetch daily market data
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
# Start the FastAPI server
python mandi_fastapi.py
```

Access the API documentation at `http://localhost:8000/docs`

### Automation

For Windows users, run the daily automation:
```cmd
run_farmflow_daily.bat
```

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

This project is maintained by the FarmFlow team. part of our final year major project and i lead the ML part

<div align="center">
Made with ❤️ by the FarmFlow Team
</div>
