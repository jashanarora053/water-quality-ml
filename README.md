# Water Quality ML Models — README

## 🌊 Overview

This project contains **three ML models** for real-time water quality monitoring:

1. **Water Quality Classifier** — Classifies water as Excellent/Good/Poor/Dangerous
2. **Anomaly Detector** — Flags unusual readings (possible contamination)
3. **Time-Series Forecaster** — Predicts future parameter values

## 🚀 Quick Start

### 1. Setup
```bash
cd water-quality-ml
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Train Models (run in order)
```bash
python notebooks/01_eda.py              # Data exploration & cleaning
python notebooks/02_classification.py    # Train classifier
python notebooks/03_anomaly_detection.py # Train anomaly detector
python notebooks/04_forecasting.py       # Train forecaster (needs 'prophet')
```

### 3. Start API Server
```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for interactive API documentation.

## 📡 API Endpoints (For Teammates)

### POST `/predict` — Classify + Detect Anomalies

**Request:**
```json
{
  "ph": 7.2,
  "Hardness": 200,
  "Solids": 20000,
  "Chloramines": 7.0,
  "Sulfate": 300,
  "Conductivity": 400,
  "Organic_carbon": 14,
  "Trihalomethanes": 60,
  "Turbidity": 4.0,
  "temperature": 28.3,
  "do": 6.8
}
```

**Response:**
```json
{
  "classification": {
    "quality_class": "Good",
    "confidence": 0.92,
    "probabilities": { "Excellent": 0.05, "Good": 0.92, "Poor": 0.03, "Dangerous": 0.0 }
  },
  "anomaly_detection": {
    "is_anomaly": false,
    "anomaly_score": -0.12,
    "severity": "NONE",
    "message": "Reading is within normal range.",
    "parameter_status": { "ph": "Normal", "Turbidity": "Normal", ... }
  }
}
```

### GET `/forecast?parameter=ph&hours=24` — Forecast Parameter

### GET `/forecast/all?hours=24` — Forecast All Parameters

### GET `/health` — Health Check

## 📁 Project Structure
```
water-quality-ml/
├── data/raw/                    # Raw datasets
├── data/processed/              # Cleaned data
├── notebooks/                   # Training scripts
│   ├── 01_eda.py
│   ├── 02_classification.py
│   ├── 03_anomaly_detection.py
│   └── 04_forecasting.py
├── models/                      # Saved trained models
├── src/                         # Inference modules + API
│   ├── api.py                   # FastAPI server
│   ├── classifier.py
│   ├── anomaly_detector.py
│   └── forecaster.py
└── requirements.txt
```
