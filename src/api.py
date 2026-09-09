"""
===============================================================================
 api.py — FastAPI Prediction Server
===============================================================================
 This is the main API that your teammates will call. It wraps all three
 ML models behind clean REST endpoints.
 
 Endpoints:
   POST /predict          → Classification + Anomaly Detection
   GET  /forecast         → Time-Series Forecast for a parameter
   GET  /forecast/all     → Forecast all parameters
   GET  /health           → Health check
   GET  /docs             → Auto-generated Swagger docs
 
 Usage:
   cd water-quality-ml
   venv\\Scripts\\python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
 
 Then visit: http://localhost:8000/docs
===============================================================================
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier import classifier
from src.anomaly_detector import anomaly_detector

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌊 Water Quality ML API",
    description=(
        "Real-time water quality prediction API for the RC Boat monitoring system.\n\n"
        "**Three ML Models:**\n"
        "1. **Water Quality Classification** — Classifies water as Excellent/Good/Poor/Dangerous\n"
        "2. **Anomaly Detection** — Flags unusual readings (possible contamination)\n"
        "3. **Time-Series Forecasting** — Predicts future parameter values\n"
    ),
    version="1.0.0",
)

# Allow all origins (for teammates' frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response Models ─────────────────────────────────────────────────

class SensorReading(BaseModel):
    """Input sensor data from the RC boat."""
    ph: float = Field(..., ge=0, le=14, description="pH value (0-14)")
    Hardness: float = Field(default=200, ge=0, description="Hardness (mg/L)")
    Solids: float = Field(default=20000, ge=0, description="Total Dissolved Solids (mg/L)")
    Chloramines: float = Field(default=7, ge=0, description="Chloramines (mg/L)")
    Sulfate: float = Field(default=300, ge=0, description="Sulfate (mg/L)")
    Conductivity: float = Field(default=400, ge=0, description="Conductivity (µS/cm)")
    Organic_carbon: float = Field(default=14, ge=0, description="Organic Carbon (mg/L)")
    Trihalomethanes: float = Field(default=60, ge=0, description="Trihalomethanes (µg/L)")
    Turbidity: float = Field(default=4, ge=0, description="Turbidity (NTU)")
    
    # Optional boat-specific params (not in training data but good to track)
    temperature: Optional[float] = Field(default=None, description="Water temperature (°C)")
    do: Optional[float] = Field(default=None, description="Dissolved Oxygen (mg/L)")
    tds: Optional[float] = Field(default=None, description="TDS from sensor (mg/L)")

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                    "do": 6.8,
                    "tds": 320
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Combined prediction results from all models."""
    classification: dict
    anomaly_detection: dict
    input_data: dict


class ForecastResponse(BaseModel):
    """Forecast results for a parameter."""
    parameter: str
    forecast_hours: int
    model_metrics: dict
    forecasts: list


# ─── Load Models at Startup ──────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    """Load all ML models when the server starts."""
    print("\n" + "=" * 60)
    print("  Loading ML Models...")
    print("=" * 60)
    
    try:
        classifier.load()
    except FileNotFoundError as e:
        print(f"  ⚠️  Classifier not loaded: {e}")
    
    try:
        anomaly_detector.load()
    except FileNotFoundError as e:
        print(f"  ⚠️  Anomaly detector not loaded: {e}")
    
    # Forecaster is loaded lazily (only when /forecast is called)
    # because Prophet imports are slow
    
    print("=" * 60)
    print("  🚀 API is ready!")
    print("  📖 Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API and models are running."""
    return {
        "status": "healthy",
        "models": {
            "classifier": classifier._loaded,
            "anomaly_detector": anomaly_detector._loaded,
        },
        "message": "Water Quality ML API is running! 🌊"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(reading: SensorReading):
    """
    🏷️ Classify water quality AND detect anomalies in a single call.
    
    Send sensor readings → get back quality classification + anomaly detection.
    This is the main endpoint your teammates should use.
    """
    features = reading.model_dump(exclude_none=True)
    
    try:
        # Run classification
        classification_result = classifier.predict(features)
    except Exception as e:
        classification_result = {"error": str(e)}
    
    try:
        # Run anomaly detection
        anomaly_result = anomaly_detector.detect(features)
    except Exception as e:
        anomaly_result = {"error": str(e)}
    
    return PredictionResponse(
        classification=classification_result,
        anomaly_detection=anomaly_result,
        input_data=features
    )


@app.get("/forecast", tags=["Forecasting"])
async def forecast(
    parameter: str = Query(
        default="ph",
        description="Parameter to forecast (e.g., ph, Solids, Turbidity, Conductivity)"
    ),
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
        description="Hours ahead to forecast (1-168)"
    )
):
    """
    📈 Forecast future values for a specific water quality parameter.
    
    Returns predicted values with confidence intervals for the next N hours.
    """
    try:
        from src.forecaster import forecaster
        result = forecaster.forecast(parameter, hours)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.get("/forecast/all", tags=["Forecasting"])
async def forecast_all(
    hours: int = Query(default=24, ge=1, le=168, description="Hours ahead to forecast")
):
    """
    📈 Forecast ALL available parameters at once.
    
    Returns forecasts for every parameter the model was trained on.
    """
    try:
        from src.forecaster import forecaster
        result = forecaster.forecast_all(hours)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.get("/forecast/parameters", tags=["Forecasting"])
async def forecast_parameters():
    """
    📋 List all parameters available for forecasting.
    """
    try:
        from src.forecaster import forecaster
        params = forecaster.get_available_parameters()
        return {"available_parameters": params}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─── Run directly ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
