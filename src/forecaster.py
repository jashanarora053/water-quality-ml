"""
===============================================================================
 forecaster.py — Time-Series Forecasting Inference Module
===============================================================================
 Loads trained Prophet models and provides forecasting functionality.
 Used by the API server (api.py).
===============================================================================
"""

import json
import os
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'prophet_models')


class WaterQualityForecaster:
    """Wrapper for trained Prophet forecasting models."""
    
    def __init__(self):
        self.models = {}
        self.meta = None
        self._loaded = False
    
    def load(self):
        """Load all trained Prophet models."""
        from prophet import Prophet
        from prophet.serialize import model_from_json
        
        meta_path = os.path.join(MODELS_DIR, 'forecast_meta.json')
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Forecast metadata not found at {meta_path}. "
                "Run 04_forecasting.py first!"
            )
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # Load each parameter's model
        for param, filename in self.meta['model_files'].items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    self.models[param] = model_from_json(f.read())
                print(f"  ✅ Forecast model loaded: {param}")
            else:
                print(f"  ⚠️  Model not found for {param}: {model_path}")
        
        self._loaded = True
    
    def forecast(self, parameter: str, hours: int = 24) -> dict:
        """
        Forecast future values for a specific parameter.
        
        Args:
            parameter: which parameter to forecast (e.g., 'ph', 'Solids')
            hours: how many hours ahead to forecast (default: 24)
        
        Returns:
            dict with parameter name, current info, and forecast values
        """
        if not self._loaded:
            self.load()
        
        if parameter not in self.models:
            available = list(self.models.keys())
            raise ValueError(
                f"No model for '{parameter}'. Available: {available}"
            )
        
        model = self.models[parameter]
        
        # Create future dataframe starting from now
        last_date = pd.Timestamp.now()
        future = pd.DataFrame({
            'ds': pd.date_range(start=last_date, periods=hours, freq='h')
        })
        
        forecast = model.predict(future)
        
        # Build response
        forecasts = []
        for _, row in forecast.iterrows():
            forecasts.append({
                'timestamp': row['ds'].isoformat(),
                'predicted': round(float(row['yhat']), 4),
                'lower_bound': round(float(row['yhat_lower']), 4),
                'upper_bound': round(float(row['yhat_upper']), 4)
            })
        
        # Model performance metrics
        metrics = self.meta['results'].get(parameter, {})
        
        return {
            'parameter': parameter,
            'forecast_hours': hours,
            'model_metrics': {
                'mae': metrics.get('mae', None),
                'rmse': metrics.get('rmse', None),
                'mape': metrics.get('mape', None)
            },
            'forecasts': forecasts
        }
    
    def forecast_all(self, hours: int = 24) -> dict:
        """Forecast all available parameters."""
        if not self._loaded:
            self.load()
        
        results = {}
        for param in self.models.keys():
            try:
                results[param] = self.forecast(param, hours)
            except Exception as e:
                results[param] = {'error': str(e)}
        
        return results
    
    def get_available_parameters(self) -> list:
        """Return list of parameters that can be forecasted."""
        if not self._loaded:
            self.load()
        return list(self.models.keys())


# Singleton instance
forecaster = WaterQualityForecaster()
