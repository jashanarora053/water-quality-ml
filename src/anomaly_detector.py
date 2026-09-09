"""
===============================================================================
 anomaly_detector.py — Anomaly / Pollution Detection Inference Module
===============================================================================
 Loads the trained Isolation Forest model and provides anomaly detection.
 Used by the API server (api.py).
===============================================================================
"""

import joblib
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class WaterAnomalyDetector:
    """Wrapper for the trained Isolation Forest anomaly detection model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta = None
        self._loaded = False
    
    def load(self):
        """Load the trained model and metadata."""
        model_path = os.path.join(MODEL_DIR, 'anomaly_detector.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
        meta_path = os.path.join(MODEL_DIR, 'anomaly_meta.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Anomaly detector not found at {model_path}. "
                "Run 03_anomaly_detection.py first!"
            )
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.meta = joblib.load(meta_path)
        self._loaded = True
        print(f"  ✅ Anomaly detector loaded: IsolationForest")
    
    def detect(self, features: dict) -> dict:
        """
        Detect if a sensor reading is anomalous.
        
        Args:
            features: dict with sensor readings
                      e.g. {'ph': 7.2, 'Solids': 20000, ...}
        
        Returns:
            dict with is_anomaly, anomaly_score, and message
        """
        if not self._loaded:
            self.load()
        
        feature_cols = self.meta['feature_columns']
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X_scaled = self.scaler.transform(X)
        
        # Predict: 1 = normal, -1 = anomaly
        prediction = self.model.predict(X_scaled)[0]
        score = float(self.model.decision_function(X_scaled)[0])
        
        is_anomaly = prediction == -1
        
        # Determine severity based on score
        if score < -0.3:
            severity = "CRITICAL"
            message = "⛔ Critical anomaly detected — possible severe contamination!"
        elif score < -0.1:
            severity = "HIGH"
            message = "🚨 High anomaly — unusual readings detected, investigate immediately."
        elif score < 0:
            severity = "MODERATE"
            message = "⚠️ Moderate anomaly — readings slightly outside normal range."
        else:
            severity = "NONE"
            message = "✅ Reading is within normal range."
        
        # Check individual parameter thresholds
        param_status = self._check_individual_params(features)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': round(score, 4),
            'severity': severity,
            'message': message,
            'parameter_status': param_status
        }
    
    def _check_individual_params(self, features: dict) -> dict:
        """Check each parameter against known safe ranges."""
        thresholds = {
            'ph': {'min': 6.5, 'max': 8.5, 'unit': ''},
            'Solids': {'min': 0, 'max': 50000, 'unit': 'mg/L'},
            'Turbidity': {'min': 0, 'max': 5, 'unit': 'NTU'},
            'Conductivity': {'min': 0, 'max': 800, 'unit': 'µS/cm'},
            'Chloramines': {'min': 0, 'max': 4, 'unit': 'mg/L'},
            'Sulfate': {'min': 0, 'max': 250, 'unit': 'mg/L'},
            'Hardness': {'min': 0, 'max': 300, 'unit': 'mg/L'},
            'Organic_carbon': {'min': 0, 'max': 15, 'unit': 'mg/L'},
            'Trihalomethanes': {'min': 0, 'max': 80, 'unit': 'µg/L'},
            # Sensor-specific params (for your boat)
            'temperature': {'min': 10, 'max': 35, 'unit': '°C'},
            'do': {'min': 4, 'max': 14, 'unit': 'mg/L'},
            'tds': {'min': 0, 'max': 500, 'unit': 'mg/L'},
        }
        
        status = {}
        for param, value in features.items():
            if param in thresholds:
                t = thresholds[param]
                if t['min'] <= value <= t['max']:
                    status[param] = 'Normal'
                elif value < t['min']:
                    status[param] = f'Low (min: {t["min"]}{t["unit"]})'
                else:
                    status[param] = f'High (max: {t["max"]}{t["unit"]})'
            else:
                status[param] = 'Unknown parameter'
        
        return status


# Singleton instance
anomaly_detector = WaterAnomalyDetector()
