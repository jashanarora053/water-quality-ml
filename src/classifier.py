"""
===============================================================================
 classifier.py — Water Quality Classification Inference Module
===============================================================================
 Loads the trained classification model and provides a prediction function.
 Used by the API server (api.py).
===============================================================================
"""

import joblib
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class WaterQualityClassifier:
    """Wrapper for the trained water quality classification model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_meta = None
        self._loaded = False
    
    def load(self):
        """Load the trained model, scaler, and metadata."""
        model_path = os.path.join(MODEL_DIR, 'classifier.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
        meta_path = os.path.join(MODEL_DIR, 'feature_meta.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Classifier model not found at {model_path}. "
                "Run 02_classification.py first!"
            )
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_meta = joblib.load(meta_path)
        self._loaded = True
        print(f"  ✅ Classifier loaded: {type(self.model).__name__}")
    
    def predict(self, features: dict) -> dict:
        """
        Predict water quality class from sensor readings.
        
        Args:
            features: dict with keys matching feature_columns
                      e.g. {'ph': 7.2, 'Solids': 20000, 'Turbidity': 4.5, ...}
        
        Returns:
            dict with quality_class, confidence, and probabilities
        """
        if not self._loaded:
            self.load()
        
        # Build feature vector in correct order
        feature_cols = self.feature_meta['feature_columns']
        class_names = self.feature_meta['class_names']
        
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X_scaled = self.scaler.transform(X)
        
        # Predict class and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        return {
            'quality_class': predicted_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                name: round(float(prob), 4)
                for name, prob in zip(class_names, probabilities)
            }
        }


# Singleton instance
classifier = WaterQualityClassifier()
