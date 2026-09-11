import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

print("="*60)
print(" River Beas Water Quality Classification (XGBoost)")
print("="*60)

# ─── 1. Load the Cleaned Data ────────────────────────────────────────────────
data_path = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\cleaned_beas_data.csv"
print(f"Loading dataset from: {data_path}")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find the dataset at {data_path}. Please run the EDA script first.")

df = pd.read_csv(data_path)
print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# ─── 2. Display Class Distribution ───────────────────────────────────────────
print("\nClass Distribution:")
class_map = {0: 'Excellent', 1: 'Good', 2: 'Poor', 3: 'Dangerous'}
counts = df['quality_label'].value_counts().sort_index()
pcts = (counts / len(df) * 100).round(2)

dist_df = pd.DataFrame({
    'Label': counts.index.map(class_map), 
    'Count': counts, 
    'Percentage (%)': pcts
})
print(dist_df.to_string(index=False))
print("-" * 40)

# ─── 3. Data Preparation ─────────────────────────────────────────────────────
print("\nPreparing data for training...")

# Drop the 'cheating' columns (labels are already generated)
cols_to_drop = ['BOD(mg/l)', 'COD(mg/l)', 'quality_score']
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_cols_to_drop)

print(f"Final sensor features used for training: {list(df.columns[:-1])}")

# Separate Features (X) and Labels (y)
X = df.drop(columns=['quality_label'])
y = df['quality_label'].astype(int)

# Train/Test Split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── 4. Model Training ───────────────────────────────────────────────────────
print("\nTraining XGBoost Model (This might take 30-60 seconds)...")

# Calculate class weights to handle the rare 'Dangerous' samples
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb_boat_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1  # Use all CPU cores for faster training
)

xgb_boat_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# ─── 5. Evaluation ───────────────────────────────────────────────────────────
print(" Training Complete! Evaluating against test set...")

xgb_pred = xgb_boat_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"\n{'='*40}")
print(f"🏆 FINAL MODEL ACCURACY: {xgb_accuracy * 100:.2f}%")
print(f"{'='*40}")

class_names = ['Excellent (0)', 'Good (1)', 'Poor (2)', 'Dangerous (3)']
print("\nDetailed Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=class_names))

# ─── 6. Save Models for Production ───────────────────────────────────────────
save_dir = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\Models\river_beas_v2"
os.makedirs(save_dir, exist_ok=True)
print(f"\n Saving production models to: {save_dir}")

# Save the trained XGBoost model
model_path = os.path.join(save_dir, 'xgboost_beas_model.joblib')
joblib.dump(xgb_boat_model, model_path)

# Save the scaler
scaler_path = os.path.join(save_dir, 'beas_scaler.joblib')
joblib.dump(scaler, scaler_path)

# Save the metadata (tells the API exactly what columns to expect)
feature_meta = {
    'feature_columns': list(X.columns),
    'class_mapping': class_map
}
meta_path = os.path.join(save_dir, 'beas_feature_meta.joblib')
joblib.dump(feature_meta, meta_path)

print(" Success! Pipeline complete.")
