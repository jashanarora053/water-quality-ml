import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

print("="*70)
print(" 🔬 XGBoost Feature Combination Experiment")
print("="*70)

# ─── 1. Load the Cleaned Data ────────────────────────────────────────────────
data_path = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\cleaned_beas_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find the dataset at {data_path}.")

df = pd.read_csv(data_path)

# Drop the 'cheating' columns (we never train on these)
cheating_cols = ['BOD(mg/l)', 'COD(mg/l)', 'quality_score']
existing_cheating_cols = [c for c in cheating_cols if c in df.columns]
df = df.drop(columns=existing_cheating_cols)

# ─── 2. Define Feature Groups ────────────────────────────────────────────────
# These are the 4 pure, independent sensors
core_features = ['pH', 'TURBIDITY(NTU)', 'COND.(µS/cm)', 'DO(mg/l)']

# These are the redundant/bad sensors
redundant_features = ['TDS', 'CL(mg/l)']

# Setup the target variable
y = df['quality_label'].astype(int)

# ─── 3. Define the Experiments ───────────────────────────────────────────────
experiments = {
    "1. Base Model (Core 4 Sensors Only)": core_features,
    
    # Adding 1 Redundant Feature
    "2. Core + TDS": core_features + ['TDS'],
    "3. Core + Chlorides (CL)": core_features + ['CL(mg/l)'],

    # Adding all 
    "4. All Features (Kitchen Sink)": core_features + redundant_features
}


results = []

print(f"Total Rows: {len(df)}\n")
print("Running 4 separate experiments. This will take about 2-3 minutes...\n")

# ─── 4. Run the Training Loop ────────────────────────────────────────────────
for exp_name, features in experiments.items():
    # Only use the features that actually exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    
    print(f"Training: {exp_name} (Using {len(valid_features)} features)...")
    
    # Select features for this specific experiment
    X = df[valid_features]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Initialize Model
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1 
    )
    
    # Train
    xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Predict & Evaluate
    xgb_pred = xgb_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, xgb_pred)
    
    # Save result
    results.append({
        'Experiment': exp_name,
        'Num Features': len(valid_features),
        'Accuracy (%)': round(acc * 100, 3)
    })

# ─── 5. Display Results ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  EXPERIMENT RESULTS SUMMARY")
print("="*70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("="*70)
