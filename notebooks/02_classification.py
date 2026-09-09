"""
===============================================================================
 02_classification.py — Water Quality Classification Model
===============================================================================
 Trains and evaluates Random Forest & XGBoost classifiers on the cleaned
 water quality dataset.
 
 What this script does:
   1. Loads cleaned dataset (from 01_eda.py output)
   2. Prepares features & labels
   3. Splits into train/test (80/20 stratified)
   4. Scales features (StandardScaler)
   5. Trains Random Forest
   6. Trains XGBoost
   7. Evaluates both (accuracy, F1, confusion matrix)
   8. Plots feature importance
   9. Saves best model + scaler
 
 Prerequisites:
   Run 01_eda.py first!
 
 Usage:
   python notebooks/02_classification.py
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join('data', 'processed', 'water_quality_cleaned.csv')
MODELS_DIR = 'models'
PLOTS_DIR = os.path.join('notebooks', 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Feature columns (all numeric sensor-type parameters)
FEATURE_COLS = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

print("=" * 70)
print("  WATER QUALITY — CLASSIFICATION MODEL TRAINING")
print("=" * 70)

# ─── Step 1: Load Data ───────────────────────────────────────────────────────
print("\n📂 Step 1: Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Use available feature columns
available_features = [col for col in FEATURE_COLS if col in df.columns]
print(f"  Features: {available_features}")

# ─── Step 2: Prepare Features & Labels ────────────────────────────────────────
print("\n" + "─" * 70)
print("🔧 Step 2: Preparing Features & Labels")
print("─" * 70)

X = df[available_features].values
y = df['quality_label'].values  # 0=Excellent, 1=Good, 2=Poor, 3=Dangerous

class_names = ['Excellent', 'Good', 'Poor', 'Dangerous']
print(f"  Feature matrix shape: {X.shape}")
print(f"  Label distribution:")
for i, name in enumerate(class_names):
    count = (y == i).sum()
    print(f"    {name}: {count} ({count/len(y)*100:.1f}%)")

# ─── Step 3: Train/Test Split ────────────────────────────────────────────────
print("\n" + "─" * 70)
print("✂️  Step 3: Train/Test Split (80/20, stratified)")
print("─" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set:  {X_test.shape[0]} samples")

# ─── Step 4: Feature Scaling ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("⚖️  Step 4: Feature Scaling (StandardScaler)")
print("─" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler (teammates will need this for live predictions)
scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"  ✅ Scaler saved → {scaler_path}")

# Also save the feature column names
feature_meta = {
    'feature_columns': available_features,
    'class_names': class_names,
    'class_mapping': {'Excellent': 0, 'Good': 1, 'Poor': 2, 'Dangerous': 3}
}
joblib.dump(feature_meta, os.path.join(MODELS_DIR, 'feature_meta.joblib'))
print(f"  ✅ Feature metadata saved")

# ─── Step 5: Train Random Forest ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("🌲 Step 5: Training Random Forest Classifier")
print("─" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

print(f"\n  Random Forest Results:")
print(f"  {'─' * 40}")
print(f"  Accuracy:  {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, rf_pred, target_names=class_names, digits=3))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── Step 6: Train XGBoost ───────────────────────────────────────────────────
print("\n" + "─" * 70)
print("🚀 Step 6: Training XGBoost Classifier")
print("─" * 70)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')

print(f"\n  XGBoost Results:")
print(f"  {'─' * 40}")
print(f"  Accuracy:  {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
print(f"  F1-Score:  {xgb_f1:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=class_names, digits=3))

cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

# ─── Step 7: Compare & Save Best Model ───────────────────────────────────────
print("\n" + "─" * 70)
print("🏆 Step 7: Model Comparison")
print("─" * 70)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score (weighted)', '5-Fold CV Mean'],
    'Random Forest': [rf_accuracy, rf_f1, cv_scores.mean()],
    'XGBoost': [xgb_accuracy, xgb_f1, cv_scores_xgb.mean()]
})
print(comparison.to_string(index=False))

# Pick best model
if xgb_f1 >= rf_f1:
    best_model = xgb_model
    best_name = 'XGBoost'
    best_pred = xgb_pred
    best_proba = xgb_proba
    best_f1 = xgb_f1
    best_accuracy = xgb_accuracy
else:
    best_model = rf_model
    best_name = 'Random Forest'
    best_pred = rf_pred
    best_proba = rf_proba
    best_f1 = rf_f1
    best_accuracy = rf_accuracy

print(f"\n  🏆 Best Model: {best_name} (F1: {best_f1:.4f})")

# Save best model
model_path = os.path.join(MODELS_DIR, 'classifier.joblib')
joblib.dump(best_model, model_path)
print(f"  ✅ Best model saved → {model_path}")

# Also save both models for comparison
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_classifier.joblib'))
joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_classifier.joblib'))

# ─── Step 8: Confusion Matrices ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("📊 Step 8: Generating Visualization Plots")
print("─" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest confusion matrix
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title(f'Random Forest\nAccuracy: {rf_accuracy:.3f} | F1: {rf_f1:.3f}',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# XGBoost confusion matrix
cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title(f'XGBoost\nAccuracy: {xgb_accuracy:.3f} | F1: {xgb_f1:.3f}',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('Confusion Matrices — Model Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved confusion matrices → {cm_path}")

# ─── Step 9: Feature Importance ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], 
             color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Random Forest — Feature Importance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance')

# XGBoost feature importance
xgb_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'],
             color='forestgreen', edgecolor='black', alpha=0.8)
axes[1].set_title('XGBoost — Feature Importance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.suptitle('Which Parameters Matter Most?', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fi_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved feature importance → {fi_path}")

# ─── Step 10: Model Comparison Bar Chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.35

rf_scores = [rf_accuracy, rf_f1, cv_scores.mean()]
xgb_scores = [xgb_accuracy, xgb_f1, cv_scores_xgb.mean()]
metrics = ['Accuracy', 'F1-Score', 'CV Mean']

bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='forestgreen', alpha=0.85)

ax.set_ylabel('Score')
ax.set_title('Model Comparison — Random Forest vs XGBoost', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.15)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
comp_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
plt.savefig(comp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved model comparison → {comp_path}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CLASSIFICATION TRAINING COMPLETE ✅")
print("=" * 70)
print(f"""
  🏆 Best Model: {best_name}
  📊 Accuracy:   {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
  📊 F1-Score:   {best_f1:.4f}
  
  📁 Saved Models:
     {os.path.join(MODELS_DIR, 'classifier.joblib')} (best)
     {os.path.join(MODELS_DIR, 'scaler.joblib')}
     {os.path.join(MODELS_DIR, 'feature_meta.joblib')}
  
  📁 Plots:
     {cm_path}
     {fi_path}
     {comp_path}
  
  Next step: Run 03_anomaly_detection.py
""")
