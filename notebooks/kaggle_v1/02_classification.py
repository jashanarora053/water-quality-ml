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
BASE_DIR = r'C:\Users\omen\.gemini\antigravity-ide\scratch\water-quality-ml'
DATA_PATH = os.path.join(BASE_DIR,'data', 'processed', 'water_quality_cleaned.csv')
MODELS_DIR =  os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR,'notebooks', 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Feature columns (all numeric sensor-type parameters)
FEATURE_COLS = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# ─── Step 1: Load Data ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Shape:{df.shape}")

# Use available feature columns
available_features = [col for col in FEATURE_COLS if col in df.columns]
print(f"Features:{available_features}")

# ─── Step 2: Prepare Features & Labels ────────────────────────────────────────
print("Step 2:Preparing Features & Labels")

X = df[available_features].values
y = df['quality_label'].values 
class_names = ['Excellent', 'Good', 'Poor', 'Dangerous']

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution:")
for i, name in enumerate(class_names):
    count = (y == i).sum()
    print(f"{name}:{count} ({count/len(y)*100:.1f}%)")

# ─── Step 3: Train/Test Split ────────────────────────────────────────────────
print("Step 3: Train/Test Split (80/20, stratified)")
# Keep only classes with at least 2 samples

# Find classes with at least 2 samples
classes, counts = np.unique(y, return_counts=True)
valid_classes = classes[counts >= 2]

# Create a boolean mask of valid samples
mask = np.isin(y, valid_classes)

# Filter X and y
X = X[mask]
y = y[mask]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# ─── Step 4: Feature Scaling ─────────────────────────────────────────────────
print("Step 4: Feature Scaling (StandardScaler)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f" Scaler saved → {scaler_path}")

# Also save the feature column names
feature_meta = {
    'feature_columns': available_features,
    'class_names': class_names,
    'class_mapping': {'Excellent': 0, 'Good': 1, 'Poor': 2, 'Dangerous': 3}
}
joblib.dump(feature_meta, os.path.join(MODELS_DIR, 'feature_meta.joblib'))
print(f"Feature metadata saved")

'''
Random Forest works by creating multiple decision trees (in our case, 200) and combining their results to make a
final prediction. Each decision tree is trained on a random subset of the data and a random subset of the features,
so every tree learns slightly different patterns.When a new water sample comes in, all 200 trees independently predict
the water quality class,and the final answer is decided by majority voting.
Random Forest handles class imbalance well, works efficiently with multiple parameters,
and is resistant to overfitting, making it one of the most widely used algorithms for classification tasks.
'''

# ─── Step 5: Train Random Forest ─────────────────────────────────────────────
print("Step 5: Training Random Forest Classifier")

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
print(classification_report(y_test, rf_pred, digits=3))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


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
print(classification_report(y_test, xgb_pred, digits=3))

cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

# ─── Step 7: Compare & Save Best Model ───────────────────────────────────────
print("Step 7: Model Comparison")

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

print(f"\n Best Model: {best_name} (F1: {best_f1:.4f})")

# Save best model
model_path = os.path.join(MODELS_DIR, 'classifier.joblib')
joblib.dump(best_model, model_path)
print(f"Best model saved → {model_path}")

# Also save both models for comparison
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_classifier.joblib'))
joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_classifier.joblib'))

"""
What is a Confusion Matrix?
Accuracy tells you "91% correct" — but it doesn't tell you WHERE the model is making mistakes.
A confusion matrix shows you exactly that.
It's a table that answers: "For each class, what did the model actually predict?"

How to read it:
Diagonal (top-left to bottom-right) = Correct predictions ✅
Off-diagonal = Mistakes ❌
"""

# ─── Step 8: Confusion Matrices ──────────────────────────────────────────────
print("Step 8: Generating Visualization Plots")

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
plt.show()
plt.close()
print(f" Saved confusion matrices → {cm_path}")

"""
Feature Importance tells you which parameters (features) the model relied on the most to make its decisions.
When XGBoost or Random Forest builds its 200 trees, it keeps track of every time it uses a parameter (like pH)to split a branch.
  ->  If it uses pH frequently, and splitting by pH results in highly accurate branches, then pH gets a high "importance score."
  ->  If it never uses a parameter (like Sulfate), that parameter gets a score near 0.

Why Do We Need It? 

1. Saving Money on Hardware (Sensors)
We can invest on sensors based on importnace of their readings

2. Validating the Model (Sanity Check)
If feature importance for water potability matches with real world hydrology, that means model is actually learning.

3. Explaination to logic
Machine Learning is often criticized as a "Black Box" (data goes in, answers come out, nobody knows why).
Feature importance makes your model a "White Box." You can confidently say: "Our model achieved 98% accuracy primarily by looking at pH and Turbidity levels, which matches WHO guidelines."
That sounds much better than "The computer just figured it out."
"""

# ─── Step 9: Feature Importance ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rf_importance = pd.DataFrame({'Feature': available_features, 'Importance': rf_model.feature_importances_})
xgb_importance = pd.DataFrame({'Feature': available_features, 'Importance': xgb_model.feature_importances_})

xgb_importance = xgb_importance.sort_values('Importance', ascending=True)
rf_importance = rf_importance.set_index('Feature').reindex(xgb_importance['Feature']).reset_index()

# Plot Random Forest
axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], 
             color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Random Forest — Feature Importance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance')

# Plot XGBoost
axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'],
             color='forestgreen', edgecolor='black', alpha=0.8)
axes[1].set_title('XGBoost — Feature Importance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.suptitle('Which Parameters Matter Most?', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fi_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved feature importance → {fi_path}")

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
plt.show()
plt.close()
print(f"Saved model comparison → {comp_path}")


# ─── Step 9: Feature Importance ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rf_importance = pd.DataFrame({'Feature': available_features, 'Importance': rf_model.feature_importances_})
xgb_importance = pd.DataFrame({'Feature': available_features, 'Importance': xgb_model.feature_importances_})

xgb_importance = xgb_importance.sort_values('Importance', ascending=True)
rf_importance = rf_importance.set_index('Feature').reindex(xgb_importance['Feature']).reset_index()

# Plot Random Forest
axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], 
             color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Random Forest — Feature Importance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance')

# Plot XGBoost
axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'],
             color='forestgreen', edgecolor='black', alpha=0.8)
axes[1].set_title('XGBoost — Feature Importance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.suptitle('Which Parameters Matter Most?', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fi_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved feature importance → {fi_path}")
