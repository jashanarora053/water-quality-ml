import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  🌊 RIVER BEAS — ANOMALY DETECTION (ISOLATION FOREST)")
print("=" * 70)

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\cleaned_beas_data.csv"
MODELS_DIR = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\Models\river_beas_v2"
PLOTS_DIR = os.path.join(MODELS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Step 1: Load Data & Metadata ────────────────────────────────────────────
print("\nStep 1: Loading cleaned dataset and metadata...")

# We MUST use the exact same features the classification model uses
meta_path = os.path.join(MODELS_DIR, 'beas_feature_meta.joblib')
if not os.path.exists(meta_path):
    raise FileNotFoundError("Classification metadata not found! Run the classification script first.")

feature_meta = joblib.load(meta_path)
FEATURE_COLS = feature_meta['feature_columns']

df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS].values

print(f"  Shape: {df.shape[0]} rows")
print(f"  Monitoring Sensors: {FEATURE_COLS}")

# ─── Step 2: Scale Features ──────────────────────────────────────────────────
print("\n" + "─" * 70)
print("⚖️  Step 2: Scaling Features (Matching XGBoost Scaler)")
print("─" * 70)

scaler_path = os.path.join(MODELS_DIR, 'beas_scaler.joblib')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("  ✅ Successfully loaded the boat's master scaler.")
else:
    raise FileNotFoundError("Scaler not found! The Anomaly Detector must use the exact same scaler as the Classifier.")

X_scaled = scaler.transform(X)

# ─── Step 3: Train Isolation Forest ──────────────────────────────────────────
print("\n" + "─" * 70)
print("🌲 Step 3: Training Isolation Forest")
print("─" * 70)

# We assume about 2% of the historical Beas river data contains weird spikes
CONTAMINATION = 0.02

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_scaled)
print(f"  ✅ Model trained with contamination={CONTAMINATION}")

# 1 = normal, -1 = anomaly
predictions = iso_forest.predict(X_scaled)  
anomaly_scores = iso_forest.decision_function(X_scaled)  

n_normal = (predictions == 1).sum()
n_anomaly = (predictions == -1).sum()

print(f"\n  Results on historical Beas data:")
print(f"    Normal readings:  {n_normal:5d} ({n_normal/len(predictions)*100:.1f}%)")
print(f"    Anomalies found:  {n_anomaly:5d} ({n_anomaly/len(predictions)*100:.1f}%)")

# ─── Step 4: Validation with Synthetic River Disasters ───────────────────────
print("\n" + "─" * 70)
print("🧪 Step 4: Validation — Injecting Synthetic River Disasters")
print("─" * 70)

# Let's test if the model can catch physical impossibilities
# Columns: ['pH', 'TURBIDITY(NTU)', 'COND.(µS/cm)', 'DO(mg/l)', 'CL(mg/l)']
synthetic_anomalies = pd.DataFrame({
    'pH':             [2.0,  13.0, 7.2,   7.0,  7.5],
    'TURBIDITY(NTU)': [10.0, 15.0, 5000.0, 2.0,  5.0],
    'COND.(µS/cm)':   [300.0, 350.0, 400.0, 8000.0, 350.0],
    'DO(mg/l)':       [6.5,  7.0,  6.8,   6.5,  0.1],
    'CL(mg/l)':       [15.0, 18.0, 20.0,  250.0, 12.0]
})

scenarios = [
    'Acid Spill (pH 2.0)', 
    'Bleach Dump (pH 13.0)', 
    'Massive Mudslide (Turbidity 5000)', 
    'Invisible Solvent (Cond 8000, Clear Water)', 
    'Dead Zone (DO 0.1 mg/L)'
]

syn_scaled = scaler.transform(synthetic_anomalies.values)
syn_predictions = iso_forest.predict(syn_scaled)
syn_scores = iso_forest.decision_function(syn_scaled)

print(f"  {'Disaster Scenario':<45s} {'Predicted':<12s} {'Score':<10s} {'Caught?':<10s}")
print(f"  {'─'*77}")

detected = 0
for pred, score, scenario in zip(syn_predictions, syn_scores, scenarios):
    label = "ANOMALY" if pred == -1 else "NORMAL"
    correct = "✅" if pred == -1 else "❌"
    if pred == -1: detected += 1
    print(f"  {scenario:<45s} {label:<12s} {score:<10.4f} {correct}")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


print("RIVER BEAS — ANOMALY DETECTION (TWO-LAYER DEFENSE)")


# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\cleaned_beas_data.csv"
MODELS_DIR = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\Models\river_beas_v2"
PLOTS_DIR = os.path.join(MODELS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── LAYER 1: Rule-Based Thresholds (CPCB/BIS Standards) ─────────────────────
# These are hard scientific limits. If ANY sensor crosses these, it is
# GUARANTEED to be an anomaly — no AI needed.
RULE_THRESHOLDS = {
    'pH':             {'min': 4.0,  'max': 11.0},   # Normal river: 6.5-8.5
    'TURBIDITY(NTU)': {'min': 0.0,  'max': 1000.0},  # Extreme mudslide threshold
    'COND.(µS/cm)':   {'min': 0.0,  'max': 3000.0},  # Extreme dissolved solids
    'DO(mg/l)':       {'min': 1.0,  'max': 20.0},    # Below 1 = dead zone
    'CL(mg/l)':       {'min': 0.0,  'max': 600.0},   # BIS limit for chlorides
}

def rule_based_check(row, thresholds):
    """
    Layer 1: Hard rule-based anomaly detection.
    Returns (is_anomaly: bool, reason: str)
    """
    for feature, limits in thresholds.items():
        value = row[feature]
        if value < limits['min']:
            return True, f"{feature} = {value} (Below minimum {limits['min']})"
        if value > limits['max']:
            return True, f"{feature} = {value} (Above maximum {limits['max']})"
    return False, "All readings within safe limits"

# ─── Step 1: Load Data & Metadata ────────────────────────────────────────────
print("Step 1: Loading cleaned dataset and metadata...")

meta_path = os.path.join(MODELS_DIR, 'beas_feature_meta.joblib')
if not os.path.exists(meta_path):
    raise FileNotFoundError("Classification metadata not found! Run the classification script first.")

feature_meta = joblib.load(meta_path)
FEATURE_COLS = feature_meta['feature_columns']

df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS].values

print(f"  Shape: {df.shape[0]} rows")
print(f"  Monitoring Sensors: {FEATURE_COLS}")

# ─── Step 2: Scale Features ──────────────────────────────────────────────────
print("Step 2: Scaling Features (Matching XGBoost Scaler)")


scaler_path = os.path.join(MODELS_DIR, 'beas_scaler.joblib')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("  Successfully loaded the boat's master scaler.")
else:
    raise FileNotFoundError("Scaler not found!")

X_scaled = scaler.transform(X)

# ─── Step 3: Train Isolation Forest (Layer 2) ────────────────────────────────

print("Step 3: Training Isolation Forest (Layer 2 — Unknown Anomalies)")

CONTAMINATION = 0.05

iso_forest = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    max_samples=4096,
    max_features=3,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_scaled)
print(f"  Model trained with contamination={CONTAMINATION}")

# Get predictions on training data
predictions = iso_forest.predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

n_normal = (predictions == 1).sum()
n_anomaly = (predictions == -1).sum()

print(f"\n  Results on historical Beas data:")
print(f"    Normal readings:  {n_normal:5d} ({n_normal/len(predictions)*100:.1f}%)")
print(f"    Anomalies found:  {n_anomaly:5d} ({n_anomaly/len(predictions)*100:.1f}%)")

# ─── Step 4: Validation — Two-Layer Defense Test ─────────────────────────────
print("\n" + "─" * 70)
print("Step 4: Validation — Testing Both Layers Against Disasters")
print("─" * 70)

# Synthetic disaster scenarios
synthetic_anomalies = pd.DataFrame({
    'pH':             [2.0,   13.0,  7.2,    7.0,    7.5,   7.0,    6.8],
    'TURBIDITY(NTU)': [10.0,  15.0,  5000.0, 2.0,    5.0,   400.0,  12.0],
    'COND.(µS/cm)':   [300.0, 350.0, 400.0,  8000.0, 350.0, 1800.0, 350.0],
    'DO(mg/l)':       [6.5,   7.0,   6.8,    6.5,    0.1,   6.0,    6.5],
    'CL(mg/l)':       [15.0,  18.0,  20.0,   250.0,  12.0,  0.5,    15.0]
})

scenarios = [
    'Acid Spill (pH 2.0)',
    'Bleach Dump (pH 13.0)',
    'Massive Mudslide (Turbidity 5000)',
    'Invisible Solvent (Cond 8000)',
    'Dead Zone (DO 0.1 mg/L)',
    'Unknown Chemical (High Cond + Low CL)',
    'Normal Water (Control Test)'
]

# Expected results (True = should be anomaly, False = should be normal)
expected = [True, True, True, True, True, True, False]

syn_scaled = scaler.transform(synthetic_anomalies.values)
iso_predictions = iso_forest.predict(syn_scaled)
iso_scores = iso_forest.decision_function(syn_scaled)

print(f"\n  {'Scenario':<40s} {'Layer 1':<12s} {'Layer 2':<12s} {'FINAL':<12s} {'Correct?'}")
print(f"  {'─'*88}")

total_correct = 0
for i, scenario in enumerate(scenarios):
    row = synthetic_anomalies.iloc[i]

    # Layer 1: Rule-based check
    l1_anomaly, l1_reason = rule_based_check(row, RULE_THRESHOLDS)
    l1_label = "⛔ RULE HIT" if l1_anomaly else "✅ PASS"

    # Layer 2: Isolation Forest check
    l2_anomaly = iso_predictions[i] == -1
    l2_label = "⛔ ANOMALY" if l2_anomaly else "✅ NORMAL"

    # FINAL VERDICT: Anomaly if EITHER layer flags it
    final_anomaly = l1_anomaly or l2_anomaly
    final_label = "🚨 ANOMALY" if final_anomaly else "✅ NORMAL"

    # Check correctness
    correct = final_anomaly == expected[i]
    correct_icon = "✅" if correct else "❌"
    if correct: total_correct += 1

    print(f"  {scenario:<40s} {l1_label:<12s} {l2_label:<12s} {final_label:<12s} {correct_icon}")

print(f"\n  Detection Rate: {total_correct}/{len(scenarios)} ({total_correct/len(scenarios)*100:.0f}%)")

# Show detailed Layer 1 reasons
print(f"\n  Layer 1 (Rule-Based) Detailed Breakdown:")
print(f"  {'─'*60}")
for i, scenario in enumerate(scenarios):
    row = synthetic_anomalies.iloc[i]
    l1_anomaly, l1_reason = rule_based_check(row, RULE_THRESHOLDS)
    icon = "⛔" if l1_anomaly else "✅"
    print(f"  {icon} {scenario:<40s} → {l1_reason}")

# ─── Step 5: Visualization ───────────────────────────────────────────────────
print("\n" + "─" * 70)
print("📊 Step 5: Generating Visualizations")
print("─" * 70)

df_plot = pd.DataFrame(X, columns=FEATURE_COLS)
df_plot['Anomaly'] = predictions

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Two-Layer Anomaly Detection: River Beas', fontsize=16, fontweight='bold')

# Plot 1: pH vs Conductivity
sns.scatterplot(data=df_plot, x='pH', y='COND.(µS/cm)', hue='Anomaly',
                palette={1: '#2ECC71', -1: '#E74C3C'}, alpha=0.6, ax=axes[0])
axes[0].set_title('pH vs Conductivity')

# Plot 2: Turbidity vs Conductivity
sns.scatterplot(data=df_plot, x='TURBIDITY(NTU)', y='COND.(µS/cm)', hue='Anomaly',
                palette={1: '#2ECC71', -1: '#E74C3C'}, alpha=0.6, ax=axes[1])
axes[1].set_title('Turbidity vs Conductivity')

# Plot 3: DO vs pH
sns.scatterplot(data=df_plot, x='DO(mg/l)', y='pH', hue='Anomaly',
                palette={1: '#2ECC71', -1: '#E74C3C'}, alpha=0.6, ax=axes[2])
axes[2].set_title('Dissolved Oxygen vs pH')

plt.tight_layout()
scatter_path = os.path.join(PLOTS_DIR, 'beas_anomaly_scatter.png')
plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved anomaly scatter plots!")

# ─── Step 6: Save Models ─────────────────────────────────────────────────────
print(" Step 6: Saving Anomaly Detection Models")


# Save the Isolation Forest model
model_path = os.path.join(MODELS_DIR, 'anomaly_detector_beas.joblib')
joblib.dump(iso_forest, model_path)
print(f"  Isolation Forest model saved → {model_path}")

# Save the rule-based thresholds (so the API can use them too)
rules_path = os.path.join(MODELS_DIR, 'anomaly_rules.joblib')
joblib.dump(RULE_THRESHOLDS, rules_path)
print(f" Rule-based thresholds saved → {rules_path}")

# Save metadata
anomaly_meta = {
    'model_type': 'TwoLayerDefense',
    'layer_1': 'Rule-Based (CPCB/BIS Thresholds)',
    'layer_2': 'IsolationForest',
    'contamination': CONTAMINATION,
    'n_estimators': 300,
    'max_samples': 4096,
    'max_features': 3,
    'feature_columns': FEATURE_COLS,
    'training_samples': len(X),
}
joblib.dump(anomaly_meta, os.path.join(MODELS_DIR, 'anomaly_meta.joblib'))
print(f"  Anomaly metadata saved")


print(" TWO-LAYER ANOMALY DETECTION COMPLETE!")

print(f"""
    Layer 1 (Rules):  Catches known disasters instantly (pH, DO, etc.)
    Layer 2 (AI):     Catches unknown multi-dimensional anomalies
  
   Validation: {total_correct}/{len(scenarios)} scenarios correctly handled
  
      Saved Files:
     {model_path}
     {rules_path}
""")
