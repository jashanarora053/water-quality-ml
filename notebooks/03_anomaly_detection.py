"""
===============================================================================
 03_anomaly_detection.py — Water Quality Anomaly / Pollution Detection
===============================================================================
 Trains an Isolation Forest model to detect abnormal water quality readings
 that may indicate contamination or sensor faults.
 
 What this script does:
   1. Loads cleaned dataset
   2. Trains Isolation Forest on "normal" data
   3. Validates by injecting synthetic anomalies
   4. Visualizes normal vs anomaly readings
   5. Saves trained model
 
 Prerequisites:
   Run 01_eda.py first!
 
 Usage:
   python notebooks/03_anomaly_detection.py
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join('data', 'processed', 'water_quality_cleaned.csv')
MODELS_DIR = 'models'
PLOTS_DIR = os.path.join('notebooks', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Feature columns (same as classification)
FEATURE_COLS = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

print("=" * 70)
print("  WATER QUALITY — ANOMALY DETECTION (ISOLATION FOREST)")
print("=" * 70)

# ─── Step 1: Load Data ───────────────────────────────────────────────────────
print("\n📂 Step 1: Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
available_features = [col for col in FEATURE_COLS if col in df.columns]
print(f"  Shape: {df.shape}")
print(f"  Features: {available_features}")

X = df[available_features].values

# ─── Step 2: Scale Features ──────────────────────────────────────────────────
print("\n" + "─" * 70)
print("⚖️  Step 2: Scaling Features")
print("─" * 70)

# Load the scaler from classification (for consistency)
scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("  ✅ Loaded existing scaler from classification model")
else:
    scaler = StandardScaler()
    scaler.fit(X)
    print("  ⚠️  Created new scaler (classification scaler not found)")

X_scaled = scaler.transform(X)

# ─── Step 3: Train Isolation Forest ──────────────────────────────────────────
print("\n" + "─" * 70)
print("🌲 Step 3: Training Isolation Forest")
print("─" * 70)

# Contamination = expected fraction of anomalies
# Start with 5%, tune later based on domain knowledge
CONTAMINATION = 0.05

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples='auto',
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_scaled)
print(f"  ✅ Model trained with contamination={CONTAMINATION}")
print(f"  Parameters: n_estimators=200, max_samples='auto'")

# Get predictions on training data
predictions = iso_forest.predict(X_scaled)  # 1 = normal, -1 = anomaly
anomaly_scores = iso_forest.decision_function(X_scaled)  # Lower = more anomalous

n_normal = (predictions == 1).sum()
n_anomaly = (predictions == -1).sum()

print(f"\n  Results on training data:")
print(f"    Normal readings:  {n_normal:5d} ({n_normal/len(predictions)*100:.1f}%)")
print(f"    Anomalies found:  {n_anomaly:5d} ({n_anomaly/len(predictions)*100:.1f}%)")

# ─── Step 4: Analyze Detected Anomalies ──────────────────────────────────────
print("\n" + "─" * 70)
print("🔍 Step 4: Analyzing Detected Anomalies")
print("─" * 70)

df_analysis = df[available_features].copy()
df_analysis['anomaly'] = predictions
df_analysis['anomaly_score'] = anomaly_scores

# Compare normal vs anomaly statistics
print("\n  Normal readings — Mean values:")
normal_stats = df_analysis[df_analysis['anomaly'] == 1][available_features].mean()
for feat in available_features:
    print(f"    {feat:25s}: {normal_stats[feat]:.2f}")

print("\n  Anomalous readings — Mean values:")
anomaly_stats = df_analysis[df_analysis['anomaly'] == -1][available_features].mean()
for feat in available_features:
    diff_pct = ((anomaly_stats[feat] - normal_stats[feat]) / normal_stats[feat]) * 100
    arrow = "↑" if diff_pct > 0 else "↓"
    print(f"    {feat:25s}: {anomaly_stats[feat]:.2f}  ({arrow}{abs(diff_pct):.1f}% vs normal)")

# ─── Step 5: Validation with Synthetic Anomalies ─────────────────────────────
print("\n" + "─" * 70)
print("🧪 Step 5: Validation — Injecting Synthetic Anomalies")
print("─" * 70)

# Create obviously anomalous readings
synthetic_anomalies = pd.DataFrame({
    'ph': [2.0, 12.5, 1.0, 14.0, 3.0],
    'Hardness': [500, 10, 600, 5, 800],
    'Solids': [50000, 100, 80000, 50, 60000],
    'Chloramines': [15, 0, 20, 0.1, 18],
    'Sulfate': [600, 10, 700, 5, 800],
    'Conductivity': [800, 50, 900, 30, 1000],
    'Organic_carbon': [25, 1, 30, 0.5, 28],
    'Trihalomethanes': [120, 1, 130, 0.5, 100],
    'Turbidity': [15, 0.1, 20, 0.05, 18],
})

# Only use available features
syn_available = synthetic_anomalies[[c for c in available_features if c in synthetic_anomalies.columns]]
syn_scaled = scaler.transform(syn_available.values)
syn_predictions = iso_forest.predict(syn_scaled)
syn_scores = iso_forest.decision_function(syn_scaled)

print("\n  Synthetic anomaly detection results:")
print(f"  {'Scenario':<30s} {'Predicted':<12s} {'Score':<10s} {'Correct?':<10s}")
print(f"  {'─'*62}")

scenarios = ['Extreme low pH (2.0)', 'Extreme high pH (12.5)', 
             'Very low pH (1.0)', 'Very high pH (14.0)', 'Low pH (3.0)']

detected = 0
for i, (pred, score, scenario) in enumerate(zip(syn_predictions, syn_scores, scenarios)):
    label = "ANOMALY" if pred == -1 else "NORMAL"
    correct = "✅" if pred == -1 else "❌"
    if pred == -1:
        detected += 1
    print(f"  {scenario:<30s} {label:<12s} {score:<10.4f} {correct}")

detection_rate = detected / len(syn_predictions) * 100
print(f"\n  Synthetic anomaly detection rate: {detected}/{len(syn_predictions)} ({detection_rate:.0f}%)")

if detection_rate >= 80:
    print("  ✅ Model successfully detects extreme anomalies!")
else:
    print("  ⚠️  Model may need tuning — try lower contamination value")

# Also test with normal-looking data
print("\n  Testing with normal readings:")
normal_test = pd.DataFrame({
    'ph': [7.0, 7.5, 6.8],
    'Hardness': [200, 180, 210],
    'Solids': [20000, 25000, 18000],
    'Chloramines': [7, 6.5, 7.2],
    'Sulfate': [300, 280, 320],
    'Conductivity': [400, 420, 380],
    'Organic_carbon': [14, 13, 15],
    'Trihalomethanes': [60, 55, 65],
    'Turbidity': [4, 3.5, 4.2],
})

norm_available = normal_test[[c for c in available_features if c in normal_test.columns]]
norm_scaled = scaler.transform(norm_available.values)
norm_predictions = iso_forest.predict(norm_scaled)

normal_correct = (norm_predictions == 1).sum()
print(f"  Normal readings correctly identified: {normal_correct}/{len(norm_predictions)}")

# ─── Step 6: Visualization ───────────────────────────────────────────────────
print("\n" + "─" * 70)
print("📊 Step 6: Generating Visualizations")
print("─" * 70)

# Plot 1: Anomaly score distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram of anomaly scores
axes[0].hist(anomaly_scores[predictions == 1], bins=50, alpha=0.7, 
             color='#2ECC71', label='Normal', edgecolor='black')
axes[0].hist(anomaly_scores[predictions == -1], bins=50, alpha=0.7,
             color='#E74C3C', label='Anomaly', edgecolor='black')
axes[0].axvline(0, color='black', linestyle='--', linewidth=2, label='Decision boundary')
axes[0].set_xlabel('Anomaly Score', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

# Pie chart of normal vs anomaly
colors_pie = ['#2ECC71', '#E74C3C']
axes[1].pie([n_normal, n_anomaly], labels=['Normal', 'Anomaly'],
            colors=colors_pie, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            explode=(0, 0.1))
axes[1].set_title('Normal vs Anomaly Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
dist_path = os.path.join(PLOTS_DIR, 'anomaly_distribution.png')
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved anomaly distribution → {dist_path}")

# Plot 2: Scatter plots showing anomalies across parameter pairs
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Anomaly Detection — Parameter Scatter Plots', fontsize=16, fontweight='bold', y=1.02)

param_pairs = [
    (0, 2),  # ph vs Solids
    (0, 8),  # ph vs Turbidity
    (2, 5),  # Solids vs Conductivity
    (0, 5),  # ph vs Conductivity
    (2, 8),  # Solids vs Turbidity
    (3, 4),  # Chloramines vs Sulfate
]

for idx, (i, j) in enumerate(param_pairs):
    if i < len(available_features) and j < len(available_features):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        
        # Plot normal points
        mask_normal = predictions == 1
        mask_anomaly = predictions == -1
        
        ax.scatter(X[mask_normal, i], X[mask_normal, j], 
                   c='#2ECC71', alpha=0.3, s=10, label='Normal')
        ax.scatter(X[mask_anomaly, i], X[mask_anomaly, j],
                   c='#E74C3C', alpha=0.8, s=30, marker='x', label='Anomaly')
        ax.set_xlabel(available_features[i], fontsize=10)
        ax.set_ylabel(available_features[j], fontsize=10)
        ax.legend(fontsize=8)
    else:
        row, col = divmod(idx, 3)
        axes[row][col].set_visible(False)

plt.tight_layout()
scatter_path = os.path.join(PLOTS_DIR, 'anomaly_scatter.png')
plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved anomaly scatter plots → {scatter_path}")

# Plot 3: Box plots comparing normal vs anomaly for each parameter
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Parameter Distributions: Normal vs Anomaly', fontsize=16, fontweight='bold', y=1.02)

for idx, feat in enumerate(available_features):
    if idx < 9:
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        
        data_plot = pd.DataFrame({
            'Value': df_analysis[feat],
            'Type': df_analysis['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        })
        
        sns.boxplot(data=data_plot, x='Type', y='Value', ax=ax,
                    palette={'Normal': '#2ECC71', 'Anomaly': '#E74C3C'})
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.set_xlabel('')

for idx in range(len(available_features), 9):
    row, col = divmod(idx, 3)
    axes[row][col].set_visible(False)

plt.tight_layout()
box_path = os.path.join(PLOTS_DIR, 'anomaly_boxplots.png')
plt.savefig(box_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved anomaly box plots → {box_path}")

# ─── Step 7: Save Model ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("💾 Step 7: Saving Anomaly Detection Model")
print("─" * 70)

model_path = os.path.join(MODELS_DIR, 'anomaly_detector.joblib')
joblib.dump(iso_forest, model_path)
print(f"  ✅ Model saved → {model_path}")

# Save model metadata
anomaly_meta = {
    'model_type': 'IsolationForest',
    'contamination': CONTAMINATION,
    'n_estimators': 200,
    'feature_columns': available_features,
    'decision_threshold': 0,  # scores below 0 = anomaly
    'training_samples': len(X),
    'anomalies_detected': int(n_anomaly),
    'synthetic_detection_rate': detection_rate
}
joblib.dump(anomaly_meta, os.path.join(MODELS_DIR, 'anomaly_meta.joblib'))
print(f"  ✅ Model metadata saved")

# ─── Step 8: How to Use (for teammates) ──────────────────────────────────────
print("\n" + "─" * 70)
print("📖 Step 8: Usage Guide")
print("─" * 70)
print("""
  # How to detect anomalies in new data:
  
  import joblib
  import numpy as np
  
  # Load model and scaler
  model = joblib.load('models/anomaly_detector.joblib')
  scaler = joblib.load('models/scaler.joblib')
  
  # New sensor reading
  reading = np.array([[7.2, 200, 20000, 7.0, 300, 400, 14, 60, 4]])
  
  # Scale and predict
  reading_scaled = scaler.transform(reading)
  prediction = model.predict(reading_scaled)     # 1=Normal, -1=Anomaly
  score = model.decision_function(reading_scaled) # Lower = more anomalous
  
  if prediction[0] == -1:
      print(f"🚨 ANOMALY DETECTED! Score: {score[0]:.4f}")
  else:
      print(f"✅ Normal reading. Score: {score[0]:.4f}")
""")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ANOMALY DETECTION TRAINING COMPLETE ✅")
print("=" * 70)
print(f"""
  🌲 Model: Isolation Forest
  📊 Contamination: {CONTAMINATION} ({CONTAMINATION*100:.0f}%)
  📊 Anomalies found in training data: {n_anomaly} ({n_anomaly/len(predictions)*100:.1f}%)
  📊 Synthetic anomaly detection rate: {detection_rate:.0f}%
  
  📁 Saved:
     {model_path}
     {os.path.join(MODELS_DIR, 'anomaly_meta.joblib')}
  
  📁 Plots:
     {dist_path}
     {scatter_path}
     {box_path}
  
  Next step: Run 04_forecasting.py
""")
