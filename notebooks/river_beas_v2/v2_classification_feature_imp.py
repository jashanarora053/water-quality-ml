import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading model and metadata...")

# 1. Paths to your saved files
save_dir = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\Models\river_beas_v2"
model_path = os.path.join(save_dir, 'xgboost_beas_model.joblib')
meta_path = os.path.join(save_dir, 'beas_feature_meta.joblib')

# 2. Load the XGBoost model and the feature names
xgb_boat_model = joblib.load(model_path)
feature_meta = joblib.load(meta_path)
feature_names = feature_meta['feature_columns']

# 3. Extract the feature importance scores
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_boat_model.feature_importances_
})

# Sort from lowest to highest for the horizontal bar chart
importance_df = importance_df.sort_values('Importance', ascending=False)

# 4. Create a beautiful plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)

# Add styling
plt.title('Which Sensors Matter Most? (River Beas Data)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score (Higher = More Important)', fontsize=12)
plt.ylabel('Sensor Parameter', fontsize=12)

# Save the plot to your folder
plot_path = os.path.join(save_dir, 'feature_importance_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')

plt.tight_layout()
plt.show()

print(f"Plot saved successfully to: {plot_path}")
