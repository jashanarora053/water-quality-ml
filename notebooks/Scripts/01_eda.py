import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

BASE_DIR = r'C:\Users\omen\.gemini\antigravity-ide\scratch\water-quality-ml'
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'water_potability.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'water_quality_cleaned.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'notebooks', 'plots')

COLUMN_MAPPING = {
    'ph': 'pH',
    'Hardness': 'hardness',
    'Solids': 'tds',
    'Chloramines': 'chloramines',
    'Sulfate': 'sulfate',
    'Conductivity': 'conductivity',
    'Organic_carbon': 'organic_carbon',
    'Trihalomethanes': 'trihalomethanes',
    'Turbidity': 'turbidity',
    'Potability': 'potability'
}

sensor_params = ['pH', 'tds', 'turbidity', 'conductivity']
# Note: This Kaggle dataset doesn't have temperature & DO directly,
# but has related water quality params. We'll work with what's available
# and our labeling system will be based on these + additional params.

# ─── Step 1: Load Dataset ────────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA_PATH)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# ─── Step 2: Basic Inspection ────────────────────────────────────────────────
print("\n  Data Types:")
for col in df.columns:
    print(f" {col:25s} → {df[col].dtype}")

print("\n  First 5 rows:")
print(df.head().to_string(index=False))

print("\n  Statistical Summary:")
print(df.describe().round(2).to_string())

# ─── Step 3: Missing Values ──────────────────────────────────────────────────
# Using median Imputation for handleing missing values 
# It's the standard approach for water quality datasets

print("Missing Values Analysis")

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0].to_string())

total_missing = df.isnull().sum().sum()
if total_missing > 0:
    print(f"\n  Total missing values: {total_missing}")
    print("  Strategy: Filling with MEDIAN (robust to outliers)")
    df_clean = df.fillna(df.median(numeric_only=True))
    print(" Missing values filled with median")
else:
    df_clean = df.copy()
    print(" No missing values found!")

# Verify no missing values remain
assert df_clean.isnull().sum().sum() == 0, "Still have missing values!"

# ─── Step 4: Distribution Plots ──────────────────────────────────────────────
print("Distribution Plots")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Distribution of Water Quality Parameters', fontsize=16, fontweight='bold', y=1.02)

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'Potability' in numeric_cols:
    numeric_cols.remove('Potability')

for idx, col in enumerate(numeric_cols):
    row, c = divmod(idx, 3)
    if row < 3 and c < 3:
        ax = axes[row][c]
        # Plot histogram with KDE
        sns.histplot(df_clean[col], kde=True, ax=ax, color='steelblue', alpha=0.7)
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.axvline(df_clean[col].mean(), color='red', linestyle='--', label=f'Mean: {df_clean[col].mean():.2f}')
        ax.axvline(df_clean[col].median(), color='green', linestyle='--', label=f'Median: {df_clean[col].median():.2f}')
        ax.legend(fontsize=8)

# Hide unused subplots
for idx in range(len(numeric_cols), 9):
    row, c = divmod(idx, 3)
    axes[row][c].set_visible(False)

plt.tight_layout()
dist_path = os.path.join(PLOTS_DIR, 'distributions.png')
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# ─── Step 5: Correlation Heatmap ─────────────────────────────────────────────
print("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df_clean.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdBu_r', center=0, square=True,
    linewidths=0.5, ax=ax,
    cbar_kws={'label': 'Correlation Coefficient'}
)
ax.set_title('Correlation Heatmap — Water Quality Parameters', fontsize=14, fontweight='bold')
plt.tight_layout()
corr_path = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
plt.savefig(corr_path, dpi=150, bbox_inches='tight')
plt.close()

# Print notable correlations
print("\n  Notable correlations (|r| > 0.3):")
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.3:
            print(f"    {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {r:.3f}")

if not any(abs(corr_matrix.iloc[i, j]) > 0.3 
           for i in range(len(corr_matrix.columns)) 
           for j in range(i + 1, len(corr_matrix.columns))):
    print("    None found (all correlations are weak — good for independent features!)")

# ─── Step 6: Box Plots (Outlier Detection) ───────────────────────────────────
print("Box Plots for Outlier Detection")


fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Box Plots — Outlier Detection', fontsize=16, fontweight='bold', y=1.02)

for idx, col in enumerate(numeric_cols):
    row, c = divmod(idx, 3)
    if row < 3 and c < 3:
        ax = axes[row][c]
        sns.boxplot(data=df_clean, y=col, ax=ax, color='lightcoral', width=0.5)
        ax.set_title(col, fontsize=12, fontweight='bold')

for idx in range(len(numeric_cols), 9):
    row, c = divmod(idx, 3)
    axes[row][c].set_visible(False)

plt.tight_layout()
box_path = os.path.join(PLOTS_DIR, 'boxplots.png')
plt.savefig(box_path, dpi=150, bbox_inches='tight')
plt.close()

# Quantify outliers using IQR method
print("\n  Outlier count (IQR method):")
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_clean[col] < Q1 - 1.5 * IQR) | (df_clean[col] > Q3 + 1.5 * IQR)).sum()
    pct = outliers / len(df_clean) * 100
    if outliers > 0:
        print(f"    {col:25s}: {outliers:4d} outliers ({pct:.1f}%)")

# ─── Step 7: Class Distribution ──────────────────────────────────────────────
print("Class Distribution (Potability)")


if 'Potability' in df_clean.columns:
    class_counts = df_clean['Potability'].value_counts()
    print(f"  Class 0 (Not Potable): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
    print(f"  Class 1 (Potable):     {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 1.5:
        print(f"\n  Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        print("  → Will use stratified split and consider class_weight='balanced'")
    else:
        print(f"\n  Classes are reasonably balanced (ratio: {imbalance_ratio:.2f})")
    
    # Plot class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF6B6B', '#4ECDC4']
    class_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title('Water Potability — Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Potability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticklabels(['Not Potable (0)', 'Potable (1)'], rotation=0)
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    class_path = os.path.join(PLOTS_DIR, 'class_distribution.png')
    plt.savefig(class_path, dpi=150, bbox_inches='tight')
    plt.close()

# ─── Step 8: Create Multi-Class Quality Labels ───────────────────────────────
print("Step 8: Creating Multi-Class Quality Labels")

def calculate_quality_score(row):
    """
    Calculate water quality score based on WHO/BIS standards.
    Each parameter contributes a sub-score (0-100).
    Final score = weighted average of sub-scores.
    
    Quality Classes:
      - Excellent: score >= 80
      - Good:      score >= 60
      - Poor:      score >= 40
      - Dangerous:  score < 40
    """
    
    scores = []
    
    # pH scoring (ideal: 6.5-8.5)
    if 'ph' in row.index:
        ph = row['ph']
    elif 'pH' in row.index:
        ph = row['pH']
    else:
        ph = 7.0  # default
    
    if 6.5 <= ph <= 8.5:
        scores.append(100)
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0:
        scores.append(70)
    elif 5.5 <= ph < 6.0 or 9.0 < ph <= 9.5:
        scores.append(40)
    else:
        scores.append(10)
    
    # TDS / Solids scoring (ideal: < 500 mg/L)
    tds_col = 'Solids' if 'Solids' in row.index else 'tds'
    if tds_col in row.index:
        tds = row[tds_col]
        if tds < 500:
            scores.append(100)
        elif tds < 1000:
            scores.append(70)
        elif tds < 2000:
            scores.append(40)
        else:
            scores.append(10)
    
    # Turbidity scoring (ideal: < 5 NTU)
    turb_col = 'Turbidity' if 'Turbidity' in row.index else 'turbidity'
    if turb_col in row.index:
        turb = row[turb_col]
        if turb < 1:
            scores.append(100)
        elif turb < 5:
            scores.append(80)
        elif turb < 10:
            scores.append(40)
        else:
            scores.append(10)
    
    # Conductivity scoring (ideal: < 500 µS/cm)
    cond_col = 'Conductivity' if 'Conductivity' in row.index else 'conductivity'
    if cond_col in row.index:
        cond = row[cond_col]
        if cond < 500:
            scores.append(100)
        elif cond < 1000:
            scores.append(70)
        elif cond < 2000:
            scores.append(40)
        else:
            scores.append(10)
    
    # Chloramines scoring (ideal: < 4 mg/L)
    if 'Chloramines' in row.index:
        chlor = row['Chloramines']
        if chlor < 4:
            scores.append(100)
        elif chlor < 7:
            scores.append(70)
        elif chlor < 10:
            scores.append(40)
        else:
            scores.append(10)
    
    # Sulfate scoring (ideal: < 250 mg/L)
    if 'Sulfate' in row.index:
        sulf = row['Sulfate']
        if sulf < 250:
            scores.append(100)
        elif sulf < 400:
            scores.append(70)
        elif sulf < 500:
            scores.append(40)
        else:
            scores.append(10)
    
    return np.mean(scores) if scores else 50


def score_to_class(score):
    """Convert numeric quality score to class label."""
    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Poor'
    else:
        return 'Dangerous'


# Apply scoring
df_clean['quality_score'] = df_clean.apply(calculate_quality_score, axis=1)
df_clean['quality_class'] = df_clean['quality_score'].apply(score_to_class)

# Print distribution
print("\n  Quality Class Distribution:")
quality_counts = df_clean['quality_class'].value_counts()
for cls in ['Excellent', 'Good', 'Poor', 'Dangerous']:
    count = quality_counts.get(cls, 0)
    pct = count / len(df_clean) * 100
    print(f"    {cls:12s}: {count:5d} ({pct:.1f}%)")

# Map to numeric for ML
class_mapping = {'Excellent': 0, 'Good': 1, 'Poor': 2, 'Dangerous': 3}
df_clean['quality_label'] = df_clean['quality_class'].map(class_mapping)

# Plot multi-class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors_quality = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
quality_order = ['Excellent', 'Good', 'Poor', 'Dangerous']
ordered_counts = [quality_counts.get(c, 0) for c in quality_order]

axes[0].bar(quality_order, ordered_counts, color=colors_quality, edgecolor='black', alpha=0.85)
axes[0].set_title('Multi-Class Quality Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(ordered_counts):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Score histogram
axes[1].hist(df_clean['quality_score'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(80, color='green', linestyle='--', label='Excellent threshold')
axes[1].axvline(60, color='blue', linestyle='--', label='Good threshold')
axes[1].axvline(40, color='orange', linestyle='--', label='Poor threshold')
axes[1].set_title('Quality Score Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Quality Score')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
quality_path = os.path.join(PLOTS_DIR, 'quality_classes.png')
plt.savefig(quality_path, dpi=150, bbox_inches='tight')
plt.close()

# ─── Step 9: Save Cleaned Dataset ────────────────────────────────────────────
print("Step 9: Saving Cleaned Dataset")

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

# Also save a time-series version with synthetic timestamps
# (for the forecasting model later)
print("\n  Creating time-series version with synthetic timestamps...")
df_ts = df_clean.copy()
df_ts['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df_ts), freq='h')
ts_path = os.path.join(BASE_DIR, 'data', 'processed', 'water_quality_timeseries.csv')
df_ts.to_csv(ts_path, index=False)