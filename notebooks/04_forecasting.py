"""
===============================================================================
 04_forecasting.py — Water Quality Time-Series Forecasting
===============================================================================
 Trains Prophet models to forecast future water quality parameter values.
 
 What this script does:
   1. Loads time-series dataset (with timestamps)
   2. Trains a Prophet model for each parameter (pH, TDS, etc.)
   3. Generates forecasts for the next 24/48 hours
   4. Evaluates with MAE/RMSE on held-out data
   5. Visualizes forecasts with confidence intervals
   6. Saves all models
 
 Note: Since we're using synthetic timestamps on the Kaggle dataset,
 the forecasts are for demonstration/pipeline testing. Once your boat
 generates real time-series data, retrain with actual timestamps.
 
 Prerequisites:
   Run 01_eda.py first!
 
 Usage:
   python notebooks/04_forecasting.py
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join('data', 'processed', 'water_quality_timeseries.csv')
MODELS_DIR = os.path.join('models', 'prophet_models')
PLOTS_DIR = os.path.join('notebooks', 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Parameters to forecast
FORECAST_PARAMS = ['ph', 'Solids', 'Turbidity', 'Conductivity', 'Chloramines', 'Sulfate']

# Forecast horizon
FORECAST_HOURS = 48

print("=" * 70)
print("  WATER QUALITY — TIME-SERIES FORECASTING (PROPHET)")
print("=" * 70)

# ─── Step 0: Check if Prophet is installed ────────────────────────────────────
try:
    from prophet import Prophet
    print("\n  ✅ Prophet is installed")
except ImportError:
    print("\n  ⚠️  Prophet not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'prophet'])
    from prophet import Prophet
    print("  ✅ Prophet installed successfully")

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ─── Step 1: Load Data ───────────────────────────────────────────────────────
print("\n📂 Step 1: Loading time-series dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
print(f"  Shape: {df.shape}")
print(f"  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

available_params = [p for p in FORECAST_PARAMS if p in df.columns]
print(f"  Parameters to forecast: {available_params}")

# ─── Step 2: Train & Evaluate Prophet Models ─────────────────────────────────
print("\n" + "─" * 70)
print("📈 Step 2: Training Prophet Models")
print("─" * 70)

results = {}
models_dict = {}

for param in available_params:
    print(f"\n  ┌─ Training model for: {param}")
    print(f"  │")
    
    # Prepare Prophet format: ds (datetime), y (value)
    df_prophet = df[['timestamp', param]].rename(
        columns={'timestamp': 'ds', param: 'y'}
    ).dropna()
    
    # Train/test split: use last 10% as test
    split_idx = int(len(df_prophet) * 0.9)
    df_train = df_prophet[:split_idx]
    df_test = df_prophet[split_idx:]
    
    print(f"  │  Training samples: {len(df_train)}")
    print(f"  │  Testing samples:  {len(df_test)}")
    
    # Train Prophet model
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # Not enough data for yearly
    )
    
    # Suppress Prophet's verbose output
    model.fit(df_train)
    
    # Evaluate on test set
    forecast_test = model.predict(df_test[['ds']])
    y_true = df_test['y'].values
    y_pred = forecast_test['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"  │  MAE:  {mae:.4f}")
    print(f"  │  RMSE: {rmse:.4f}")
    print(f"  │  MAPE: {mape:.2f}%")
    
    # Generate future forecast
    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq='h')
    forecast = model.predict(future)
    
    # Store results
    results[param] = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'train_size': len(df_train),
        'test_size': len(df_test)
    }
    models_dict[param] = model
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f'{param.lower()}_model.json')
    with open(model_path, 'w') as f:
        f.write(model.to_json())
    print(f"  │  ✅ Model saved → {model_path}")
    print(f"  └─ Done!")

# ─── Step 3: Results Summary ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("📊 Step 3: Results Summary")
print("─" * 70)

results_df = pd.DataFrame(results).T
results_df.index.name = 'Parameter'
print("\n" + results_df.round(4).to_string())

# ─── Step 4: Generate Visualizations ─────────────────────────────────────────
print("\n" + "─" * 70)
print("📊 Step 4: Generating Forecast Visualizations")
print("─" * 70)

# Plot 1: Individual forecast plots
n_params = len(available_params)
n_cols = 2
n_rows = (n_params + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
fig.suptitle(f'Water Quality Forecasts ({FORECAST_HOURS}h Horizon)', 
             fontsize=16, fontweight='bold', y=1.02)

if n_rows == 1:
    axes = axes.reshape(1, -1)

for idx, param in enumerate(available_params):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]
    
    model = models_dict[param]
    
    # Prepare data
    df_prophet = df[['timestamp', param]].rename(
        columns={'timestamp': 'ds', param: 'y'}
    ).dropna()
    
    # Only plot last 200 hours + forecast
    recent = df_prophet.tail(200)
    
    # Get forecast for this period
    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq='h')
    forecast = model.predict(future)
    
    # Plot actual data (last 200 points)
    ax.plot(recent['ds'], recent['y'], 'b.', alpha=0.4, markersize=2, label='Actual')
    
    # Plot forecast
    forecast_future = forecast[forecast['ds'] > df_prophet['ds'].max()]
    forecast_recent = forecast[forecast['ds'].isin(recent['ds'])]
    
    ax.plot(forecast_recent['ds'], forecast_recent['yhat'], 
            'r-', alpha=0.7, linewidth=1, label='Fitted')
    ax.plot(forecast_future['ds'], forecast_future['yhat'],
            'g-', linewidth=2, label=f'Forecast ({FORECAST_HOURS}h)')
    
    # Confidence interval for forecast
    ax.fill_between(forecast_future['ds'],
                     forecast_future['yhat_lower'],
                     forecast_future['yhat_upper'],
                     alpha=0.2, color='green', label='95% CI')
    
    # Vertical line at forecast start
    ax.axvline(df_prophet['ds'].max(), color='black', linestyle='--', alpha=0.5)
    
    ax.set_title(f'{param} — MAE: {results[param]["mae"]:.3f}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel(param)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)

# Hide unused subplots
for idx in range(len(available_params), n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes[row][col].set_visible(False)

plt.tight_layout()
forecast_path = os.path.join(PLOTS_DIR, 'forecasts.png')
plt.savefig(forecast_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved forecast plots → {forecast_path}")

# Plot 2: Model performance comparison
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(available_params))
width = 0.25

mae_vals = [results[p]['mae'] for p in available_params]
rmse_vals = [results[p]['rmse'] for p in available_params]
mape_vals = [results[p]['mape'] / 100 for p in available_params]  # Normalize for display

bars1 = ax.bar(x - width, mae_vals, width, label='MAE', color='#3498DB', alpha=0.85)
bars2 = ax.bar(x, rmse_vals, width, label='RMSE', color='#E74C3C', alpha=0.85)
bars3 = ax.bar(x + width, mape_vals, width, label='MAPE (normalized)', color='#2ECC71', alpha=0.85)

ax.set_xlabel('Parameter', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Forecast Model Performance by Parameter', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(available_params, fontsize=10)
ax.legend(fontsize=10)

plt.tight_layout()
perf_path = os.path.join(PLOTS_DIR, 'forecast_performance.png')
plt.savefig(perf_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved performance comparison → {perf_path}")

# Plot 3: Prophet components for the first parameter
print("  Generating component plots...")
for param in available_params[:2]:  # Just first 2 to keep it manageable
    model = models_dict[param]
    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq='h')
    forecast = model.predict(future)
    
    fig = model.plot_components(forecast)
    comp_path = os.path.join(PLOTS_DIR, f'components_{param.lower()}.png')
    fig.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved components for {param} → {comp_path}")

# ─── Step 5: Save Forecast Metadata ──────────────────────────────────────────
print("\n" + "─" * 70)
print("💾 Step 5: Saving Forecast Metadata")
print("─" * 70)

forecast_meta = {
    'parameters': available_params,
    'forecast_horizon_hours': FORECAST_HOURS,
    'model_type': 'Prophet',
    'results': {k: {kk: float(vv) for kk, vv in v.items()} 
                for k, v in results.items()},
    'model_files': {p: f'{p.lower()}_model.json' for p in available_params}
}

meta_path = os.path.join(MODELS_DIR, 'forecast_meta.json')
with open(meta_path, 'w') as f:
    json.dump(forecast_meta, f, indent=2)
print(f"  ✅ Metadata saved → {meta_path}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TIME-SERIES FORECASTING COMPLETE ✅")
print("=" * 70)
print(f"""
  📈 Models trained for: {', '.join(available_params)}
  ⏰ Forecast horizon: {FORECAST_HOURS} hours
  
  📊 Average MAE across parameters:  {np.mean(mae_vals):.4f}
  📊 Average RMSE across parameters: {np.mean(rmse_vals):.4f}
  
  📁 Models saved to: {MODELS_DIR}/
  📁 Plots saved to:  {PLOTS_DIR}/
  
  ⚠️  NOTE: These models were trained on synthetic timestamps.
  Retrain with real boat data for production use!
  
  Next step: Run the FastAPI prediction API (src/api.py)
""")
