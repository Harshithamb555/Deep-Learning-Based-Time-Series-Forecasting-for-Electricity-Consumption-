import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from Transformer import Transformer
from utils import series_to_supervised

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
dataset_path = 'data/household_power_consumption.csv'
dataset = pd.read_csv(
    dataset_path,
    header=0,
    parse_dates=['datetime'],
    index_col=['datetime']
)

# Resample to daily
daily_data = dataset.resample('D').sum()

# Keep only 'Global_active_power'
to_drop = ['Global_reactive_power', 'Voltage', 'Global_intensity',
           'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
daily_data.drop(columns=to_drop, inplace=True)

# Calendar-related features
daily_data['day'] = daily_data.index.day
daily_data['weekday'] = ((daily_data.index.dayofweek) // 5 == 1).astype(float)
daily_data['season'] = [month % 12 // 3 + 1 for month in daily_data.index.month]

print(daily_data.info())

# -------------------------------
# 2️⃣ Prepare Data
# -------------------------------
look_back = 7
n_features = daily_data.shape[1]

# Walk-forward split
X_train, y_train, X_test, y_test, scale_X = series_to_supervised(
    daily_data,
    train_size=0.8,
    n_in=look_back,
    n_out=7,
    target_column='Global_active_power',
    dropnan=True,
    scale_X=False  # no scaler to avoid inverse_transform errors
)

# Reshape to 3D for Transformer
X_train_reshaped = X_train.values.reshape((-1, look_back, n_features))
X_test_reshaped = X_test.values.reshape((-1, look_back, n_features))

y_train_reshaped = y_train.values
y_test_reshaped = y_test.values

# -------------------------------
# 3️⃣ Train Transformer
# -------------------------------
tr = Transformer()
tr.train(X_train_reshaped, y_train_reshaped)

# Predict on test set
y_pred_actual = tr.model.predict(X_test_reshaped)
y_test_actual = y_test_reshaped

# -------------------------------
# 4️⃣ Evaluate
# -------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse_val = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae_val = mean_absolute_error(y_test_actual, y_pred_actual)
r2_val = r2_score(y_test_actual, y_pred_actual) * 100  # %

print(f"Result \nRMSE = {rmse_val:.2f} kWh\nMAE = {mae_val:.2f} kWh\nR2 = {r2_val:.1f} %")

metrics_text = f"RMSE={rmse_val:.2f} kWh\nMAE={mae_val:.2f} kWh\nR²={r2_val:.1f}%"

# -------------------------------
# 5️⃣ Plot Results
# -------------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Full test set
plt.figure(figsize=(15,6))
plt.plot(y_test_actual, label="Actual", color='blue')
plt.plot(y_pred_actual, label="Predicted", color='red')
plt.title("Transformer Forecast vs Actual - Full Test Set")
plt.xlabel("Time Steps")
plt.ylabel("Global Active Power (kWh)")
plt.legend()
plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "full_forecast_plot.png"))
plt.close()

# Zoomed-in (first 100 days)
plt.figure(figsize=(12,5))
plt.plot(y_test_actual[:100], label="Actual", color='blue')
plt.plot(y_pred_actual[:100], label="Predicted", color='red')
plt.title("Transformer Forecast vs Actual - First 100 Days")
plt.xlabel("Days")
plt.ylabel("Global Active Power (kWh)")
plt.legend()
plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "zoomed_forecast_plot.png"))
plt.close()

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5, color='green')
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         color='red', linestyle='--')  # perfect prediction line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "scatter_actual_vs_predicted.png"))
plt.close()

print("✅ Plots saved in the 'results' folder.")
