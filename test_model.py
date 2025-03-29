import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Data and Define Features
df = pd.read_csv('Data/data.csv', encoding='ISO-8859-1')

features = ['make', 'model', 'year', 'engine_fuel_type', 'engine_hp', 'engine_cylinders',
            'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category',
            'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity']
target = 'MSRP'

# For the baseline model (original target)
X = df[features]
y = df[target]

# For the enhanced model, we use a log transformation
df['log_MSRP'] = np.log(df[target])
y_log = df['log_MSRP']

# Split Data into Test Sets
X_train_base, X_test_base, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enhanced test split
X_train_enh, X_test_enh, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Load Models
baseline_pipeline = joblib.load('model.joblib')
enhanced_pipeline = joblib.load('enhanced_model.joblib')

# Evaluate Baseline Model (Original Target)
X_test_transformed = baseline_pipeline.named_steps['preprocess'].transform(X_test_base)
y_pred_baseline = baseline_pipeline.named_steps['model'].predict(X_test_transformed)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)

print("Baseline Model Performance:")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"MAE: {baseline_mae:.2f}")
print(f"R2 Score: {baseline_r2:.2f}\n")

# Evaluate Enhanced Model (Log-transformed Target)
y_pred_enhanced_log = enhanced_pipeline.predict(X_test_enh)

enhanced_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_enhanced_log))
enhanced_mae = mean_absolute_error(y_test_log, y_pred_enhanced_log)
enhanced_r2 = r2_score(y_test_log, y_pred_enhanced_log)

print("Enhanced Model Performance (Log-transformed target):")
print(f"RMSE (log scale): {enhanced_rmse:.4f}")
print(f"MAE (log scale): {enhanced_mae:.4f}")
print(f"R2 (log scale): {enhanced_r2:.4f}\n")

# Only for test: Convert Enhanced Predictions Back to Original Scale
y_pred_enhanced = np.exp(y_pred_enhanced_log)
enhanced_rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_enhanced))
enhanced_mae_orig = mean_absolute_error(y_test, y_pred_enhanced)
enhanced_r2_orig = r2_score(y_test, y_pred_enhanced)

print("Enhanced Model Performance (Converted to Original Scale):")
print(f"RMSE: {enhanced_rmse_orig:.2f}")
print(f"MAE: {enhanced_mae_orig:.2f}")
print(f"R2 Score: {enhanced_r2_orig:.2f}")
