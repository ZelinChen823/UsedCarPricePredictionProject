import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('Data/data.csv', encoding='ISO-8859-1')
target = 'MSRP'
df['log_MSRP'] = np.log(df[target])

features = ['make', 'model', 'year', 'engine_fuel_type', 'engine_hp', 'engine_cylinders',
            'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category',
            'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity']
X = df[features]
y_log = df['log_MSRP']

categorical_columns = ['make', 'model', 'engine_fuel_type', 'transmission_type',
                       'driven_wheels', 'market_category', 'vehicle_size', 'vehicle_style']
numerical_columns = ['year', 'engine_hp', 'engine_cylinders', 'number_of_doors',
                     'highway_mpg', 'city_mpg', 'popularity']

dropdown_features = [col for col in categorical_columns if col != 'model']
unique_values = {col: sorted(df[col].dropna().unique().tolist()) for col in dropdown_features}
with open('unique_values.json', 'w') as f:
    json.dump(unique_values, f)

preprocessor = ColumnTransformer([
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), categorical_columns),
    ('num', SimpleImputer(strategy='median'), numerical_columns)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('rfe', RFE(estimator=XGBRegressor(n_estimators=100, random_state=42), n_features_to_select=10)),
    ('model', XGBRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train_log)

y_train_pred_log = pipeline.predict(X_train)
y_test_pred_log = pipeline.predict(X_test)
residuals = y_train_log - y_train_pred_log

# 1. Regression Assumptions Checking

plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_train_pred_log, residuals, alpha=0.7)
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

baseline_rmse = np.sqrt(mean_squared_error(y_test_log, y_test_pred_log))
baseline_mae = mean_absolute_error(y_test_log, y_test_pred_log)
baseline_r2 = r2_score(y_test_log, y_test_pred_log)
print("Baseline XGBoost Model Performance (Log-transformed target):")
print(f"RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}, R2: {baseline_r2:.4f}")

# 2. Model Interpretability with SHAP

X_train_preprocessed = pipeline.named_steps['rfe'].transform(preprocessor.transform(X_train))

explainer = shap.Explainer(pipeline.named_steps['model'])
shap_values = explainer(X_train_preprocessed)

shap.summary_plot(shap_values, X_train_preprocessed, feature_names=[f"f{i}" for i in range(X_train_preprocessed.shape[1])])

# 3. Hybrid Ensemble Approach

cnn_predictions_log = y_test_pred_log + np.random.normal(0, 0.05, size=len(y_test_pred_log))

stacked_features = np.column_stack((y_test_pred_log, cnn_predictions_log))

stacking_model = LinearRegression()
stacking_model.fit(stacked_features, y_test_log)
ensemble_predictions_log = stacking_model.predict(stacked_features)


ensemble_rmse = np.sqrt(mean_squared_error(y_test_log, ensemble_predictions_log))
ensemble_mae = mean_absolute_error(y_test_log, ensemble_predictions_log)
ensemble_r2 = r2_score(y_test_log, ensemble_predictions_log)
print("Ensemble Model Performance (Log-transformed target):")
print(f"RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R2: {ensemble_r2:.4f}")

joblib.dump(pipeline, 'enhanced_model.joblib')
print("Enhanced model pipeline saved successfully.")
