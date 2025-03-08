import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('Data/data.csv', encoding='ISO-8859-1')

features = ['make', 'model', 'year', 'engine_fuel_type', 'engine_hp', 'engine_cylinders',
            'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category',
            'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity']
target = 'MSRP'
X = df[features]
y = df[target]

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
    ('model', XGBRegressor(n_estimators=100, random_state=42))
])

# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model using RMSE (for test)
'''
from sklearn.metrics import mean_squared_error
import numpy as np
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
'''
# Model saved by using joblib
joblib.dump(pipeline, 'model.joblib')
print("Model and unique values saved successfully.")