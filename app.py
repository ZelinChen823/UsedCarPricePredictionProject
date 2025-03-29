from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from scipy.spatial.distance import cdist
from train_model import training_info
import json
import os
from datetime import datetime

app = Flask(__name__)

# Import the trained model
model = joblib.load('enhanced_model.joblib')
with open('unique_values.json', 'r') as f:
    unique_values = json.load(f)

numerical_columns = ['year', 'engine_hp', 'engine_cylinders', 'number_of_doors',
                     'highway_mpg', 'city_mpg', 'popularity']

HISTORY_FILE = 'search_history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    return history

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', unique_values=unique_values)

@app.route('/history', methods=['GET'])
def history():
    # Load the saved history from file and pass it to the template
    history = load_history()
    return render_template('history.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    input_df = pd.DataFrame([data])
    # Convert numerical columns to numeric types
    for col in numerical_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Make prediction
    prediction = np.exp(model.predict(input_df)[0])

    input_transformed = model.named_steps['preprocess'].transform(input_df)

    distances = cdist(input_transformed, training_info['X_train_transformed'], metric='euclidean')[0]

    k = 5
    nearest_indices = distances.argsort()[:k]
    mean_error = np.mean(training_info['residuals'][nearest_indices])

    confidence_score = max(0, 1 - (mean_error / prediction)) * 100

    prediction_formatted = f"{prediction:,.0f}"
    confidence_formatted = f"{confidence_score:.1f}%"

    history = load_history()
    history_item = {
        'input_data': data,
        'prediction': prediction_formatted,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    history.append(history_item)
    save_history(history)

    return render_template('result.html', prediction=prediction_formatted, input_data=data,
                           confidence=confidence_formatted)

@app.route('/predict_update', methods=['POST'])
def predict_update():
    # Get updated data from the client
    data = request.get_json()
    # DataFrame with updated data
    input_df = pd.DataFrame([data])

    # Convert numerical columns to numeric types
    for col in numerical_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Make new prediction
    prediction = np.exp(model.predict(input_df)[0])
    prediction_formatted = f"{prediction:,.0f}"

    # Return the new prediction as JSON
    return jsonify({'prediction': prediction_formatted})

@app.route('/trend', methods=['POST'])
def trend():
    data = request.get_json()
    feature_to_vary = data['feature_to_vary']
    input_data = data['input_data']

    feature_ranges = {
        'year': (1990, 2025),
        'highway_mpg': (10, 50),
        'city_mpg': (5, 40),
        'engine_hp': (50, 1000),
        'engine_cylinders': (2, 16),
        'number_of_doors': (2, 4),
        'popularity': (0, 10000)
    }

    if feature_to_vary in feature_ranges:
        min_val, max_val = feature_ranges[feature_to_vary]
        if feature_to_vary == 'year':
            values = list(range(min_val, max_val + 1))
        else:
            values = np.linspace(min_val, max_val, 10).tolist()
    else:
        return jsonify({'error': 'Feature not supported for trend analysis'}), 400

    trend_data = []
    for val in values:
        updated_data = input_data.copy()
        updated_data[feature_to_vary] = val
        input_df = pd.DataFrame([updated_data])

        for col in numerical_columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        prediction = model.predict(input_df)[0]
        trend_data.append({'value': val, 'prediction': float(prediction)})

    return jsonify(trend_data)

if __name__ == '__main__':
    app.run(debug=True)