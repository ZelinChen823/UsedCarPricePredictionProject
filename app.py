from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

# Import the trained model
model = joblib.load('model.joblib')
with open('unique_values.json', 'r') as f:
    unique_values = json.load(f)

numerical_columns = ['year', 'engine_hp', 'engine_cylinders', 'number_of_doors',
                     'highway_mpg', 'city_mpg', 'popularity']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    input_df = pd.DataFrame([data])
    # Convert numerical columns to numeric types
    for col in numerical_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_formatted = f"{prediction:,.0f}"
    return render_template('result.html', prediction=prediction_formatted, input_data=data)

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
    prediction = model.predict(input_df)[0]
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