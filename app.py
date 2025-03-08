from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load the trained model and unique values
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

    # Return prediction and original input data to result.html
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

if __name__ == '__main__':
    app.run(debug=True)