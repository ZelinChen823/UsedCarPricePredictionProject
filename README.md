# Vehicle Value Lookup (VLT) – Used Car Price Prediction

## Overview
VLT is a Flask‑based web application that predicts used car prices using an XGBoost regression pipeline with log‑transformation and RFE and provides:

- **Instant price estimates** based on user-provided vehicle features  
- **Confidence score** via nearest‑neighbor residual analysis  
- **What‑if analysis** for key numerical features 
- **Trend analysis** showing how predicted price varies across feature ranges  
- **Search history** of past predictions

All work is implemented in **Python 3.12** using **PyCharm Professional**.

## Features
- Preprocessing pipeline with imputation & ordinal encoding  
- Log‑transform target variable and RFE feature selection  
- Model interpretability via SHAP  
- Hybrid ensemble stacking (XGBoost + CNN placeholder)  
- REST endpoints for prediction, live updates, trend generation

## File Structure
```
VLT/
├── Data/  
│   └── data.csv  
├── templates/  
│   ├── index.html  
│   ├── result.html  
│   └── history.html  
├── app.py  
├── train_model.py  
├── hybrid_model.py  
├── enhanced_model.joblib  
├── unique_values.json        # created at runtime 
├── search_history.json       # created at runtime  
├── README.md  
└── requirements.txt
```

## Usage

1. **Clone the repo**  
```
git clone https://github.com/ZelinChen823/UsedCarPricePredictionProject
cd Car-Prices-Prediction/VLT
```
