from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models and data
def load_models():
    """Load all necessary models and files"""
    try:
        feature_names = pd.read_csv("feature_names.csv")["Feature Names"].tolist()
        scaler = joblib.load("feature_scaler.pkl")
        cardio_probabilities = joblib.load("cardio_probabilities.pkl")
        
        models_pvd = {
            "DecisionTree": joblib.load("DecisionTree_model_pvd.pkl"),
            "RandomForest": joblib.load("RandomForest_model_pvd.pkl"),
            "SGD": joblib.load("SGD_model_pvd.pkl"),
            "XGBoost": joblib.load("XGBoost_model_pvd.pkl")
        }
        
        models_cardio = {
            "DecisionTree": joblib.load("DecisionTree_model_cardio.pkl"),
            "RandomForest": joblib.load("RandomForest_model_cardio.pkl"),
            "SGD": joblib.load("SGD_model_cardio.pkl"),
            "XGBoost": joblib.load("XGBoost_model_cardio.pkl")
        }
        
        return feature_names, scaler, models_pvd, models_cardio, cardio_probabilities
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Validate user input
def validate_input(user_input):
    """Validate user input against realistic ranges"""
    ranges = {
        'age': (18, 100),
        'height': (140, 220),
        'weight': (40, 200),
        'ap_hi': (80, 220),
        'ap_lo': (40, 140),
        'cholesterol': (1, 3),
        'gluc': (1, 3),
        'smoke': (0, 1),
        'alco': (0, 1),
        'active': (0, 1),
        'gender': (1, 2)
    }
    
    errors = {}
    for key, (min_val, max_val) in ranges.items():
        value = user_input.get(key)
        if value is None or not min_val <= value <= max_val:
            errors[key] = f"must be between {min_val} and {max_val}"
    
    if user_input.get('ap_hi', 0) <= user_input.get('ap_lo', 0):
        errors['ap_hi'] = "Systolic pressure must be greater than diastolic pressure"
    
    return errors

# Preprocess input data
def preprocess_input(user_input, feature_names, scaler):
    """Preprocess user input for prediction"""
    df = pd.DataFrame([user_input])
    
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_names]

# Analyze PVD impact on cardiovascular risk
def analyze_pvd_impact(cardio_probabilities):
    """Calculate how much PVD increases cardiovascular risk"""
    without_pvd_prob = cardio_probabilities['without_pvd']
    with_pvd_prob = cardio_probabilities['with_pvd']
    
    # Calculate how much PVD increases cardio risk
    pvd_impact = ((with_pvd_prob - without_pvd_prob) / without_pvd_prob) * 100
    
    # Determine impact severity
    if pvd_impact > 100:
        severity = "Very High"
    elif pvd_impact > 50:
        severity = "High"
    elif pvd_impact > 25:
        severity = "Moderate"
    else:
        severity = "Low"
    
    return {
        "without_pvd": without_pvd_prob * 100,
        "with_pvd": with_pvd_prob * 100,
        "impact_percentage": pvd_impact,
        "severity": severity
    }

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load models
        feature_names, scaler, models_pvd, models_cardio, cardio_probabilities = load_models()
        if not all([feature_names, scaler, models_pvd, models_cardio, cardio_probabilities]):
            return jsonify({'error': 'Failed to load models'}), 500
        
        # Get and parse input data
        data = request.get_json()
        user_input = {
            'age': int(data['age']),
            'gender': int(data['gender']),
            'height': float(data['height']),
            'weight': float(data['weight']),
            'ap_hi': int(data['ap_hi']),
            'ap_lo': int(data['ap_lo']),
            'cholesterol': int(data['cholesterol']),
            'gluc': int(data['gluc']),
            'smoke': int(data['smoke']),
            'alco': int(data['alco']),
            'active': int(data['active'])
        }
        
        # Validate input
        errors = validate_input(user_input)
        if errors:
            return jsonify({'errors': errors}), 400
        
        # Process input
        processed_input = preprocess_input(user_input, feature_names, scaler)
        
        # Get selected model
        model_name = data['model']
        if model_name not in models_pvd:
            return jsonify({'error': 'Invalid model selection'}), 400
            
        # PVD Prediction
        model_pvd = models_pvd[model_name]
        pvd_prediction = int(model_pvd.predict(processed_input)[0])
        pvd_probability = float(model_pvd.predict_proba(processed_input)[0][1] * 100)
        
        # Prepare response
        response = {
            'pvd_detected': bool(pvd_prediction),
            'pvd_probability': pvd_probability,
            'pvd_risk_level': "Low" if pvd_probability < 30 else "Moderate" if pvd_probability < 70 else "High"
        }
        
        # If PVD detected, check cardiovascular risk
        if pvd_prediction == 1:
            model_cardio = models_cardio[model_name]
            cardio_prediction = int(model_cardio.predict(processed_input)[0])
            cardio_probability = float(model_cardio.predict_proba(processed_input)[0][1] * 100)
            
            # Analyze impact
            impact_data = analyze_pvd_impact(cardio_probabilities)
            
            response['cardio_detected'] = bool(cardio_prediction)
            response['cardio_probability'] = cardio_probability
            response['impact'] = impact_data
            
            # Generate recommendations
            recommendations = []
            
            # Based on PVD impact severity
            if impact_data['severity'] in ["High", "Very High"]:
                recommendations.extend([
                    "Urgent cardiovascular specialist consultation recommended",
                    "Comprehensive cardiovascular assessment needed",
                    "Frequent monitoring of cardiovascular health"
                ])
            else:
                recommendations.extend([
                    "Regular cardiovascular health monitoring",
                    "Scheduled follow-ups with healthcare provider"
                ])
            
            # Based on risk factors
            if user_input['smoke'] == 1:
                recommendations.append("Smoking cessation is crucial - PVD significantly increases this risk")
            if user_input['alco'] == 1:
                recommendations.append("Reduce alcohol consumption")
            if user_input['active'] == 0:
                recommendations.append("Start a supervised exercise program")
            if user_input['cholesterol'] > 1:
                recommendations.append("Strict cholesterol management needed")
            
            recommendations.extend([
                "Maintain healthy blood pressure",
                "Follow heart-healthy diet"
            ])
            
            response['recommendations'] = recommendations
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use environment variable for port if available (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)