from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging
import traceback
import sys

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load models and data
def load_models():
    """Load all necessary models and files"""
    try:
        logger.info("Starting to load models and data files")
        
        # Log the current working directory
        current_working_directory = os.getcwd()
        logger.info(f"Current working directory: {current_working_directory}")
        
        # Check if files exist before loading
        required_files = [
            "feature_names.csv", "feature_scaler.pkl", "cardio_probabilities.pkl",
            "DecisionTree_model_pvd.pkl", "RandomForest_model_pvd.pkl", 
            "SGD_model_pvd.pkl", "XGBoost_model_pvd.pkl",
            "DecisionTree_model_cardio.pkl", "RandomForest_model_cardio.pkl",
            "SGD_model_cardio.pkl", "XGBoost_model_cardio.pkl"
        ]
        
        for file in required_files:
            absolute_path = os.path.abspath(file)
            if not os.path.exists(absolute_path):
                logger.error(f"Required file not found: {absolute_path}")
                return None, None, None, None, None
            else:
                logger.info(f"Found required file: {absolute_path}")
        
        feature_names = pd.read_csv("feature_names.csv")["Feature Names"].tolist()
        logger.info(f"Loaded feature names: {feature_names}")
        
        scaler = joblib.load("feature_scaler.pkl")
        logger.info("Loaded feature scaler")
        
        cardio_probabilities = joblib.load("cardio_probabilities.pkl")
        logger.info("Loaded cardio probabilities")
        
        models_pvd = {
            "DecisionTree": joblib.load("DecisionTree_model_pvd.pkl"),
            "RandomForest": joblib.load("RandomForest_model_pvd.pkl"),
            "SGD": joblib.load("SGD_model_pvd.pkl"),
            "XGBoost": joblib.load("XGBoost_model_pvd.pkl")
        }
        logger.info("Loaded PVD models successfully")
        
        models_cardio = {
            "DecisionTree": joblib.load("DecisionTree_model_cardio.pkl"),
            "RandomForest": joblib.load("RandomForest_model_cardio.pkl"),
            "SGD": joblib.load("SGD_model_cardio.pkl"),
            "XGBoost": joblib.load("XGBoost_model_cardio.pkl")
        }
        logger.info("Loaded Cardio models successfully")
        
        return feature_names, scaler, models_pvd, models_cardio, cardio_probabilities
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
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

# Test endpoint to verify model loading
@app.route('/test', methods=['GET'])
def test():
    try:
        logger.info("Testing model loading")
        feature_names, scaler, models_pvd, models_cardio, cardio_probabilities = load_models()
        
        if not all([feature_names, scaler, models_pvd, models_cardio, cardio_probabilities]):
            logger.error("Failed to load models during test")
            return jsonify({
                'status': 'error',
                'message': 'Failed to load models'
            }), 500
        
        return jsonify({
            'status': 'success',
            'message': 'All models loaded successfully',
            'details': {
                'feature_count': len(feature_names),
                'pvd_models': list(models_pvd.keys()),
                'cardio_models': list(models_cardio.keys())
            }
        })
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Prediction request received")
        
        # Get and parse input data
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        # Load models
        logger.info("Loading models for prediction")
        feature_names, scaler, models_pvd, models_cardio, cardio_probabilities = load_models()
        if not all([feature_names, scaler, models_pvd, models_cardio, cardio_probabilities]):
            logger.error("Failed to load models for prediction")
            return jsonify({'error': 'Failed to load models'}), 500
        
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
        logger.info("Validating user input")
        errors = validate_input(user_input)
        if errors:
            logger.warning(f"Input validation failed: {errors}")
            return jsonify({'errors': errors}), 400
        
        # Process input
        logger.info("Preprocessing input data")
        processed_input = preprocess_input(user_input, feature_names, scaler)
        
        # Get selected model
        model_name = data['model']
        logger.info(f"Selected model: {model_name}")
        if model_name not in models_pvd:
            logger.warning(f"Invalid model selection: {model_name}")
            return jsonify({'error': 'Invalid model selection'}), 400
            
        # PVD Prediction
        logger.info("Making PVD prediction")
        model_pvd = models_pvd[model_name]
        pvd_prediction = int(model_pvd.predict(processed_input)[0])
        pvd_probability = float(model_pvd.predict_proba(processed_input)[0][1] * 100)
        logger.info(f"PVD prediction: {pvd_prediction}, probability: {pvd_probability}")
        
        # Prepare response
        response = {
            'pvd_detected': bool(pvd_prediction),
            'pvd_probability': pvd_probability,
            'pvd_risk_level': "Low" if pvd_probability < 30 else "Moderate" if pvd_probability < 70 else "High"
        }
        
        # If PVD detected, check cardiovascular risk
        if pvd_prediction == 1:
            logger.info("PVD detected, checking cardiovascular risk")
            model_cardio = models_cardio[model_name]
            cardio_prediction = int(model_cardio.predict(processed_input)[0])
            cardio_probability = float(model_cardio.predict_proba(processed_input)[0][1] * 100)
            logger.info(f"Cardio prediction: {cardio_prediction}, probability: {cardio_probability}")
            
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
        
        logger.info("Successfully generated prediction response")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Use environment variable for port if available (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)