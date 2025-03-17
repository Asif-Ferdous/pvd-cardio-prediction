import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

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

def validate_input(user_input):
    """Validate user input against realistic ranges"""
    ranges = {
        'age': (18, 100, "years"),
        'height': (140, 220, "cm"),
        'weight': (40, 200, "kg"),
        'ap_hi': (80, 220, "mmHg"),
        'ap_lo': (40, 140, "mmHg"),
        'cholesterol': (1, 3, "(1=Normal, 2=Above Normal, 3=Well Above Normal)"),
        'gluc': (1, 3, "(1=Normal, 2=Above Normal, 3=Well Above Normal)"),
        'smoke': (0, 1, "(0=No, 1=Yes)"),
        'alco': (0, 1, "(0=No, 1=Yes)"),
        'active': (0, 1, "(0=No, 1=Yes)"),
        'gender': (1, 2, "(1=Male, 2=Female)")
    }
    
    for key, (min_val, max_val, units) in ranges.items():
        value = user_input.get(key)
        if value is None or not min_val <= value <= max_val:
            raise ValueError(f"{key} must be between {min_val} and {max_val} {units}")
    
    if user_input['ap_hi'] <= user_input['ap_lo']:
        raise ValueError("Systolic pressure (ap_hi) must be greater than diastolic pressure (ap_lo)")

def get_user_input():
    """Get and validate user input"""
    print("\nPlease enter patient details for PVD prediction:")
    
    try:
        user_data = {
            'age': int(input("Enter Age (18-100 years): ")),
            'gender': int(input("Enter Gender (1=Male, 2=Female): ")),
            'height': float(input("Enter Height (140-220 cm): ")),
            'weight': float(input("Enter Weight (40-200 kg): ")),
            'ap_hi': int(input("Enter Systolic Blood Pressure (80-220 mmHg): ")),
            'ap_lo': int(input("Enter Diastolic Blood Pressure (40-140 mmHg): ")),
            'cholesterol': int(input("Enter Cholesterol Level (1=Normal, 2=Above Normal, 3=Well Above Normal): ")),
            'gluc': int(input("Enter Glucose Level (1=Normal, 2=Above Normal, 3=Well Above Normal): ")),
            'smoke': int(input("Do you smoke? (0=No, 1=Yes): ")),
            'alco': int(input("Do you consume alcohol? (0=No, 1=Yes): ")),
            'active': int(input("Are you physically active? (0=No, 1=Yes): "))
        }
        
        validate_input(user_data)
        return user_data
        
    except ValueError as e:
        print(f"\nError: {str(e)}")
        return None

def preprocess_input(user_input, feature_names, scaler):
    """Preprocess user input for prediction"""
    df = pd.DataFrame([user_input])
    
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_names]

def analyze_pvd_impact(cardio_probability, cardio_probabilities):
    """Calculate how much PVD increases cardiovascular risk"""
    without_pvd_prob = cardio_probabilities['without_pvd']
    with_pvd_prob = cardio_probabilities['with_pvd']
    
    # Calculate how much PVD increases cardio risk
    pvd_impact = ((with_pvd_prob - without_pvd_prob) / without_pvd_prob) * 100
    
    print("\n=== Impact of PVD on Cardiovascular Risk ===")
    print(f"📊 Cardiovascular Risk without PVD: {without_pvd_prob:.1%}")
    print(f"📊 Cardiovascular Risk with PVD: {with_pvd_prob:.1%}")
    print(f"📈 PVD Increases Cardiovascular Risk by: {pvd_impact:+.1f}%")
    
    # Determine impact severity
    if pvd_impact > 100:
        severity = "Very High"
    elif pvd_impact > 50:
        severity = "High"
    elif pvd_impact > 25:
        severity = "Moderate"
    else:
        severity = "Low"
    
    print(f"⚠️ Impact Level: {severity}")
    
    return severity, pvd_impact

def predict_disease():
    """Main prediction function"""
    feature_names, scaler, models_pvd, models_cardio, cardio_probabilities = load_models()
    if not all([feature_names, scaler, models_pvd, models_cardio, cardio_probabilities]):
        print("Failed to load necessary files. Please ensure all model files are present.")
        return
    
    user_input = get_user_input()
    if user_input is None:
        return
    
    processed_input = preprocess_input(user_input, feature_names, scaler)
    
    print("\nChoose a model for prediction:")
    print("1 - DecisionTree")
    print("2 - RandomForest")
    print("3 - SGD")
    print("4 - XGBoost")
    
    model_choice = input("Enter model number (1-4): ")
    model_mapping = {
        "1": "DecisionTree",
        "2": "RandomForest",
        "3": "SGD",
        "4": "XGBoost"
    }
    
    if model_choice not in model_mapping:
        print("Invalid choice! Defaulting to RandomForest.")
        model_choice = "2"
    
    selected_model_name = model_mapping[model_choice]
    
    # PVD Prediction
    model_pvd = models_pvd[selected_model_name]
    pvd_prediction = model_pvd.predict(processed_input)[0]
    pvd_probability = model_pvd.predict_proba(processed_input)[0][1]
    
    print("\n=== PVD Prediction Results ===")
    print(f"🔍 Prediction: {'PVD Detected' if pvd_prediction == 1 else 'No PVD Detected'}")
    print(f"📊 Probability: {pvd_probability * 100:.2f}% chance of having PVD")
    risk_level = "Low" if pvd_probability < 0.3 else "Moderate" if pvd_probability < 0.7 else "High"
    print(f"⚠️ Risk Level: {risk_level}")
    
    # If PVD detected, check cardiovascular risk
    if pvd_prediction == 1:
        print("\n=== Cardiovascular Disease Risk Assessment ===")
        print("Since PVD was detected, analyzing cardiovascular risk impact...")
        
        model_cardio = models_cardio[selected_model_name]
        cardio_prediction = model_cardio.predict(processed_input)[0]
        cardio_probability = model_cardio.predict_proba(processed_input)[0][1]
        
        # Analyze how PVD increases cardiovascular risk
        impact_severity, impact_percentage = analyze_pvd_impact(cardio_probability, cardio_probabilities)
        
        # Print recommendations based on impact
        print("\n=== Recommendations Based on PVD Impact ===")
        if impact_severity in ["High", "Very High"]:
            print(f"Due to the significant impact of PVD (increasing cardiovascular risk by {impact_percentage:.1f}%):")
            print("1. Urgent cardiovascular specialist consultation recommended")
            print("2. Comprehensive cardiovascular assessment needed")
            print("3. Frequent monitoring of cardiovascular health")
        else:
            print(f"With PVD increasing cardiovascular risk by {impact_percentage:.1f}%:")
            print("1. Regular cardiovascular health monitoring")
            print("2. Scheduled follow-ups with healthcare provider")
        
        # Additional risk factor recommendations
        print("\n=== Risk Factor Management ===")
        if user_input['smoke'] == 1:
            print("• Smoking cessation is crucial - PVD significantly increases this risk")
        if user_input['alco'] == 1:
            print("• Reduce alcohol consumption")
        if user_input['active'] == 0:
            print("• Start a supervised exercise program")
        if user_input['cholesterol'] > 1:
            print("• Strict cholesterol management needed")
        print("• Maintain healthy blood pressure")
        print("• Follow heart-healthy diet")
    
    print("\nNOTE: This is a screening tool only. Please consult with a healthcare professional for proper medical advice and diagnosis.")

if __name__ == "__main__":
    try:
        predict_disease()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try again or contact technical support if the problem persists.")
    finally:
        input("\nPress Enter to exit...")