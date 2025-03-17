import joblib
import os
import numpy as np

def test_model_loading():
    print("Current working directory:", os.getcwd())
    print("Listing models directory content:")
    models_path = os.path.join(os.getcwd(), "models")
    print(os.listdir(models_path))
    
    # Try to load different models
    try:
        print("\nAttempting to load RandomForest model for cardio...")
        rf_model = joblib.load(os.path.join(models_path, 'RandomForest_model_cardio.pkl'))
        print("RandomForest model loaded successfully!")
        print(f"Model type: {type(rf_model)}")
    except Exception as e:
        print(f"Error loading RandomForest model: {e}")
    
    try:
        print("\nAttempting to load XGBoost model for cardio...")
        xgb_model = joblib.load(os.path.join(models_path, 'XGBoost_model_cardio.pkl'))
        print("XGBoost model loaded successfully!")
        print(f"Model type: {type(xgb_model)}")
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")

if __name__ == "__main__":
    test_model_loading()