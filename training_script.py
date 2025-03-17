import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE

def validate_data(df):
    """Validate and clean the input data with realistic constraints"""
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Define realistic ranges
    realistic_ranges = {
        'age': (18, 100),
        'height': (140, 220),  # cm
        'weight': (40, 200),   # kg
        'ap_hi': (80, 220),    # systolic
        'ap_lo': (40, 140),    # diastolic
        'cholesterol': (1, 3),
        'gluc': (1, 3),
        'smoke': (0, 1),
        'alco': (0, 1),
        'active': (0, 1)
    }
    
    # Apply realistic ranges
    for column, (min_val, max_val) in realistic_ranges.items():
        mask = (df[column] >= min_val) & (df[column] <= max_val)
        if column in ['ap_hi', 'ap_lo']:
            # Additional check for blood pressure relationship
            mask = mask & (df['ap_hi'] > df['ap_lo'])
        df = df[mask]
    
    return df

def train_models():
    # Load the dataset
    print("Loading and preprocessing data...")
    df = pd.read_csv("converted_cardio_train_with_PVD_years.csv")
    
    # Drop unnecessary columns
    df = df.drop(columns=['id', 'BMI'], errors='ignore')
    
    # Validate and clean data
    df = validate_data(df)
    
    # Calculate and save baseline cardio probabilities
    baseline_cardio_prob = df['cardio'].mean()
    cardio_prob_with_pvd = df[df['PVD'] == 1]['cardio'].mean()
    cardio_prob_without_pvd = df[df['PVD'] == 0]['cardio'].mean()
    
    cardio_probabilities = {
        'baseline': baseline_cardio_prob,
        'with_pvd': cardio_prob_with_pvd,
        'without_pvd': cardio_prob_without_pvd
    }
    
    print(f"\nBaseline Cardiovascular Disease Probabilities:")
    print(f"Overall population: {baseline_cardio_prob:.2%}")
    print(f"Population with PVD: {cardio_prob_with_pvd:.2%}")
    print(f"Population without PVD: {cardio_prob_without_pvd:.2%}")
    
    # Save cardio probabilities with protocol=4
    joblib.dump(cardio_probabilities, 'models/cardio_probabilities.pkl', protocol=4)
    
    # Define features
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
               'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    # Create and save scaler
    scaler = MinMaxScaler()
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    joblib.dump(scaler, 'models/feature_scaler.pkl', protocol=4)
    
    # Save feature names
    pd.DataFrame({'Feature Names': features}).to_csv('models/feature_names.csv', index=False)
    
    # Prepare data for PVD prediction
    print("\nPreparing data for PVD prediction...")
    X_pvd = df[features]
    y_pvd = df['PVD']
    
    # Apply SMOTE for PVD
    smote = SMOTE(sampling_strategy=0.7, random_state=42)  # Reduced from 1.0 to 0.7
    X_pvd_balanced, y_pvd_balanced = smote.fit_resample(X_pvd, y_pvd)
    
    # Split PVD data
    X_train_pvd, X_test_pvd, y_train_pvd, y_test_pvd = train_test_split(
        X_pvd_balanced, y_pvd_balanced, test_size=0.3, random_state=42
    )
    
    # Define models for PVD
    models_pvd = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=3,  # Reduced from 5
            min_samples_split=10,  # Increased from default
            min_samples_leaf=8,    # Added to prevent overfitting
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,      # Reduced from 100
            max_depth=3,          # Reduced from 5
            min_samples_split=10,  # Increased
            min_samples_leaf=8,    # Increased
            max_features='sqrt',   # Added to reduce complexity
            random_state=42
        ),
        "SGD": SGDClassifier(
            loss='log_loss',
            alpha=0.01,           # Increased regularization
            max_iter=100,         # Reduced from 1000
            random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            max_depth=2,          # Reduced from 3
            learning_rate=0.01,   # Reduced from 0.1
            n_estimators=50,      # Reduced from 100
            subsample=0.7,        # Added to reduce overfitting
            colsample_bytree=0.7, # Added to reduce overfitting
            min_child_weight=5,   # Increased to reduce overfitting
            gamma=1,              # Added to reduce overfitting
            random_state=42
        )
    }
    
    # Train and evaluate PVD models
    print("\nTraining PVD models...")
    metrics_pvd = {}
    
    for name, model in models_pvd.items():
        print(f"\nTraining {name} for PVD...")
        model.fit(X_train_pvd, y_train_pvd)
        
        # Save model with protocol=4
        joblib.dump(model, f'models/{name}_model_pvd.pkl', protocol=4)
        
        # Evaluate
        y_pred = model.predict(X_test_pvd)
        metrics_pvd[name] = {
            'Accuracy': accuracy_score(y_test_pvd, y_pred),
            'Precision': precision_score(y_test_pvd, y_pred),
            'Recall': recall_score(y_test_pvd, y_pred),
            'F1': f1_score(y_test_pvd, y_pred),
            'AUC-ROC': roc_auc_score(y_test_pvd, y_pred)
        }
        print(f"{name} metrics:", metrics_pvd[name])
    
    # Prepare data for Cardio prediction (only for PVD positive cases)
    print("\nPreparing data for Cardio prediction...")
    df_cardio = df[df['PVD'] == 1]
    X_cardio = df_cardio[features]
    y_cardio = df_cardio['cardio']
    
    # Apply SMOTE for Cardio
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_cardio_balanced, y_cardio_balanced = smote.fit_resample(X_cardio, y_cardio)
    
    # Split Cardio data
    X_train_cardio, X_test_cardio, y_train_cardio, y_test_cardio = train_test_split(
        X_cardio_balanced, y_cardio_balanced, test_size=0.3, random_state=42
    )
    
    # Define models for Cardio
    models_cardio = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SGD": SGDClassifier(
        loss="log_loss",  # Use log loss for probabilistic classification
        max_iter=5000,  # Increase iterations for better convergence
        alpha=0.0001,  # Reduce regularization to allow better learning
        penalty="elasticnet",  # L1 + L2 regularization to avoid overfitting
        l1_ratio=0.15,  # Balance L1 and L2 regularization
        learning_rate="adaptive",  # Keeps learning rate stable for more iterations
        eta0=0.02,  # Slightly higher initial learning rate
        tol=1e-4,  # More precise convergence criteria
        power_t=0.5,  # Adjust decay rate for learning rate
        random_state=42
),
        "XGBoost": xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
    }
    
    # Train and evaluate Cardio models
    print("\nTraining Cardio models...")
    metrics_cardio = {}
    
    for name, model in models_cardio.items():
        print(f"\nTraining {name} for Cardio...")
        model.fit(X_train_cardio, y_train_cardio)
        
        # Save model with protocol=4
        joblib.dump(model, f'models/{name}_model_cardio.pkl', protocol=4)
        
        # Evaluate
        y_pred = model.predict(X_test_cardio)
        metrics_cardio[name] = {
            'Accuracy': accuracy_score(y_test_cardio, y_pred),
            'Precision': precision_score(y_test_cardio, y_pred),
            'Recall': recall_score(y_test_cardio, y_pred),
            'F1': f1_score(y_test_cardio, y_pred),
            'AUC-ROC': roc_auc_score(y_test_cardio, y_pred)
        }
        print(f"{name} metrics:", metrics_cardio[name])
    
    # Save metrics
    with open('model_metrics.txt', 'w') as f:
        f.write("PVD Models Metrics:\n")
        for model, metrics in metrics_pvd.items():
            f.write(f"\n{model}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nCardio Models Metrics:\n")
        for model, metrics in metrics_cardio.items():
            f.write(f"\n{model}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        # Add cardio probabilities to metrics file
        f.write("\nCardiovascular Disease Baseline Probabilities:\n")
        f.write(f"Overall population: {baseline_cardio_prob:.4f}\n")
        f.write(f"Population with PVD: {cardio_prob_with_pvd:.4f}\n")
        f.write(f"Population without PVD: {cardio_prob_without_pvd:.4f}\n")

if __name__ == "__main__":
    train_models()
    print("\nTraining complete! All models and necessary files have been saved.")