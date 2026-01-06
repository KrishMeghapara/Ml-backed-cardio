import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

def clean_data(df):
    """Clean the cardiovascular dataset"""
    print("Original dataset shape:", df.shape)
    
    # Convert age from days to years
    df['age'] = df['age'] / 365
    
    # Remove outliers and invalid values
    # Remove negative blood pressure values
    df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]
    
    # Remove unrealistic blood pressure values
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]
    
    # Remove unrealistic height values
    df = df[(df['height'] >= 120) & (df['height'] <= 220)]
    
    # Remove unrealistic weight values
    df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
    
    # Remove unrealistic age values
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]
    
    print("Cleaned dataset shape:", df.shape)
    return df

def create_features(df):
    """Create additional features"""
    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # You can add more features here if needed
    # df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    return df

def train_model():
    """Train the cardiovascular disease prediction model"""
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("cardio_train.csv", sep=';')
    
    # Clean the data
    df_clean = clean_data(df.copy())
    
    # Create features
    df_features = create_features(df_clean)
    
    # Prepare features and target
    # Remove 'id' column as it's not useful for prediction
    feature_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    
    X = df_features[feature_columns]
    y = df_features['cardio']
    
    print("Feature columns:", feature_columns)
    print("Features shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train RandomForest model
    print("\nTraining RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model
    model_filename = 'cardio_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nModel saved as {model_filename}")
    
    # Save feature names for reference
    feature_info = {
        'feature_columns': feature_columns,
        'model_accuracy': accuracy,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    import json
    with open('model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("Model information saved as model_info.json")
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = train_model()
    print(f"\nModel training completed with accuracy: {accuracy:.4f}")
    print("You can now use the trained model in your Flask application!")