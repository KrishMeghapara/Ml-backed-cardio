from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Configure CORS for production - allow specific origins
cors_origins = os.environ.get('CORS_ORIGINS', '*').split(',')
CORS(app, origins=cors_origins, methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type'])

# Global variable to store the model
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        # Use decision tree model (smaller file size - 40MB)
        model_path = 'decision_tree_model.pkl'
        if os.path.exists(model_path):
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}!")
        else:
            # Fallback to joblib format
            model_path_alt = 'cardio_model.joblib'
            if os.path.exists(model_path_alt):
                model = joblib.load(model_path_alt)
                print(f"Model loaded successfully from {model_path_alt}!")
            else:
                print("Model file not found. Please train and save your model first.")
                return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    return True

def validate_input(data):
    """Validate user input data"""
    errors = []
    
    # Age validation (18-100 years)
    if 'age' not in data or not isinstance(data['age'], (int, float)):
        errors.append("Age is required and must be a number")
    elif data['age'] < 18 or data['age'] > 100:
        errors.append("Age must be between 18 and 100 years")
    
    # Gender validation (1 or 2)
    if 'gender' not in data or data['gender'] not in [1, 2]:
        errors.append("Gender is required (1 for male, 2 for female)")
    
    # Height validation (120-220 cm)
    if 'height' not in data or not isinstance(data['height'], (int, float)):
        errors.append("Height is required and must be a number")
    elif data['height'] < 120 or data['height'] > 220:
        errors.append("Height must be between 120 and 220 cm")
    
    # Weight validation (30-200 kg)
    if 'weight' not in data or not isinstance(data['weight'], (int, float)):
        errors.append("Weight is required and must be a number")
    elif data['weight'] < 30 or data['weight'] > 200:
        errors.append("Weight must be between 30 and 200 kg")
    
    # Blood pressure validation
    if 'ap_hi' not in data or not isinstance(data['ap_hi'], (int, float)):
        errors.append("Systolic blood pressure is required and must be a number")
    elif data['ap_hi'] < 80 or data['ap_hi'] > 250:
        errors.append("Systolic blood pressure must be between 80 and 250 mmHg")
    
    if 'ap_lo' not in data or not isinstance(data['ap_lo'], (int, float)):
        errors.append("Diastolic blood pressure is required and must be a number")
    elif data['ap_lo'] < 50 or data['ap_lo'] > 150:
        errors.append("Diastolic blood pressure must be between 50 and 150 mmHg")
    
    # Cholesterol validation (1, 2, or 3)
    if 'cholesterol' not in data or data['cholesterol'] not in [1, 2, 3]:
        errors.append("Cholesterol level is required (1=normal, 2=above normal, 3=well above normal)")
    
    # Glucose validation (1, 2, or 3)
    if 'gluc' not in data or data['gluc'] not in [1, 2, 3]:
        errors.append("Glucose level is required (1=normal, 2=above normal, 3=well above normal)")
    
    # Binary validations (0 or 1)
    binary_fields = ['smoke', 'alco', 'active']
    for field in binary_fields:
        if field not in data or data[field] not in [0, 1]:
            errors.append(f"{field.capitalize()} is required (0=no, 1=yes)")
    
    return errors

def preprocess_data(data):
    """Preprocess the input data for model prediction"""
    # Create a DataFrame with the input data
    df = pd.DataFrame([data])
    
    # Calculate BMI (Body Mass Index)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Select features in the same order as training
    # Note: Adjust this based on your actual model features
    feature_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    
    return df[feature_columns]

def format_prediction_result(prediction_proba):
    """Format the prediction result for the frontend"""
    # Get probability of cardiovascular disease (class 1)
    probability = float(prediction_proba[0][1])  # Probability of positive class
    percentage = round(probability * 100, 1)
    
    # Determine risk level
    if percentage < 30:
        risk_level = "Low"
        risk_color = "green"
    elif percentage < 70:
        risk_level = "Medium"
        risk_color = "orange"
    else:
        risk_level = "High"
        risk_color = "red"
    
    return {
        "probability": probability,
        "percentage": percentage,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "interpretation": get_risk_interpretation(percentage)
    }

def get_risk_interpretation(percentage):
    """Get human-readable interpretation of the risk"""
    if percentage < 30:
        return "Your cardiovascular disease risk appears to be low based on the provided health indicators."
    elif percentage < 70:
        return "Your cardiovascular disease risk appears to be moderate. Consider consulting with a healthcare professional."
    else:
        return "Your cardiovascular disease risk appears to be high. It's recommended to consult with a healthcare professional soon."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "success": False,
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": "Machine learning model is not available"
                },
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": {
                    "code": "NO_DATA",
                    "message": "No input data provided"
                },
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Validate input data
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed",
                    "details": validation_errors
                },
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Format result
        result = format_prediction_result(prediction_proba)
        
        return jsonify({
            "success": True,
            "prediction": result,
            "message": "Prediction completed successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": {
                "code": "PREDICTION_ERROR",
                "message": "An error occurred during prediction",
                "details": str(e)
            },
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Cardiovascular Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    })

if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        print("Starting Flask application...")
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_ENV') == 'development'
        app.run(debug=debug, host='0.0.0.0', port=port)
    else:
        print("Failed to load model. Please check the model file.")
else:
    # For gunicorn: load model when imported
    load_model()