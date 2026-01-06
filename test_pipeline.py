import requests
import json

def test_prediction_pipeline():
    """Test the prediction pipeline with sample data"""
    
    # Base URL for your Flask app
    base_url = "http://localhost:5000"
    
    # Test data - valid input
    test_data = {
        "age": 45,
        "gender": 1,  # Male
        "height": 175,
        "weight": 80,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,  # Above normal
        "gluc": 1,  # Normal
        "smoke": 0,  # No
        "alco": 0,  # No
        "active": 1   # Yes
    }
    
    print("Testing Cardiovascular Disease Prediction Pipeline")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Valid prediction
    print("\n2. Testing valid prediction...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get('success'):
            prediction = result['prediction']
            print(f"\n📊 Prediction Results:")
            print(f"   Risk Level: {prediction['risk_level']}")
            print(f"   Probability: {prediction['percentage']}%")
            print(f"   Interpretation: {prediction['interpretation']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Invalid data (age too young)
    print("\n3. Testing invalid data (age too young)...")
    invalid_data = test_data.copy()
    invalid_data['age'] = 15  # Too young
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Invalid data (blood pressure too high)
    print("\n4. Testing invalid data (blood pressure too high)...")
    invalid_data2 = test_data.copy()
    invalid_data2['ap_hi'] = 300  # Too high
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_data2,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Missing data
    print("\n5. Testing missing data...")
    incomplete_data = {
        "age": 45,
        "gender": 1
        # Missing other required fields
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=incomplete_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_multiple_scenarios():
    """Test multiple prediction scenarios"""
    base_url = "http://localhost:5000"
    
    scenarios = [
        {
            "name": "Low Risk Profile",
            "data": {
                "age": 25, "gender": 2, "height": 165, "weight": 60,
                "ap_hi": 110, "ap_lo": 70, "cholesterol": 1, "gluc": 1,
                "smoke": 0, "alco": 0, "active": 1
            }
        },
        {
            "name": "High Risk Profile",
            "data": {
                "age": 65, "gender": 1, "height": 170, "weight": 95,
                "ap_hi": 180, "ap_lo": 110, "cholesterol": 3, "gluc": 3,
                "smoke": 1, "alco": 1, "active": 0
            }
        },
        {
            "name": "Medium Risk Profile",
            "data": {
                "age": 45, "gender": 1, "height": 175, "weight": 80,
                "ap_hi": 140, "ap_lo": 90, "cholesterol": 2, "gluc": 1,
                "smoke": 0, "alco": 0, "active": 1
            }
        }
    ]
    
    print("\n" + "=" * 50)
    print("Testing Multiple Risk Scenarios")
    print("=" * 50)
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=scenario['data'],
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    pred = result['prediction']
                    print(f"   Risk Level: {pred['risk_level']} ({pred['percentage']}%)")
                    print(f"   Color: {pred['risk_color']}")
                else:
                    print(f"   Error: {result.get('error', {}).get('message')}")
            else:
                print(f"   HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   Exception: {e}")

if __name__ == "__main__":
    print("Make sure your Flask app is running on http://localhost:5000")
    print("Run: python app.py")
    input("Press Enter when ready to test...")
    
    test_prediction_pipeline()
    test_multiple_scenarios()