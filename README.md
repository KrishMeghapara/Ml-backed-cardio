# CardioPredict Backend - AI Health Assessment API

Flask-based REST API for cardiovascular disease risk prediction using machine learning.

## 🔧 Tech Stack
- Python 3.9+
- Flask
- scikit-learn (Random Forest)
- Pandas, NumPy

## 📁 Project Structure
```
backend/
├── app.py              # Main Flask application
├── model/              # ML model files
│   ├── cardio_model.pkl
│   └── scaler.pkl
├── requirements.txt    # Python dependencies
├── Procfile           # Render deployment
└── README.md
```

## 🚀 Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The API will be available at `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Predict Risk
```
POST /predict
Content-Type: application/json

{
  "age": 45,
  "gender": 1,
  "height": 175,
  "weight": 80,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

## 🌐 Deployment on Render

1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3

## 📄 License
MIT License