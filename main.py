from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "ML API is running successfully!"}

@app.get("/predict")
def predict(value: float):
    prediction = model.predict(np.array([[value]]))
    return {"input": value, "prediction": float(prediction[0])}
