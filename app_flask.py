from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Flask ML API Running"})

@app.route("/predict", methods=["GET"])
def predict():
    value = float(request.args.get("value"))
    prediction = model.predict(np.array([[value]]))
    return jsonify({"input": value, "prediction": float(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)