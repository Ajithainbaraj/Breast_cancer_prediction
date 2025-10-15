from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = load_model("breast_cancer_dnn.h5")
scaler = joblib.load("scaler.pkl")

# Expected features
important_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # ✅ Allow patient_name but ignore it for prediction
    patient_name = data.get("patient_name", "Unknown")

    # ✅ Try to get features safely
    try:
        inputs = [data[feature] for feature in important_features]
    except KeyError:
        return jsonify({"error": f"Expected 5 features: {important_features}"}), 400

    # ✅ Scale + predict
    inputs_scaled = scaler.transform([inputs])
    prediction = model.predict(inputs_scaled)
    pred_class = "Malignant" if prediction[0][0] > 0.5 else "Benign"

    # ✅ Return patient name + prediction
    return jsonify({
        "patient_name": patient_name,
        "prediction": pred_class
    })

if __name__ == '__main__':
    app.run(debug=True)
