from flask import Blueprint
import joblib
import os
import numpy as np

# Optional: define Blueprint if you're registering it
heart_bp = Blueprint('heart', __name__)

# Load model (adjust path to your model file if needed)
model_path = os.path.join(os.path.dirname(__file__), 'heart_model.pkl')
model = joblib.load(model_path)

def predict_heart_disease(input_data):
    """
    Predict heart disease from input features (13 values).
    """
    input_array = np.array([input_data])  # wrap in array for sklearn
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        return "The person is likely to have heart disease."
    else:
        return "The person is unlikely to have heart disease."
