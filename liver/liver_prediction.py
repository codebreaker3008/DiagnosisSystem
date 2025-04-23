import numpy as np # type: ignore 
import pickle
import os

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'liver_model.pkl')
with open(model_path, 'rb') as f:
    liver_model = pickle.load(f)

def predict_liver_disease(input_data):
    """
    input_data: list or array of input features (e.g., [Age, Gender, Total_Bilirubin, ...])
    returns: prediction result as string
    """
    input_array = np.array(input_data).reshape(1, -1)
    prediction = liver_model.predict(input_array)

    return "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease Detected"
    return render_template('result.html', prediction=result_text)
