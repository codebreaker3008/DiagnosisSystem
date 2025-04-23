# brain/brain_prediction.py
import numpy as np  
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
model = load_model('brain/model/brain_tumor_model.h5')

def predict_brain_tumor(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Brain Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    return result
