import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('xray/xray_model.h5')

def predict_fracture(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0

    prediction = model.predict(img_tensor)[0][0]
    return "Fractured" if prediction < 0.5 else "Not Fractured"
