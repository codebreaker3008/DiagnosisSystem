import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_brain_data(base_dir='brain', image_size=(150, 150)):
    categories = ['no_tumor', 'tumor']
    X, y = [], []

    for label, category in enumerate(categories):
        folder_path = os.path.join(base_dir, 'train', category)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(image_path).resize(image_size).convert('RGB')
                    img_array = np.array(img) / 255.0
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=2)
    return train_test_split(X, y, test_size=0.2, random_state=42)
