# Disease Diagnosis System

## Overview

The **Disease Diagnosis System** is a web application designed to assist users in diagnosing diseases based on various inputs such as symptoms, medical imaging, and pre-trained models. It includes detection modules for liver disease, brain tumors (using MRI scans), X-ray fractures, and heart disease. The system also features a symptom-based disease prediction module, where users can enter symptoms and receive probable diseases, suggested medications, causes, and preventive measures.

## Features

- **Liver Disease Detection**: A pre-trained model that analyzes liver disease based on input data.
- **Brain Tumor Detection (MRI)**: Detects brain tumors from MRI scans using a deep learning model.
- **X-ray Fracture Detection**: Identifies fractures in X-ray images.
- **Heart Disease Detection**: Analyzes data to predict heart disease.
- **Symptom-Based Diagnosis**: Users can input symptoms, separated by commas, and the system will predict potential diseases, along with suggestions for treatments and preventive measures.

## Modules

1. **Liver Disease Module**: This module is already completed with a styled interface and trained model.
2. **Brain Tumor Module**: Using MRI scans to detect brain tumors.
3. **X-ray Fracture Module**: Detects fractures from X-ray images.
4. **Heart Disease Module**: Predicts the likelihood of heart disease.
5. **Symptom-Based Diagnosis**: Uses an API (Gemini or similar) to analyze symptoms and suggest probable diseases.

## Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, CSS (Styled interfaces for each module)
- **Machine Learning**: Pre-trained models (TensorFlow, Keras) for disease detection.
- **Symptom-based Diagnosis API**: Gemini API or similar.
- **Libraries**:
  - **Flask**: For backend and API handling.
  - **TensorFlow**: For deep learning model usage.
  - **Keras**: For building neural networks.
  - **Pandas**: For data manipulation.
  - **NumPy**: For numerical operations.
  - **OpenCV**: For image processing.
  - **Jinja2**: For templating in Flask.

## Models

Please note that the pre-trained models used for disease detection are **not included** in this repository due to their large file sizes. You will need to train the models yourself or acquire them from external sources. You can follow the training process outlined in the documentation to generate the models.

If you prefer to use the models directly, please refer to the official sources or pre-trained versions that can be found on research platforms or model repositories.

## Installation

### Prerequisites

- Python 3.6+
- Pip (Python Package Installer)

### Steps to Install

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/disease-diagnosis-system.git
   cd disease-diagnosis-system
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
python app.py
The web app should now be running at http://127.0.0.1:5000/.

Usage
Open the app in a browser by navigating to http://127.0.0.1:5000/.

Choose the disease detection module you want to use:

Liver Disease Detection: Enter the necessary details.

Brain Tumor Detection: Upload an MRI scan.

X-ray Fracture Detection: Upload an X-ray image.

Heart Disease Detection: Enter medical data.

For symptom-based diagnosis, enter symptoms separated by commas, and click "Diagnose."

The system will display the diagnosis, including the disease name, suggested treatments, causes, and preventive measures.

Future Enhancements
Adding more disease detection modules.

Improving the accuracy of disease prediction.

Supporting additional types of medical imaging for detection (e.g., CT scans, ultrasounds).

User authentication to store diagnosis history and reports.

Contributing
We welcome contributions! If you want to contribute to this project, please fork the repository, make changes, and create a pull request.

Fork the repository.

Create a new branch (git checkout -b feature-branch).

Commit your changes (git commit -am 'Add new feature').

Push to the branch (git push origin feature-branch).

Create a new pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
