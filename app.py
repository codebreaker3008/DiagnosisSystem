from flask import Flask, render_template, request
from liver.liver_prediction import predict_liver_disease
from heart.heart_prediction import heart_bp, predict_heart_disease
import google.generativeai as genai
from markdown import markdown  # ✅ Add this at the top of app.py

# Import Markup from correct module
from markupsafe import Markup

# ✅ Brain tumor + X-ray
import os
from werkzeug.utils import secure_filename
from brain.brain_prediction import predict_brain_tumor
from xray.xray_prediction import predict_fracture

app = Flask(__name__)
app.register_blueprint(heart_bp)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

genai.configure(api_key="AIzaSyDOW5mx7JMAyAShVemo1g5u8dK2bbwMr1A")  # Your Gemini API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

# ✅ Liver
@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            gender = 1 if request.form['gender'].lower() == 'male' else 0
            tb = float(request.form['total_bilirubin'])
            db = float(request.form['direct_bilirubin'])
            ap = float(request.form['alkaline_phosphotase'])
            alt = float(request.form['alamine_aminotransferase'])
            ast = float(request.form['aspartate_aminotransferase'])
            tp = float(request.form['total_proteins'])
            alb = float(request.form['albumin'])
            agr = float(request.form['ag_ratio'])

            input_data = [age, gender, tb, db, ap, alt, ast, tp, alb, agr]
            result = predict_liver_disease(input_data)

            return render_template('liver_result.html', result=result)
        except Exception as e:
            return f"Error: {e}"

@app.route('/liver_form')
def liver_form():
    return render_template('liver_form.html')

# ✅ Heart
@app.route('/heart_form')
def heart_form():
    return render_template('heart_form.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            result = predict_heart_disease(input_data)

            return render_template('heart_result.html', result=result)
        except Exception as e:
            return f"Error: {e}"

# ✅ Brain tumor
@app.route('/brain_form')
def brain_form():
    return render_template('brain_form.html')

@app.route('/predict_brain', methods=['POST'])
def predict_brain():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_brain_tumor(filepath)
        return render_template('brain_result.html', result=result)

# ✅ X-ray fracture detection
@app.route('/xray_form')
def xray_form():
    return render_template('xray_form.html')

@app.route('/predict_xray', methods=['POST'])
def predict_xray():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_fracture(filepath)
        return render_template('xray_result.html', result=result)

# ✅ Symptoms AI-based diagnosis
@app.route('/symptoms_form')
def symptoms_form():
    return render_template('symptoms_form.html')

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    symptoms = request.form['symptoms']
    prompt = f"""
    A patient has the following symptoms: {symptoms}.
    Provide the most probable disease(s), their causes, recommended medications, preventive measures, and a short explanation about the disease.
    Format the output as:

    **1. Probable Disease(s):**
    - Disease 1
    - Disease 2
    **2. Causes:**
    - Cause 1
    - Cause 2
    - Cause 3
    **3. Medications:**
    - Med 1
    **4. Preventive Measures:**
    - Measure 1
    **5. More About the Disease:**
    Description here.
    """

    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        result_text = response.text

        # ✅ Convert markdown-style formatting to actual HTML
        html_result = markdown(result_text)

        return render_template('symptoms_result.html', symptoms=symptoms, result=Markup(html_result))

    except Exception as e:
        return render_template('symptoms_result.html', symptoms=symptoms, result=Markup(f"<p>An error occurred: {e}</p>"))


if __name__ == '__main__':
    app.run(debug=True)
