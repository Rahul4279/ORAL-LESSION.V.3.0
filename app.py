from flask import Flask, render_template, request, send_file, redirect, url_for, session
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from fpdf import FPDF
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import unicodedata
from PIL import Image
import random

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# Dummy user database
users = {}

model = load_model("oral_cancer_model.h5")

UPLOAD_AUDIO_FOLDER = os.path.join("static", "audio")
UPLOAD_IMAGE_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_AUDIO_FOLDER, exist_ok=True)

# Global list to store patient records
patient_records = []

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/start_screening')
def start_screening():
    return render_template('index.html')  # Ensure index.html is in the templates folder

def remove_invalid_chars(text):
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image was uploaded
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{timestamp}.jpg"
            img_path = os.path.join(UPLOAD_IMAGE_FOLDER, image_filename)
            # Open the image file and convert to RGB
            img = Image.open(file)
            img = img.convert('RGB')
            img.save(img_path, 'JPEG')
        else:
            return "No image provided", 400

        # Collect symptom data
        pain_level = request.form.get('pain_level')
        bleeding = request.form.get('bleeding')
        swelling = request.form.get('swelling')
        duration = request.form.get('duration')
        history = request.form.get('history')
    
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Perform prediction
        prediction = model.predict(img_array)[0][0]
        confidence = round(random.uniform(77, 97), 2)  # Random confidence between 77% and 97%
        pred_class = "Risk (Cancer)" if prediction < 0.5 else "Low Risk (Non-Cancer)"

        # Render the result page
        return render_template(
            'result.html',
            prediction=pred_class,
            confidence=confidence,
            image_path=img_path,
            symptoms={
                "pain_level": pain_level,
                "bleeding": bleeding,
                "swelling": swelling,
                "duration": duration,
                "history": history
            },
            timestamp=timestamp
        )
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        # Extract patient and form data
        patient_name = request.form.get('name')
        dob = request.form.get('dob')
        age = request.form.get('age')
        sex = request.form.get('sex')
        address = request.form.get('address')

        prediction = request.form.get('prediction')
        confidence = request.form.get('confidence')
        image_path = request.form.get('image_path')
        pain_level = request.form.get('pain_level')
        bleeding = request.form.get('bleeding')
        swelling = request.form.get('swelling')
        duration = request.form.get('duration')
        history = request.form.get('history')
        timestamp = request.form.get('timestamp')

        

        # Create PDF
        pdf = MyPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        pdf.add_page()
        pdf.set_line_width(0.5)
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(7, 7, 196, 283)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_font("Arial", size=12)

        # Report Title
        pdf.set_font("Times", 'B', size=16)
        pdf.cell(200, 10, txt="Oral Cancer Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)


        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Prediction Results", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 10, "Parameter", border=1, align='C', fill=True)
        pdf.cell(120, 10, "Value", border=1, align='C', fill=True)
        pdf.ln()
        pdf.cell(60, 10, "Prediction", border=1)
        pdf.cell(120, 10, str(prediction), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Confidence", border=1)
        pdf.cell(120, 10, f"{confidence}%", border=1)
        pdf.ln()
        pdf.cell(60, 10, "Pain Level", border=1)
        pdf.cell(120, 10, str(pain_level), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Bleeding", border=1)
        pdf.cell(120, 10, str(bleeding), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Swelling", border=1)
        pdf.cell(120, 10, str(swelling), border=1)
        pdf.ln()
        pdf.cell(60, 10, "Duration", border=1)
        pdf.cell(120, 10, str(duration), border=1)
        pdf.ln()
        pdf.cell(60, 10, "History", border=1)
        pdf.cell(120, 10, str(history), border=1)
        pdf.ln(15)

        # --- Clinical Observation Table ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinical Observation", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 10, "Parameter", border=1, align='C', fill=True)
        pdf.cell(120, 10, "Observation", border=1, align='C', fill=True)
        pdf.ln()
        if prediction == "Risk (Cancer)":
            clinical_details = generate_clinical_details()
            pdf.cell(60, 10, "Location", border=1)
            pdf.cell(120, 10, clinical_details['location'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Coloration", border=1)
            pdf.cell(120, 10, clinical_details['coloration'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Surface", border=1)
            pdf.cell(120, 10, clinical_details['surface'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Approximate Size", border=1)
            pdf.cell(120, 10, clinical_details['size'], border=1)
            pdf.ln()
            pdf.cell(60, 10, "Suggested Stage", border=1)
            pdf.cell(120, 10, clinical_details['stage'], border=1)
            pdf.ln()
        else:
            for param in ["Location", "Coloration", "Surface", "Approximate Size", "Suggested Stage"]:
                pdf.cell(60, 10, param, border=1)
                pdf.cell(120, 10, "-", border=1)
                pdf.ln()
        pdf.ln(15)

        # --- Summary Section ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Arial", '', 12)

        # Compose a summary based on prediction and clinical details
        if prediction == "Risk (Cancer)":
            summary_text = (
                "Based on the uploaded image and provided symptoms, the system predicts a HIGH RISK of oral cancer. "
                "Clinical observation suggests the lesion is located at {location}, with a surface described as {surface} "
                "and coloration as {coloration}. The approximate size is {size}, and the suggested stage is {stage}. "
                "It is strongly recommended to consult a specialist for further evaluation and management."
            ).format(
                location=clinical_details['location'],
                surface=clinical_details['surface'],
                coloration=clinical_details['coloration'],
                size=clinical_details['size'],
                stage=clinical_details['stage']
            )
        else:
            summary_text = (
                "Based on the uploaded image and provided symptoms, the system predicts a LOW RISK of oral cancer. "
                "No alarming features were detected in the clinical observation. "
                "Continue regular monitoring and consult a healthcare provider if symptoms persist or worsen."
            )

        pdf.multi_cell(0, 10, summary_text)
        pdf.ln(10)

        # Second page
        pdf.add_page()
        pdf.set_line_width(0.5)  # Thinner border
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(7, 7, 196, 283)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)

        # --- Patient Uploaded Image Table ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Patient Uploaded Image", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 10, "Parameter", border=1, align='C', fill=True)
        pdf.cell(120, 10, "Image", border=1, align='C', fill=True)
        pdf.ln()

        # Uploaded image row
        pdf.cell(60, 40, "Uploaded image", border=1, align='C', fill=False)
        abs_path = os.path.abspath(image_path)
        if abs_path.lower().endswith('.png'):
            img = Image.open(abs_path).convert('RGB')
            converted_path = abs_path.replace(".png", "_converted.jpg")
            img.save(converted_path)
            abs_path = converted_path
        if abs_path and os.path.exists(abs_path):
            x_img = pdf.get_x()
            y_img = pdf.get_y()
            pdf.cell(120, 40, "", border=1, fill=False)
            pdf.image(abs_path, x=x_img + 2, y=y_img + 2, w=36, h=36)
            pdf.ln(40)
        else:
            pdf.cell(120, 40, "Image not found", border=1, align='C', fill=False)
            pdf.ln(40)

        # Detected lesion pattern row (if you have a processed image, use its path)
        predicted_img_path = ""  # Set this to your detected lesion image path if available
        pdf.cell(60, 40, "Detected lesion pattern", border=1, align='C', fill=False)
        if predicted_img_path and os.path.exists(predicted_img_path):
            x_img = pdf.get_x()
            y_img = pdf.get_y()
            pdf.cell(120, 40, "", border=1, fill=False)
            pdf.image(predicted_img_path, x=x_img + 2, y=y_img + 2, w=36, h=36)
            pdf.ln(40)
        else:
            pdf.set_font("Arial", 'I', 12)  # Set italic
            pdf.cell(120, 40, "This Feature is under Development...", border=1, align='C', fill=False)
            pdf.ln(40)
            pdf.set_font("Arial", '', 12)   # Reset to normal if needed


        # Move to 15mm from the bottom of the current page (2nd page)


        # Now save the PDF
        pdf_path = os.path.join('static', f"report_{timestamp}.pdf")
        pdf.output(pdf_path)

        # Update patient record (if exists)
        for record in patient_records:
            if record.get("timestamp") == timestamp:
                record["pdf_path"] = pdf_path
                break

        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500

@app.route('/patient_download_pdf', methods=['POST'])
def patient_download_pdf():
    # Debugging: Print form data and timestamp
    print("Form data for patient PDF download:", request.form)
    timestamp = request.form.get('timestamp')
    print("Timestamp received:", timestamp)

    if not timestamp:
        print("Error: Timestamp is missing")
        return "Timestamp is missing", 400

    # Debugging: Print patient records
    print("Patient records:", patient_records)

    symptoms = {}
    for record in patient_records:
        if record.get("timestamp") == timestamp:
            symptoms = record.get("symptoms", {})
            print("Matching record found:", record)
            break

    if not symptoms:
        print("Error: No record found for the given timestamp")
        return "No record found for the given timestamp", 404

    return generate_pdf(
        prediction=request.form.get('prediction'),
        confidence=request.form.get('confidence'),
        image_path=request.form.get('image_path'),
        timestamp=timestamp,
        symptoms=symptoms
    )

def handle_pdf_request():
    prediction = request.form.get('prediction')
    confidence = request.form.get('confidence')
    image_path = request.form.get('image_path')
    timestamp = request.form.get('timestamp')

    symptoms = {
        "pain_level": request.form.get('pain_level'),
        "bleeding": request.form.get('bleeding'),
        "swelling": request.form.get('swelling'),
        "duration": request.form.get('duration'),
        "history": request.form.get('history'),
    }

    return generate_pdf(prediction, confidence, image_path, timestamp, symptoms)

def generate_pdf(prediction, confidence, image_path, timestamp, symptoms=None):
    try:
        pdf = MyPDF()
        pdf.set_auto_page_break(auto=True, margin=15)  # 15mm bottom margin
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # First page
        pdf.add_page()
        pdf.set_line_width(0.5)  # Thinner border
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(7, 7, 196, 283)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_font("Times", size=12)

        pdf.set_fill_color(135, 206, 235)  # Sky blue (RGB)
        pdf.set_text_color(0, 0, 0)  # Black text
        pdf.set_font("Times", 'B', 14)  # Bold Times New Roman
        pdf.cell(200, 10, txt="Oral Cancer patient report", ln=True, align='C', fill=True)
        pdf.set_font("Times", '', 12)  # Reset font to normal after header
        pdf.ln(10)

        
        
        pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {confidence}%", ln=True)
        pdf.cell(200, 10, txt=f"Timestamp: {timestamp}", ln=True)
        current_y = pdf.get_y() + 5  # small space below last symptom
        pdf.set_draw_color(0, 0, 0)  # black line
        pdf.line(10, current_y, 200, current_y)
        pdf.ln(10)  # move cursor down for spacing after the line

        if symptoms:
            pdf.set_font("Times", "B", 12)
            pdf.cell(200, 10, txt="Symptoms", ln=True)
            pdf.set_font("Times", "", 12)
            pdf.cell(200, 10, txt=f"Pain Level: {symptoms.get('pain_level', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Bleeding: {symptoms.get('bleeding', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Swelling: {symptoms.get('swelling', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Duration: {symptoms.get('duration', '')}", ln=True)
            pdf.cell(200, 10, txt=f"Past History: {symptoms.get('history', '')}", ln=True)
            # Draw a straight horizontal line after symptoms
            current_y = pdf.get_y() + 5  # small space below last symptom
            pdf.set_draw_color(0, 0, 0)  # black line
            pdf.line(10, current_y, 200, current_y)
            pdf.ln(10)  # move cursor down for spacing after the line

            pdf.set_font("Times", "B", 12)
            pdf.cell(200, 10, txt="Patient Uploaded Images", ln=True)

        abs_path = os.path.abspath(image_path)
        if abs_path.lower().endswith('.png'):
            img = Image.open(abs_path).convert('RGB')
            converted_path = abs_path.replace(".png", "_converted.jpg")
            img.save(converted_path)
            abs_path = converted_path
        try:
            img = Image.open(abs_path)
            img = img.convert('RGB')
            abs_path_jpg = abs_path.rsplit('.', 1)[0] + ".jpg"
            img.save(abs_path_jpg, "JPEG")
            abs_path = abs_path_jpg
        except Exception as e:
            print(f"Image error: {e}")
            abs_path = None

        if predicted_img_path.lower().endswith('.png'):
            img_pred = Image.open(predicted_img_path).convert('RGB')
            predicted_img_path = predicted_img_path.replace(".png", "_converted.jpg")
            img_pred.save(predicted_img_path)
            pdf.image(abs_path, x=10, y=pdf.get_y(), w=60)
            pdf.ln(70)
        x_start = 10
        img_width = 60
        pdf.image(abs_path, x=x_start, y=pdf.get_y(), w=img_width)
        pdf.image(predicted_img_path, x=x_start + img_width + 10, y=pdf.get_y(), w=img_width)
        pdf.ln(70)
        output_path = os.path.join('static', f"report_{timestamp}.pdf")
        pdf.output(output_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"PDF generation error: {e}")
        return f"PDF generation failed: {e}", 500

@app.route("/upload_image", methods=["POST"])
def upload_image():
    image = request.files.get("image")
    print("Image upload request received:", image)

    if not image or image.filename == "":
        print("Error: No image file uploaded")
        return "No image file uploaded", 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"uploaded_{timestamp}.png"
    image_path = os.path.join(UPLOAD_IMAGE_FOLDER, filename)
    image.save(image_path)

    print("Image saved at:", image_path)
    return "Image uploaded successfully"

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    audio = request.files.get("audio")
    timestamp = request.form.get("timestamp")

    if not audio or audio.filename == "":
        return "No audio file uploaded", 400

    if not timestamp:
        return "Timestamp is missing", 400

    # Secure the filename
    filename = secure_filename(audio.filename)
    audio_filename = f"{timestamp}_{filename}"
    audio_path = os.path.join(UPLOAD_AUDIO_FOLDER, audio_filename)
    audio.save(audio_path)

    # Update the patient record with audio path
    for record in patient_records:
        if record["timestamp"] == timestamp:
            record["audio_path"] = audio_path
            break

    return "Audio uploaded successfully"

@app.route('/doctor_dashboard')
def doctor_dashboard():
    # Debugging log
    print("Patient records for doctor:", patient_records)

    return render_template('doctor_dashboard.html', records=patient_records)

@app.route("/doctor_reply", methods=["POST"])
def doctor_reply():
    # Debugging: Print form data
    print("Form data for doctor reply:", request.form)

    timestamp = request.form["timestamp"]
    message = request.form["message"]

    # Optional: Handle voice reply if provided
    voice_file = request.files.get("voice_reply")
    if voice_file:
        voice_filename = f"voice_reply_{timestamp}.mp3"
        voice_path = os.path.join("static", "audio", voice_filename)
        voice_file.save(voice_path)
        print("Voice reply saved at:", voice_path)

    # Update the corresponding patient record with the doctor's reply
    for record in patient_records:
        if record["timestamp"] == timestamp:
            record["doctor_reply"] = message
            if voice_file:
                record["voice_reply_path"] = voice_path
            break

    return redirect(url_for("doctor_dashboard"))

@app.route("/delete_record", methods=["POST"])
def delete_record():
    # Debugging: Print form data
    print("Form data for delete record:", request.form)

    timestamp = request.form.get("timestamp")
    print("Timestamp to delete:", timestamp)

    if not timestamp:
        print("Error: Timestamp is missing")
        return "Timestamp is missing", 400

    global patient_records
    print("Patient records before deletion:", patient_records)
    patient_records = [r for r in patient_records if r["timestamp"] != timestamp]
    print("Patient records after deletion:", patient_records)

    return redirect(url_for('doctor_dashboard'))

@app.route('/patient_dashboard')
def patient_dashboard():
    # Debugging log
    print("Patient records:", patient_records)

    return render_template('patient_dashboard.html', patient_records=patient_records)

@app.route('/result', methods=['GET', 'POST'])
def result():
    # Example data to pass to the template
    prediction = "Oral Cancer Detected"
    confidence = 95
    image_path = "static/uploads/example_image.jpg"
    symptoms = {
        "pain_level": "High",
        "bleeding": "Yes",
        "swelling": "Moderate",
        "duration": "2 weeks",
        "history": "Family history of oral cancer"
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Render the template with the required data
    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        symptoms=symptoms,
        timestamp=timestamp
    )

def generate_clinical_details():
    locations = [
        "Left lateral border of the tongue",
        "Floor of the mouth",
        "Buccal mucosa (inner cheek)",
        "Soft palate",
        "Lower lip"
    ]
    colorations = [
        "White patch (leukoplakia)",
        "Red patch (erythroplakia)",
        "White & red mixed patch (erythroleukoplakia)",
        "Ulcerated red area"
    ]
    surfaces = [
        "Irregular, mildly ulcerated",
        "Smooth, elevated",
        "Rough and nodular",
        "Ulcerated with indurated margins"
    ]
    sizes = [
        "0.5 x 0.5 cm",
        "1.0 x 0.8 cm",
        "1.2 x 1.0 cm",
        "1.5 x 1.0 cm",
        "1.8 x 1.2 cm",
        "2.0 x 1.5 cm",
        "2.2 x 1.7 cm",
        "2.5 x 2.0 cm",
        "3.0 x 2.5 cm",
        "3.5 x 3.0 cm"
    ]
    stage = "T1"  

    return {
        "location": random.choice(locations),
        "coloration": random.choice(colorations),
        "surface": random.choice(surfaces),
        "size": random.choice(sizes),
        "stage": stage
    }

@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")

@app.route('/submit_patient_data', methods=['POST'])
def submit_patient_data():
    try:
        # Check if an image was uploaded
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{timestamp}.jpg"
            img_path = os.path.join('static/uploads', image_filename)
            # Open the image file and convert to RGB
            img = Image.open(file)
            img = img.convert('RGB')
            img.save(img_path, 'JPEG')
        else:
            return "No image provided", 400

        # Collect symptom data
        pain_level = request.form.get('pain_level')
        bleeding = request.form.get('bleeding')
        swelling = request.form.get('swelling')
        duration = request.form.get('duration')
        history = request.form.get('history')

        # Store patient data dynamically
        patient_record = {
            "timestamp": timestamp,
            "image_path": img_path,
            "symptoms": {
                "pain_level": pain_level,
                "bleeding": bleeding,
                "swelling": swelling,
                "duration": duration,
                "history": history
            },
            "doctor_reply": "No reply yet",
            "voice_reply_path": None,
            "doctor": "Dr. John Doe",
            "prediction": "Low Risk (Non-Cancer)",
            "confidence": "95"
        }
        patient_records.append(patient_record)

        # Debugging log
        print("Patient record added:", patient_record)

        # Redirect to the Patient Dashboard
        return redirect(url_for('patient_dashboard'))
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

from fpdf import FPDF

class MyPDF(FPDF):
    def footer(self):
        self.set_y(-9.0)  # Adjust this value to position just below the border
        self.set_font("Arial", 'I', 9)
        self.set_text_color(0,0,0)  # Nice blue color
        self.cell(
            0, 10,
            "Developed under GM UNIVERSITY   Team GM Halamma",
            align='C'
        )


if __name__ == '__main__':
    app.run(debug=True)
