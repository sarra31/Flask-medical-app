import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Secret key for sessions
app.secret_key = 'your_secret_key_here'

# Configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='rendez_vous',
    )
    return connection

# Route for home page
@app.route('/')
def default():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username and password match
        if username == 'doctor' and password == 'doctor123':  # Hardcoded credentials for the doctor
            session['logged_in'] = True
            return redirect(url_for('liste_rdv', page_num=1))  # Redirect to the appointment list
        else:
            return "Identifiants incorrects. Veuillez réessayer."

    return render_template('login.html')  # Render the login page

# Route for the appointment form
@app.route('/index')
def index():
    return render_template('index.html')

# Route to add an appointment
@app.route('/add-appointment', methods=['POST'])
def add_appointment():
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from tensorflow.keras.models import load_model
    from werkzeug.utils import secure_filename
    import os
    print("[DEBUG] Received request to process tuberculosis info.")

    # Load the pre-trained model
    try:
        print("[DEBUG] Loading the model...")
        model = load_model("my_model.h5")
        print("Model loaded successfully.")
    except ValueError as e:
        print("Error loading model:", e)
        return "An error occurred while loading the model. Please check the model file."

    patient_name = request.form['patient_name']
    patient_email = request.form['patient_email']
    patient_phone = request.form['patient_phone']
    appointment_date = request.form['appointment_date']
    appointment_time = request.form['appointment_time']
    patient_gender = request.form['patient_gender']
    appointment_reason = request.form['appointment_reason']
    patient_photo = request.files.get('patient_photo')

    try:
        # Save the uploaded photo
        print("[DEBUG] Saving uploaded photo...")
        app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(patient_photo.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        patient_photo.save(file_path)
        print(f"[DEBUG] Photo saved to {file_path}")

        # Preprocess the image for prediction
        print("[DEBUG] Preprocessing image...")
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        image = image.resize((512, 512))  # Resize to match model input shape
        image = np.array(image) / 255.0  # Normalize pixel values
        print(f"[DEBUG] Image shape after conversion: {image.shape}")
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        print(f"[DEBUG] Image shape after adding batch and channel dimensions: {image.shape}")

        # Make the prediction
        print("[DEBUG] Making prediction...")
        prediction = model.predict(image)
        print("Prediction shape:", prediction.shape)

        # Aggregate predictions for the entire image
        collapsed_prediction = np.squeeze(prediction)  # Remove unnecessary dimensions
        aggregate_confidence = np.mean(collapsed_prediction)  # Average confidence
        print(f"[DEBUG] Aggregate confidence: {aggregate_confidence}")

        # Threshold to determine class (adjust as needed)
        threshold = 0.5
        predicted_class = 1 if aggregate_confidence > threshold else 0

        # Convert confidence to native Python float
        confidence = float(aggregate_confidence)
        print(f"[DEBUG] Predicted class: {predicted_class}, Confidence: {confidence}")

        # Map the predicted class to a label
        class_labels = {0: "Negative for Tuberculosis", 1: "Positive for Tuberculosis"}
        result = class_labels[predicted_class]
        print(f"[DEBUG] Classification result: {result}")
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO `rdv` (`nom`, `email`, `num`, `date`, `time`, `genre`, `motif`, `photo`, `result`, `confidence`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (patient_name, patient_email, patient_phone, appointment_date, appointment_time, patient_gender, appointment_reason, filename, result, confidence)
        )
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print("Error while inserting data:", e)
        return "Une erreur s'est produite lors de la prise du rendez-vous."

    return redirect(url_for('index'))

#tuberculose form
@app.route('/tuberculose_page')
def tuberculose_page():
    return render_template('tuberculose_page.html')

@app.route('/submit-tuberculosis-info', methods=['POST'])
def submit_tuberculosis_info():
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from tensorflow.keras.models import load_model
    from werkzeug.utils import secure_filename
    import os
    print("[DEBUG] Received request to process tuberculosis info.")

    # Load the pre-trained model
    try:
        print("[DEBUG] Loading the model...")
        model = load_model("my_model.h5")
        print("Model loaded successfully.")
    except ValueError as e:
        print("Error loading model:", e)
        return "An error occurred while loading the model. Please check the model file."

    # Collect form data
    print("[DEBUG] Extracting form data...")
    patient_name = request.form['patient_name']
    patient_age = request.form['patient_age']
    patient_gender = request.form['patient_gender']
    patient_photo = request.files.get('patient_photo')
    print("[DEBUG] Form data extracted successfully.")

    try:
        # Save the uploaded photo
        print("[DEBUG] Saving uploaded photo...")
        app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(patient_photo.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        patient_photo.save(file_path)
        print(f"[DEBUG] Photo saved to {file_path}")

        # Preprocess the image for prediction
        print("[DEBUG] Preprocessing image...")
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        image = image.resize((512, 512))  # Resize to match model input shape
        image = np.array(image) / 255.0  # Normalize pixel values
        print(f"[DEBUG] Image shape after conversion: {image.shape}")
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        print(f"[DEBUG] Image shape after adding batch and channel dimensions: {image.shape}")

        # Make the prediction
        print("[DEBUG] Making prediction...")
        prediction = model.predict(image)
        print("Prediction shape:", prediction.shape)

        # Aggregate predictions for the entire image
        collapsed_prediction = np.squeeze(prediction)  # Remove unnecessary dimensions
        aggregate_confidence = np.mean(collapsed_prediction)  # Average confidence
        print(f"[DEBUG] Aggregate confidence: {aggregate_confidence}")

        # Threshold to determine class (adjust as needed)
        threshold = 0.5
        predicted_class = 1 if aggregate_confidence > threshold else 0

        # Convert confidence to native Python float
        confidence = float(aggregate_confidence)
        print(f"[DEBUG] Predicted class: {predicted_class}, Confidence: {confidence}")

        # Map the predicted class to a label
        class_labels = {0: "Negative for Tuberculosis", 1: "Positive for Tuberculosis"}
        result = class_labels[predicted_class]
        print(f"[DEBUG] Classification result: {result}")

        # Insert the data along with the result into the database
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO `tb` (`nom`, `age`, `genre`, `photo`, `result`, `confidence`) VALUES (%s, %s, %s, %s, %s, %s)",
            (patient_name, patient_age, patient_gender, filename, result, confidence)
        )
        connection.commit()
        cursor.close()
        connection.close()

        # Return the result to the user
        return render_template('tuberculose_page.html', name=patient_name, result=result, confidence=confidence)

    except Exception as e:
        print("Error while processing tuberculosis info:", e)
        return f"An error occurred while processing the form: {str(e)}"

# Route for listing appointments (only accessible for logged-in users) with pagination
@app.route('/liste_rdv/<int:page_num>')
def liste_rdv(page_num):
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    # Number of items per page
    appointments_per_page = 5
    
    # Calculate the start and end for pagination
    start = (page_num - 1) * appointments_per_page
    end = start + appointments_per_page

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Get the total number of appointments to calculate page count
    cursor.execute("SELECT COUNT(*) AS count FROM rdv")
    total_appointments = cursor.fetchone()['count']
    
    # Get the appointments for the current page
    cursor.execute("SELECT * FROM rdv LIMIT %s, %s", (start, appointments_per_page))
    rendez_vous = cursor.fetchall()

    cursor.close()
    connection.close()

    # Calculate the total number of pages
    total_pages = (total_appointments + appointments_per_page - 1) // appointments_per_page

    # Determine if we have previous and next pages
    prev_page = page_num - 1 if page_num > 1 else None
    next_page = page_num + 1 if page_num < total_pages else None

    return render_template('liste_rdv.html', 
                           rendez_vous=rendez_vous,
                           page_num=page_num,
                           total_pages=total_pages,
                           prev_page=prev_page,
                           next_page=next_page)

# Default route for 'liste_rdv' (redirects to page 1)
@app.route('/liste_rdv')
def default_liste_rdv():
    return redirect(url_for('liste_rdv', page_num=1))  # Redirect to page 1 if no page is specified

if __name__ == '__main__':
    app.run(debug=True)