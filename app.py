from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Initialize the Flask app with the correct template folder
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Check the template folder path and index.html existence
print("Template Folder Path:", os.path.abspath("templates"))
print("Does index.html exist?", os.path.exists(os.path.join("templates", "index.html")))

# Load the model
model = load_model(r"C:\Users\hii\app\models\skin_proj.h5")
print(model.summary())

# Define upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part", 400
        f = request.files["image"]
        if f.filename == "":
            return "No selected file", 400
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)

            # Process the image and make predictions
            img = image.load_img(file_path, target_size=(224, 224))  # Assuming model expects 224x224
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            prediction = model.predict(img_array)

            # Assuming model outputs a label or classification
            prediction_label = "Prediction: " + str(np.argmax(prediction))  # Example, replace with actual logic

            os.remove(file_path)  # Clean up the uploaded file

    return render_template("index.html", prediction=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
