from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
model = load_model(r"C:\Users\hii\OneDrive\Desktop\app\models\skin_proj.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the image file from the request
        f = request.files["image"]

        # Save the image file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # Load and preprocess the image for prediction
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize pixel values to [0, 1]

        # Make prediction
        prediction = model.predict(x)

        # Get the predicted class label
        classes = ["acne", "redness", "bags"]
        predicted_class = classes[np.argmax(prediction)]

        # Delete the temporary image file
        os.remove(file_path)

        # Return the prediction as JSON
        return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
