import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import urllib.request

app = Flask(__name__)

# Function to perform vehicle detection
def detect_vehicle(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Load the pre-trained vehicle detection model (e.g., HAAR Cascade)
    vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the image
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the result image
    result_image_path = 'static/result.jpg'
    cv2.imwrite(result_image_path, img)

    return result_image_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has a name
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded image temporarily
        uploaded_image_path = 'static/uploaded_image.jpg'
        file.save(uploaded_image_path)

        # Perform vehicle detection
        result_image_path = detect_vehicle(uploaded_image_path)

        return render_template('index.html', uploaded_image=uploaded_image_path, result_image=result_image_path)

    return render_template('index.html', uploaded_image=None, result_image=None)

if __name__ == '__main__':
    app.run(debug=True)
