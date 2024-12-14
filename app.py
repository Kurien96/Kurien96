import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model('image_classification_model.h5')

# Create Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading images
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    img = Image.open(file.stream).resize((32, 32))  # Resize for CIFAR-10
    img_array = keras_image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return jsonify({'class_index': str(class_index), 'confidence': str(predictions[0][class_index])})

# Route for webcam feed
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Route to capture image from webcam
@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return 'Could not read frame from webcam'
    
    img = cv2.resize(frame, (32, 32))
    img_array = img / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return jsonify({'class_index': str(class_index), 'confidence': str(predictions[0][class_index])})

if __name__ == '__main__':
    app.run(debug=True)
