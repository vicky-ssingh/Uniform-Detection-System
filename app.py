import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time
import threading
import glob
from model import UniformDetectionModel

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model with correct paths
# Update these paths to match your actual dataset location
DATASET_DIR = os.path.join(os.getcwd(), 'DS18')
UNIFORM_DIR = os.path.join(DATASET_DIR, 'Uniform')
NON_UNIFORM_DIR = os.path.join(DATASET_DIR, 'Non_Uniform')

print(f"Dataset directory: {DATASET_DIR}")
print(f"Uniform directory: {UNIFORM_DIR}")
print(f"Non-uniform directory: {NON_UNIFORM_DIR}")

model = UniformDetectionModel(UNIFORM_DIR, NON_UNIFORM_DIR)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def cleanup_images(delay=300):  # Default delay of 5 minutes (300 seconds)
    """Delete uploaded and captured images after a specified delay"""

    def delayed_cleanup():
        time.sleep(delay)
        files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
        count = 0
        for file in files:
            try:
                os.remove(file)
                count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        print(f"Cleanup completed: {count} files deleted")

    # Start cleanup in a separate thread
    thread = threading.Thread(target=delayed_cleanup)
    thread.daemon = True
    thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train_model', methods=['GET'])
def train_model():
    try:
        model.build_model()
        history = model.train_model(epochs=20)
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully!'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error training model: {str(e)}'
        })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        })

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No selected file'
        })

    if file and allowed_file(file.filename):
        # Create a unique filename
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check if model is loaded, if not, try to load it
        if model.model is None:
            if not model.load_trained_model():
                # If model doesn't exist, train it
                model.build_model()
                model.train_model(epochs=20)

        # Make prediction
        result = model.predict(filepath)

        # Schedule cleanup for this image
        cleanup_images()

        return jsonify({
            'status': 'success',
            'result': result,
            'image_path': filepath.replace('\\', '/')
        })

    return jsonify({
        'status': 'error',
        'message': 'File type not allowed'
    })


@app.route('/capture', methods=['POST'])
def capture():
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image data'
        })

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No captured image'
        })

    # Create a unique filename
    filename = secure_filename(f"capture_{int(time.time())}.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Check if model is loaded, if not, try to load it
    if model.model is None:
        if not model.load_trained_model():
            # If model doesn't exist, train it
            model.build_model()
            model.train_model(epochs=20)

    # Make prediction
    result = model.predict(filepath)

    # Schedule cleanup for this image
    cleanup_images()

    return jsonify({
        'status': 'success',
        'result': result,
        'image_path': filepath.replace('\\', '/')
    })


@app.route('/cleanup', methods=['GET'])
def manual_cleanup():
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    count = 0
    for file in files:
        try:
            os.remove(file)
            count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    return jsonify({
        'status': 'success',
        'message': f'Cleanup completed: {count} files deleted'
    })


if __name__ == '__main__':
    # Check if model exists, if not, train it
    if not model.load_trained_model():
        print("Pre-trained model not found. Training a new model...")
        model.build_model()
        model.train_model(epochs=20)
        print("Model training completed!")
    else:
        print("Pre-trained model loaded successfully!")

    app.run(debug=True)
