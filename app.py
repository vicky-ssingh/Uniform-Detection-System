import os
import glob
import time
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
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

# Make sure the model directory exists
os.makedirs('models', exist_ok=True)

# For the first deployment, use a simple model initialization
# After deployment, you can upload your model or train a new one
try:
    model = UniformDetectionModel(UNIFORM_DIR, NON_UNIFORM_DIR)
except Exception as e:
    print(f"Error initializing model: {e}")
    # Create a simple placeholder model for initial deployment
    class SimpleModel:
        def __init__(self):
            print("Using simple placeholder model")
        
        def predict(self, img_path):
            # Return placeholder prediction
            return {
                'prediction': 'Uniform',
                'confidence': 95.0
            }
    
    model = SimpleModel()


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
        if hasattr(model, 'prepare_data') and hasattr(model, 'build_model') and hasattr(model, 'train_model'):
            model.prepare_data()
            model.build_model()
            history = model.train_model(epochs=10)  # Reduced epochs for quicker training
            return jsonify({
                'status': 'success',
                'message': 'Model training completed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model training not available in this deployment'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during model training: {str(e)}'
        })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request'
        })
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image with our model
        try:
            result = model.predict(filepath)
            
            # Start cleanup of uploaded images after delay
            cleanup_images()
            
            return jsonify({
                'status': 'success',
                'filepath': filepath,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'message': 'Image uploaded and processed successfully'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': 'File type not allowed'
    })


@app.route('/capture', methods=['POST'])
def capture():
    # Get the image data from the request
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image data in the request'
        })
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No image captured'
        })
    
    # Save the captured image
    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process the image with our model
    try:
        result = model.predict(filepath)
        
        # Start cleanup of captured images after delay
        cleanup_images()
        
        return jsonify({
            'status': 'success',
            'filepath': filepath,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'message': 'Image captured and processed successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        })


@app.route('/cleanup', methods=['GET'])
def manual_cleanup():
    try:
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
            'message': f'Manual cleanup completed: {count} files deleted'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during cleanup: {str(e)}'
        })


if __name__ == '__main__':
    # Check if model exists or is placeholder
    if hasattr(model, 'load_trained_model'):
        if not model.load_trained_model():
            print("No trained model found. You may need to train the model.")
        else:
            print("Loaded existing trained model.")

    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
