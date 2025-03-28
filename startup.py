import os
import sys

def prepare_environment():
    """Setup environment variables and check dependencies before app starts"""
    # Check for required directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    # Set up environment variables to handle GPU/CPU compatibility
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    # Verify NumPy version is compatible
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy not found, please install required dependencies")
        sys.exit(1)
    
    # If we get here, environment is ready to import TensorFlow
    print("Environment prepared successfully")

if __name__ == "__main__":
    prepare_environment()
    from app import app
    
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
