import tensorflow as tf

def configure_gpu():
    try:
        # Try to get GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if not physical_devices:
            print("\n" + "="*50)
            print("GPU NOT FOUND OR NOT PROPERLY CONFIGURED")
            print("="*50)
            print("\nTo enable GPU support, you need to:")
            print("1. Install NVIDIA GPU drivers")
            print("2. Install CUDA Toolkit (version 11.2 or later)")
            print("3. Install cuDNN (version 8.1 or later)")
            print("\nFor detailed instructions, visit:")
            print("https://www.tensorflow.org/install/gpu")
            print("\nContinuing with CPU training...")
            print("="*50 + "\n")
            tf.config.set_visible_devices([], 'GPU')
        else:
            print(f"\nFound {len(physical_devices)} GPU(s). Using GPU for training.")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"\nError configuring GPU: {str(e)}")
        print("Falling back to CPU training...")
        tf.config.set_visible_devices([], 'GPU')

# Configure GPU/CPU
configure_gpu()

from tensorflow.keras.preprocessing import image