import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


class UniformDetectionModel:
    def __init__(self, uniform_dir, non_uniform_dir, model_save_path='models/uniform_detection_model.h5'):
        self.uniform_dir = uniform_dir
        self.non_uniform_dir = non_uniform_dir
        self.model_save_path = model_save_path
        self.model = None
        self.img_height = 224
        self.img_width = 224
        self.class_indices = None

    def prepare_data(self):
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        # Create training and validation datasets
        base_dir = os.path.dirname(self.uniform_dir)
        print(f"Looking for dataset in: {base_dir}")

        train_generator = train_datagen.flow_from_directory(
            base_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            base_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

        # Store class indices for later use
        self.class_indices = train_generator.class_indices
        print(f"Class indices: {self.class_indices}")

        return train_generator, validation_generator

    def build_model(self):
        # Use MobileNetV2 as base model (efficient for deployment)
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze the base model layers
        base_model.trainable = False

        # Create the model
        model = Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train_model(self, epochs=20):
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        # Prepare data
        train_generator, validation_generator = self.prepare_data()

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            )
        ]

        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Save the plot
        os.makedirs('static/plots', exist_ok=True)
        plt.savefig('static/plots/training_history.png')
        plt.close()

    def load_trained_model(self):
        if os.path.exists(self.model_save_path):
            self.model = load_model(self.model_save_path)
            return True
        return False

    def predict(self, img_path):
        if self.model is None:
            self.load_trained_model()

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = self.model.predict(img_array)
        score = float(prediction[0][0])

        # FIXED: Corrected classification logic
        # If class_indices is {'Non_Uniform': 0, 'Uniform': 1}, then:
        # score close to 1 means Uniform, score close to 0 means Non_Uniform
        uniform_class_value = 1
        if self.class_indices and 'Uniform' in self.class_indices:
            uniform_class_value = self.class_indices['Uniform']

        is_uniform = False
        confidence = 0

        if uniform_class_value == 1:
            # If Uniform is mapped to 1
            is_uniform = score > 0.5
            confidence = score * 100 if is_uniform else (1 - score) * 100
        else:
            # If Uniform is mapped to 0
            is_uniform = score < 0.5
            confidence = (1 - score) * 100 if is_uniform else score * 100

        result = {
            'is_uniform': is_uniform,
            'confidence': confidence,
            'score': score
        }

        return result
