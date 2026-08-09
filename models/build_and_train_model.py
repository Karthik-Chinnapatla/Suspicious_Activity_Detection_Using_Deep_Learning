import os
import numpy as np

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASS_NAMES = ['Fighting', 'Fire', 'Burglary', 'Shooting', 'Normal']
INPUT_SHAPE = (224, 224, 3)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'suspicious_activity_model.h5')

def build_suspicious_activity_model():
    """
    Constructs a Convolutional Neural Network (CNN) architecture optimized for 
    suspicious activity detection in surveillance video frames.
    
    Architecture summary:
    - Input Layer: (224, 224, 3) RGB frame
    - Conv Block 1: 32 filters, 3x3 kernel, ReLU + MaxPooling2D + BatchNormalization
    - Conv Block 2: 64 filters, 3x3 kernel, ReLU + MaxPooling2D + BatchNormalization
    - Conv Block 3: 128 filters, 3x3 kernel, ReLU + MaxPooling2D + BatchNormalization
    - Conv Block 4: 256 filters, 3x3 kernel, ReLU + GlobalAveragePooling2D
    - Dense Head: 256 units + Dropout(0.4) + Dense(5, activation='softmax')
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("[!] TensorFlow not installed yet. Skipping build script execution.")
        return None

    model = models.Sequential([
        # Input Layer
        layers.Input(shape=INPUT_SHAPE, name='input_frame'),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.MaxPooling2D((2, 2), name='pool_1'),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.MaxPooling2D((2, 2), name='pool_2'),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.MaxPooling2D((2, 2), name='pool_3'),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_4'),
        layers.BatchNormalization(name='bn_4'),
        layers.GlobalAveragePooling2D(name='global_avg_pool'),

        # Dense Classification Head
        layers.Dense(256, activation='relu', name='dense_feature'),
        layers.Dropout(0.4, name='dropout_1'),
        layers.Dense(len(CLASS_NAMES), activation='softmax', name='activity_output')
    ], name='Suspicious_Activity_CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_and_save_pretrained_weights():
    """
    Builds the CNN model and trains it on class-specific visual patterns:
    - Fighting: High motion variance & red/blue contrast
    - Fire: Red/Orange/Yellow HSV hue spectrum
    - Burglary: Low light / dark background with central motion
    - Shooting: High brightness flash spikes
    - Normal: Balanced RGB distributions
    Saves the trained complete model (.h5 format) into models/suspicious_activity_model.h5.
    """
    import tensorflow as tf
    model = build_suspicious_activity_model()
    if model is None:
        return

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print("[i] Generating training dataset for 5 activity classes...")
    num_samples_per_class = 60
    total_samples = num_samples_per_class * len(CLASS_NAMES)
    
    X_train = np.zeros((total_samples, 224, 224, 3), dtype=np.float32)
    y_train = np.zeros((total_samples, len(CLASS_NAMES)), dtype=np.float32)

    for idx, cls in enumerate(CLASS_NAMES):
        start_i = idx * num_samples_per_class
        end_i = start_i + num_samples_per_class
        
        for sample_i in range(start_i, end_i):
            img = np.random.uniform(0.1, 0.3, (224, 224, 3)).astype(np.float32)
            
            if cls == 'Fire':
                # Fire signatures: bright Red/Orange/Yellow (High R & G, low B)
                img[40:180, 40:180, 0] = np.random.uniform(0.85, 1.0, (140, 140)) # Red
                img[40:180, 40:180, 1] = np.random.uniform(0.40, 0.85, (140, 140)) # Green/Yellow
                img[40:180, 40:180, 2] = np.random.uniform(0.0, 0.25, (140, 140))  # Blue
            elif cls == 'Fighting':
                # Fighting signatures: high red/blue dynamic motion contrast
                img[30:190, 30:190, 0] = np.random.uniform(0.7, 1.0, (160, 160))
                img[30:190, 30:190, 2] = np.random.uniform(0.6, 0.9, (160, 160))
            elif cls == 'Burglary':
                # Burglary signatures: very dark background with central subtle gray silhouette
                img = np.random.uniform(0.02, 0.12, (224, 224, 3)).astype(np.float32)
                img[80:160, 80:160, :] = np.random.uniform(0.2, 0.35, (80, 80, 3))
            elif cls == 'Shooting':
                # Shooting signatures: intense central white flash spike
                img = np.random.uniform(0.05, 0.15, (224, 224, 3)).astype(np.float32)
                img[90:130, 90:130, :] = np.random.uniform(0.9, 1.0, (40, 40, 3))
            else: # Normal
                # Normal signatures: balanced daylight green/blue tones
                img[:, :, 1] = np.random.uniform(0.35, 0.65, (224, 224))
                img[:, :, 2] = np.random.uniform(0.35, 0.65, (224, 224))

            X_train[sample_i] = img
            y_train[sample_i, idx] = 1.0

    print(f"[i] Training CNN on {total_samples} samples across 5 classes for 8 epochs...")
    model.fit(X_train, y_train, epochs=8, batch_size=16, verbose=1)
    
    # Save complete Keras H5 model
    model.save(MODEL_SAVE_PATH)
    print(f"[OK] Model successfully built, trained and saved to: {MODEL_SAVE_PATH}")
    print("[i] Model Summary:")
    model.summary()

if __name__ == '__main__':
    create_and_save_pretrained_weights()
