import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import gc
import csv
import os

# Limit GPU memory growth to prevent TensorFlow from allocating all memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Define pre-trained models
models_dict = {
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'MobileNetV2': MobileNetV2,
    'EfficientNetB0': EfficientNetB0
}

# Define image types
image_types = ['Color', 'Grayscale', 'Segmented']



# Define dataset splits and corresponding epochs
splits = {
    '80-20': {'train': 'train[:80%]', 'test': 'train[80%:]', 'epochs': 15},
    '70-30': {'train': 'train[:70%]', 'test': 'train[70%:]', 'epochs': 15},
    '60-40': {'train': 'train[:60%]', 'test': 'train[60%:]', 'epochs': 15}
}

# Number of classes in PlantVillage (38 disease classes)
num_classes = 38

# Function to segment leaves from images
def segment_leaf(image):
    # Convert RGB image to HSV color space
    hsv = tf.image.rgb_to_hsv(image)
    # Define lower and upper bounds for green color (adjustable)
    lower_green = tf.constant([0.25, 0.4, 0.4], dtype=tf.float32)  # Hue, Saturation, Value
    upper_green = tf.constant([0.5, 1.0, 1.0], dtype=tf.float32)
    # Create a mask where pixels are within the green range
    mask = tf.reduce_all((hsv >= lower_green) & (hsv <= upper_green), axis=-1)
    # Expand mask to 3 channels to match image dimensions
    mask = tf.stack([mask, mask, mask], axis=-1)
    # Apply mask to retain only leaf regions
    segmented = image * tf.cast(mask, tf.float32)
    return segmented

# Function to preprocess images
def preprocess_image(image, label, image_type, model_name):
    # Cast image to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Apply segmentation if image_type is 'Segmented'
    if image_type == 'Segmented':
        image = segment_leaf(image)
    # Resize based on model (299x299 for Inception models, 224x224 for others)
    size = 299 if model_name in ['InceptionV3', 'InceptionResNetV2'] else 224
    image = tf.image.resize(image, (size, size))
    # Convert to Grayscale if specified
    if image_type == 'Grayscale':
        image = tf.image.rgb_to_grayscale(image)
        # Convert back to 3 channels to match model expectations
        image = tf.image.grayscale_to_rgb(image)
    return image, label

# Initialize CSV file with headers if it doesn't exist
csv_file = 'out.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Image Type', 'Split', 'Test Accuracy', 'Test Loss'])

# Function to create and train a model
def create_and_train_model(model_name, image_type, split_name, train_split, test_split, epochs):
    # Define paths for the final model
    final_model_path = f'final_model_{model_name}_{image_type}_{split_name}.keras'
    # Load dataset splits from TFDS
    try:
        train_ds = tfds.load('plant_village', split=train_split, as_supervised=True)
        val_ds = tfds.load('plant_village', split=test_split, as_supervised=True)
    except Exception as e:
        print(f"Error loading dataset for {model_name}, {image_type}, {split_name}: {e}")
        return
    # Apply preprocessing
    train_ds = train_ds.map(lambda image, label: preprocess_image(image, label, image_type, model_name))
    val_ds_temp = val_ds.map(lambda image, label: preprocess_image(image, label, image_type, model_name))
    # Batch and prefetch for efficiency (batch size 8)
    train_ds = train_ds.batch(8).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_temp.batch(8).prefetch(tf.data.AUTOTUNE)
    # Check if the model already exists
    if os.path.exists(final_model_path):
        print(f"Model {model_name} with {image_type} on {split_name} already exists. Loading and evaluating...")
        try:
            model = tf.keras.models.load_model(final_model_path)
            # Evaluate the loaded model
            test_loss, test_accuracy = model.evaluate(val_ds)
            print(f"Test Accuracy for {model_name}, {image_type}, {split_name}: {test_accuracy:.4f}")
            print(f"Test Loss for {model_name}, {image_type}, {split_name}: {test_loss:.4f}")
        except Exception as e:
            print(f"Error loading or evaluating model {model_name} with {image_type} on {split_name}: {e}")
            test_accuracy, test_loss = None, None
        # Write results to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, image_type, split_name, test_accuracy, test_loss])
        # Clear memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        return
    # If model doesn't exist, proceed with training
    # Set input shape (always 3 channels)
    size = 299 if model_name in ['InceptionV3', 'InceptionResNetV2'] else 224
    input_shape = (size, size, 3)
    # Load pre-trained model with ImageNet weights
    try:
        base_model = models_dict[model_name](
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Add custom classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    # Build the model
    model = Model(inputs=base_model.input, outputs=output)
    # Compile with Adam optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        f'best_model_{model_name}_{image_type}_{split_name}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    # Train the model
    try:
        model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[checkpoint]
        )
    except Exception as e:
        print(f"Error training model {model_name} with {image_type} on {split_name}: {e}")
        return
    # Evaluate the model on the validation set
    try:
        test_loss, test_accuracy = model.evaluate(val_ds)
        print(f"Test Accuracy for {model_name}, {image_type}, {split_name}: {test_accuracy:.4f}")
        print(f"Test Loss for {model_name}, {image_type}, {split_name}: {test_loss:.4f}")
    except Exception as e:
        print(f"Error evaluating model {model_name} with {image_type} on {split_name}: {e}")
        test_accuracy, test_loss = None, None
    # Write results to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, image_type, split_name, test_accuracy, test_loss])
    # Save the final model
    model.save(final_model_path)
    print(f"Saved models for {model_name}, {image_type}, {split_name}")
    # Clear memory after training
    tf.keras.backend.clear_session()
    del model
    gc.collect()

# Train all combinations
for split_name, split_info in splits.items():
    train_split = split_info['train']
    test_split = split_info['test']
    epochs = split_info['epochs']
    for model_name in models_dict.keys():
        for image_type in image_types:
            print(f"Training {model_name} with {image_type} images on {split_name} split...")
            create_and_train_model(model_name, image_type, split_name, train_split, test_split, epochs)