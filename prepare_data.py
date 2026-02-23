import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. SETUP PARAMETERS
IMG_SIZE = 48 
BATCH_SIZE = 32
DATA_DIR = "Data/" 

# 2. WORKFLOW: Data Generators
# Augmentation for Training: Forces the model to learn general features
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,      # Randomly rotate
    width_shift_range=0.1,  # Shift horizontally
    height_shift_range=0.1, # Shift vertically
    shear_range=0.1,        # Slight distortion
    zoom_range=0.1,         # Zoom in/out
    horizontal_flip=True,   # Flip the face left/right
    validation_split=0.2    
)

# No Augmentation for Validation: Keep it pure for testing
val_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

# 3. WORKFLOW: Loading Data
print("--- Loading Training Data (Augmented) ---")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training",
    shuffle=True           # Keep shuffled for better learning
)

print("\n--- Loading Validation Data (Clean) ---")
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation",
    shuffle=False          # IMPORTANT: Do not shuffle for test_model.py accuracy
)

# Verify the labels assigned
print(f"\nLabels assigned: {train_generator.class_indices}")