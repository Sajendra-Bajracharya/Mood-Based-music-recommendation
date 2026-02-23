import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. SETUP PARAMETERS
IMG_SIZE = 48 
BATCH_SIZE = 32
DATA_DIR = "Data/" # Ensure this folder now contains Angry, Happy, Neutral, Sad folders

# 2. WORKFLOW: Normalization 
# Rescaling divides pixel values to be between 0 and 1 [cite: 55]
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

# 3. WORKFLOW: Multi-Class Loading
# We change class_mode to "sparse" to handle more than 2 emotions [cite: 431]
print("--- Loading Training Data (4 Emotions) ---")
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",           # Point 2: Convert to Grayscale [cite: 50]
    batch_size=BATCH_SIZE,
    class_mode="sparse",              # Changed from "binary" to handle 4 classes
    subset="training"                 
)

print("\n--- Loading Validation Data (4 Emotions) ---")
val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",              # Changed from "binary"
    subset="validation"               
)

# Verify the labels assigned (e.g., {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3})
print(f"\nLabels assigned: {train_generator.class_indices}")