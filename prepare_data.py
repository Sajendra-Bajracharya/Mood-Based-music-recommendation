import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. SETUP PARAMETERS
IMG_SIZE = 48 
BATCH_SIZE = 32
DATA_DIR = "Data/" 

# 2. WORKFLOW: Enhanced Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    shear_range=0.2,        
    zoom_range=0.2,         
    horizontal_flip=True,   
    brightness_range=[0.8, 1.2], # Helps model ignore lighting variations
    fill_mode='nearest',
    validation_split=0.2    
)

val_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

# 3. WORKFLOW: Loading Data
print("--- Loading Training Data (Advanced Augmentation) ---")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training",
    shuffle=True
)

print("\n--- Loading Validation Data (Clean) ---")
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation",
    shuffle=False 
)

print(f"\nLabels assigned: {train_generator.class_indices}")