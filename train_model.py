import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
from prepare_data import train_generator, val_generator

# --- STEP 1: DEFINE ARCHITECTURE ---
model = Sequential([
    # Block 1: Increased Dropout to prevent "Happy" dominance
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3), 

    # Block 2
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Block 3
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- STEP 2: CALCULATE CLASS WEIGHTS ---
# This balances the learning so "Sad" and "Fear" are prioritized
labels = train_generator.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(weights))
print(f"Using Class Weights: {class_weights}")

# --- STEP 3: REFINED CALLBACKS ---
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
checkpoint = ModelCheckpoint('emotion_model_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# --- STEP 4: TRAIN ---
print("\n--- Starting Balanced Training Process ---")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30, 
    class_weight=class_weights, # Apply the weights here
    callbacks=[lr_reducer, checkpoint, early_stop],
    verbose=1
)

model.save('emotion_model_final.h5')