import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from prepare_data import train_generator, val_generator

# --- STEP 1: DEFINE ARCHITECTURE ---
# We use a deeper "Double Conv" block structure to increase learning capacity
model = Sequential([
    # Block 1: Initial features (edges, textures)
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2), # shrinks the images. 
    Dropout(0.25),

    # Block 2: Complex facial shapes (eyes, corners of mouth)
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 3: Higher-level expressions
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Dense Classifier
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # 5 Output classes: Angry, Fear, Happy, Neutral, Sad
    Dense(5, activation='softmax') 
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # how often it is correct
)

# --- STEP 2: SMART CALLBACKS ---
# 1. Reduce Learning Rate when learning plateaus
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1) # reduce the learning speed

# 2. Save only the best version based on Validation Accuracy
checkpoint = ModelCheckpoint('emotion_model_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# 3. Early Stopping: Stop if the model hasn't improved in 10 epochs to save time/prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- STEP 3: TRAIN ---
print("\n--- Starting Fresh Training Process ---")
# Increase epochs to 50; EarlyStopping will cut it short if it finishes early
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20, 
    callbacks=[lr_reducer, checkpoint, early_stop],
    verbose=1
)

# --- STEP 4: SAVE ---
model.save('emotion_model_final.h5')
print("\n--- Training Complete ---")