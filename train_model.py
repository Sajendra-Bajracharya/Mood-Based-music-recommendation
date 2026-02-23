import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from prepare_data import train_generator, val_generator

# --- STEP 1: RESUME LOGIC (Targeting your BEST model) ---
# Your screenshot shows 'emotion_model_best.h5' exists in your folder
model_path = 'emotion_model_best.h5'

if os.path.exists(model_path):
    print(f"\n[RESUMING] Found best model '{model_path}'. Re-initializing optimizer...")
    # Load model architecture and weights, but ignore the old optimizer state
    model = load_model(model_path, compile=False)
    
    # Manually re-compile to refresh the optimizer and link the 5 emotion classes
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Lower rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
else:
    print("\n[STARTING FRESH] 'emotion_model_best.h5' not found. Building fresh architecture...")
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output for 5 emotions (Angry, Fear, Happy, Neutral, Sad)
        Dense(5, activation='softmax') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# --- STEP 2: SMART CALLBACKS ---
# Helps the model 'study harder' when improvement slows
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Continuously saves the highest-accuracy version
checkpoint = ModelCheckpoint('emotion_model_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# --- STEP 3: TRAIN ---
print("\n--- Starting Training Process ---")
# initial_epoch=0 starts the count from the beginning
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    initial_epoch=0, 
    callbacks=[lr_reducer, checkpoint],
    verbose=1
)

# --- STEP 4: SAVE FINAL STATE ---
model.save('emotion_model.h5')
print("\n--- Training Complete ---")
if 'accuracy' in history.history:
    print("Final Accuracy:", history.history['accuracy'][-1])