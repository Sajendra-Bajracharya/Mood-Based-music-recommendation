import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
from prepare_data import train_generator, val_generator

NUM_CLASSES = len(train_generator.class_indices) # How many classes you have

# --- SQUEEZE-AND-EXCITATION BLOCK ---
# Learns WHICH feature maps matter most — big accuracy boost for faces
def se_block(x, ratio=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x) # takes the avrage of each feature map
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])   # Reweight channels

# --- RESIDUAL BLOCK WITH SE ATTENTION ---
def residual_se_block(x, filters, dropout_rate=0.3):
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x)   # Apply channel attention

    # Projection shortcut if channel dimensions differ
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])   # Skip connection
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

# --- BUILD MODEL ---
def build_model():
    inputs = Input(shape=(48, 48, 1))

    # Stem: initial feature extraction
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1 — 48x48
    x = residual_se_block(x, 64, dropout_rate=0.2)
    x = layers.MaxPooling2D(2, 2)(x)   # → 24x24

    # Block 2 — 24x24
    x = residual_se_block(x, 128, dropout_rate=0.3)
    x = layers.MaxPooling2D(2, 2)(x)   # → 12x12

    # Block 3 — 12x12
    x = residual_se_block(x, 256, dropout_rate=0.3)
    x = layers.MaxPooling2D(2, 2)(x)   # → 6x6

    # Block 4 — 6x6
    x = residual_se_block(x, 512, dropout_rate=0.4)

    # Head: GAP is better than Flatten for small spatial maps
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_model()
model.summary()

# --- FOCAL LOSS ---
# Down-weights easy "Happy" predictions automatically
# More powerful than class_weights alone for imbalanced face data
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        p_t = tf.gather(y_pred, tf.cast(y_true, tf.int32), batch_dims=1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        ce = -tf.math.log(p_t)
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn

# --- COSINE DECAY with WARM RESTARTS ---
# Escapes local minima better than ReduceLROnPlateau
steps_per_epoch = len(train_generator)
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=10 * steps_per_epoch,   # Restart every 10 epochs
    t_mul=2.0,     # Double restart interval each time
    m_mul=0.9      # Slightly lower peak LR after each restart
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy']
)

# --- CLASS WEIGHTS (keep alongside focal loss for double protection) ---
labels = train_generator.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(labels), y=labels
)
class_weights = dict(enumerate(weights))
print(f"Class weights: {class_weights}")

# --- CALLBACKS ---
checkpoint = ModelCheckpoint(
    'emotion_model_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# --- TRAIN ---
print("\n--- Training Custom CNN with Residual + SE Blocks ---")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,                 # More epochs — cosine LR manages the schedule
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

model.save('emotion_model_final.keras')