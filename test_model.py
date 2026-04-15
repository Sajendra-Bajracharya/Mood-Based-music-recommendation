import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from prepare_data import val_generator

# 1. LOAD MODEL — compile=False skips needing the custom focal_loss function
print("--- Loading Model ---")
model = load_model('emotion_model_best.keras', compile=False)

# Recompile with standard loss just for evaluation (predictions aren't affected by loss)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 2. SYNC THE GENERATOR — must reset before predicting
val_generator.shuffle = False
val_generator.reset()

# 3. SANITY CHECK — grayscale now, not rgb
print(f"Validation color mode: {getattr(val_generator, 'color_mode', 'unknown')}")
if getattr(val_generator, "color_mode", None) != "grayscale":
    raise ValueError("val_generator must use color_mode='grayscale'. Check prepare_data.py.")

# 4. EVALUATE
print("\n--- Running Evaluation ---")
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"\nValidation Loss    : {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 5. GET PREDICTIONS
print("\n--- Generating Predictions ---")
predictions = model.predict(val_generator, steps=len(val_generator), verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Trim y_true in case of batch size mismatch
y_true = y_true[:len(y_pred)]

# 6. CLASSIFICATION REPORT
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 7. CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title('Emotion Detection Confusion Matrix')
plt.ylabel('Actual Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)  # Saves a copy automatically
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png")