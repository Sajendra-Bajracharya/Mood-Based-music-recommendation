import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from prepare_data import val_generator  # Make sure prepare_data.py is in the same folder

# 1. LOAD THE TRAINED MODEL
model_path = 'emotion_model_best.h5'
print(f"--- Loading Model: {model_path} ---")
model = load_model(model_path)

# 2. THE CRITICAL FIX: SYNC THE GENERATOR
# If shuffle is True, the labels (y_true) won't match the images (y_pred)
val_generator.shuffle = False  
val_generator.reset()          

# 3. GET PREDICTIONS
print("--- Evaluating on Validation Data... ---")
# Predict probabilities
predictions = model.predict(val_generator, steps=len(val_generator), verbose=1)
# Convert probabilities to class indices
y_pred = np.argmax(predictions, axis=1)

# Get the true labels (now in the correct order)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# 4. GENERATE CLASSIFICATION REPORT
print("\n--- Classification Report ---")
# This will now show your TRUE accuracy (likely 50-60%)
print(classification_report(y_true, y_pred, target_names=class_labels))

# 5. CREATE CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)



plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Emotion Detection Confusion Matrix (Corrected)')
plt.ylabel('Actual Emotion')
plt.xlabel('Predicted Emotion')
plt.show()