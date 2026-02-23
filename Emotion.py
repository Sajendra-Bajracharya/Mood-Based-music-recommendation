import cv2
import numpy as np
import tensorflow as tf
import os

# 1. Load your best trained model
model_path = 'emotion_model_best.h5'

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    # Load model and ignore the 'metrics' warning since we are only doing inference
    model = tf.keras.models.load_model(model_path, compile=False)
    EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

    # 2. Load the Face Detector
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0)

    print("--- Press 'q' to quit ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for the face detector and the model
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # DRAW: The bounding box on the original color frame for display
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # CROP: Get the face area from the grayscale frame
            roi_gray = gray_frame[y:y+h, x:x+w]
            
            # RESIZE: Shrink the face to exactly 48x48 as required by your model
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # PRE-PROCESS: Normalize and reshape to (1, 48, 48, 1)
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # PREDICT:
            prediction = model.predict(roi, verbose=0)
            label = EMOTION_LABELS[np.argmax(prediction)]
            confidence = np.max(prediction) 