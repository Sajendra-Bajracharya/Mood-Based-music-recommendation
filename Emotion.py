import cv2
import numpy as np
import tensorflow as tf
import os
import time
from collections import Counter # Added for Majority Vote

# --- CONFIGURATION ---
model_path = 'emotion_model_best.h5'
TRACKING_DURATION = 10 # Shortened to 10s for faster testing
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    model = tf.keras.models.load_model(model_path, compile=False)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    emotion_history = [] # This stores every "vote" from the AI

    print(f"--- Tracking for {TRACKING_DURATION} seconds... ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        elapsed_time = time.time() - start_time

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            # 1. Predict
            prediction = model.predict(roi, verbose=0)
            confidence = np.max(prediction)
            label = EMOTION_LABELS[np.argmax(prediction)]

            # 2. LOGIC FIX: Only count "Strong" predictions (Confidence Threshold)
            if confidence > 0.40: 
                emotion_history.append(label)

            cv2.putText(frame, f"Detecting: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. LOGIC FIX: Majority Vote after time is up
        if elapsed_time >= TRACKING_DURATION:
            if emotion_history:
                # Find the most frequent emotion in the list
                counts = Counter(emotion_history)
                final_mood = counts.most_common(1)[0][0]
                
                print(f"\n--- 10s ANALYSIS COMPLETE ---")
                print(f"Dominant Mood: {final_mood}")
                print(f"Confidence in this choice: {(counts[final_mood]/len(emotion_history))*100:.1f}% of the time")
                
                # Here is where we will call the music player!
                # play_music(final_mood) 
            
            # Reset
            start_time = time.time()
            emotion_history = []

        cv2.imshow('Mood AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()