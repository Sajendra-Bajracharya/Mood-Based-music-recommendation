import cv2
import numpy as np
import tensorflow as tf
import os
import time
from collections import Counter

# --- CONFIGURATION ---
model_path = 'emotion_model_best.h5'
TRACKING_DURATION = 5  # 5 seconds for testing
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    # Load the brain we trained earlier
    model = tf.keras.models.load_model(model_path, compile=False)
    # Load the "Face Finder" tool
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Turn on the Webcam
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    emotion_history = [] 

    print(f"--- Tracking the NEAREST person for {TRACKING_DURATION} seconds... ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Make the image gray (easier for the robot to read)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ALL faces in the room
        faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        elapsed_time = time.time() - start_time

        # --- NEW LOGIC: ONLY PICK THE NEAREST (LARGEST) FACE ---
        if len(faces) > 0:
            # We sort faces by Area (Width * Height). 
            # The person closest to the camera will have the biggest "Area".
            # We use reverse=True so the biggest face is at index [0]
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            
            # Grab ONLY the first face (the biggest/nearest one)
            (x, y, w, h) = faces[0]

            # Draw the box and prepare the "Face Image" for the AI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Normalize the pixels (Scale from 0-255 down to 0-1)
            roi = roi_gray.astype('float') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))


            # 1. Ask the AI: "What emotion is this?"
            prediction = model.predict(roi, verbose=0)
            confidence = np.max(prediction)
            label = EMOTION_LABELS[np.argmax(prediction)]

            # 2. Only record the emotion if the AI is sure (above 40% confidence)
            if confidence > 0.40: 
                emotion_history.append(label)

            # Show the label on the screen
            cv2.putText(frame, f"Nearest: {label} ({int(confidence*100)}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if elapsed_time >= TRACKING_DURATION:
            if emotion_history:
                counts = Counter(emotion_history)
                final_mood = counts.most_common(1)[0][0]
                
                print(f"\n--- ANALYSIS COMPLETE ---")
                print(f"Dominant Mood: {final_mood}")
                print(f"Accuracy: {(counts[final_mood]/len(emotion_history))*100:.1f}%")
                
                # FUTURE STEP: play_music(final_mood) 
            
            # Reset the timer and the memory for the next 5 seconds
            start_time = time.time()
            emotion_history = []

        # Show the camera window
        cv2.imshow('Nearest Person Mood AI', frame)
        
        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()