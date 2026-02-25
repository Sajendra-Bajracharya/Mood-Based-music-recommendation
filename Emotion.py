import cv2
import numpy as np
import tensorflow as tf
import os
import time
import webbrowser
from collections import Counter

# --- CONFIGURATION ---
model_path = 'emotion_model_best.h5'
TRACKING_DURATION = 10 
# Ensure these match your training labels exactly!
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# Step 1: Use Real Search URLs to ensure they always open correctly
MOOD_PLAYLISTS = {
    'Happy': 'https://open.spotify.com/search/happy%20vibes',
    'Sad': 'https://open.spotify.com/search/sad%20lofi',
    'Angry': 'https://open.spotify.com/search/heavy%20metal',
    'Fear': 'https://open.spotify.com/search/calm%20piano',
    'Neutral': 'https://open.spotify.com/search/chill%20mix'
}

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    model = tf.keras.models.load_model(model_path, compile=False)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    emotion_history = [] 
    captured_frame = None 
    system_active = True  

    print(f"--- SentiSymphonics: Analyzing for {TRACKING_DURATION}s ---")

    while system_active:
        ret, frame = cap.read()
        if not ret: break

        elapsed_time = time.time() - start_time
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255
            roi = np.reshape(roi, (1, 48, 48, 1))

            prediction = model.predict(roi, verbose=0)
            confidence = np.max(prediction)
            label = EMOTION_LABELS[np.argmax(prediction)]

            if confidence > 0.40: 
                emotion_history.append(label)
                captured_frame = frame.copy() 

            # Countdown Timer on Screen
            remaining = max(0, int(TRACKING_DURATION - elapsed_time))
            cv2.putText(frame, f"Analyzing: {remaining}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Last Detect: {label}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if elapsed_time >= TRACKING_DURATION:
            if emotion_history:
                # Get the most frequent emotion
                counts = Counter(emotion_history)
                final_mood = counts.most_common(1)[0][0]
                
                print(f"\n--- RESULTS ---")
                print(f"Dominant Mood: {final_mood}")

                # Save the image
                img_name = f"result_{final_mood}.jpg"
                cv2.imwrite(img_name, captured_frame if captured_frame is not None else frame)

                # Logic: Fetch and open the playlist
                # .get() helps prevent errors if the label doesn't match a key
                music_url = MOOD_PLAYLISTS.get(final_mood)

                if music_url:
                    print(f"Opening browser for: {final_mood}")
                    webbrowser.open(music_url)
                else:
                    print(f"ERROR: No playlist found for '{final_mood}'. Check spelling in MOOD_PLAYLISTS.")

                system_active = False 
            else:
                print("No face detected during the window. Restarting...")
                start_time = time.time()

        cv2.imshow('SentiSymphonics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show Final Result on the screen for a moment
    if not system_active:
        cv2.putText(frame, f"FINAL MOOD: {final_mood}", (50, 200), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow('SentiSymphonics', frame)
        print("Press any key to close the app...")
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()