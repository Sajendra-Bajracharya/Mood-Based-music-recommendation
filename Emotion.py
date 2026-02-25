import cv2
import numpy as np
import tensorflow as tf
import os
import time
import webbrowser

# --- CONFIGURATION ---
model_path = 'emotion_model_best.h5'
TRACKING_DURATION = 10 
# Ensure these match your training labels EXACTLY
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# DIRECT MAPPING (Cleaned and Verified)
MOOD_PLAYLISTS = {
    'Happy': 'https://open.spotify.com/search/happy%20vibes',
    'Sad': 'https://open.spotify.com/search/sad%20lofi',
    'Angry': 'https://open.spotify.com/search/heavy%20metal',
    'Fear': 'https://open.spotify.com/search/calm%20piano',
    'Neutral': 'https://open.spotify.com/search/chill%20mix'
}

class KMeansScratch:
    def __init__(self, k=3, max_iters=10):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, data):
        # Step 1: Manual Initialization
        n_samples, n_features = data.shape
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = data[random_indices].copy()

        for iteration in range(self.max_iters):
            # Step 2: Assignment Logic using Raw Euclidean Formula
            labels = []
            for x in data:
                # Manual Euclidean Distance calculation
                distances = []
                for c in self.centroids:
                    # Formula: sqrt(sum((x - c)^2))
                    squared_diff = (x - c) ** 2
                    sum_squared_diff = np.sum(squared_diff)
                    dist = sum_squared_diff ** 0.5 
                    distances.append(dist)
                
                # Assign to the index of the minimum distance
                labels.append(np.argmin(distances))
            
            labels = np.array(labels)
            new_centroids = np.zeros((self.k, n_features))

            # Step 3: Update Logic using Raw Mean Formula
            for i in range(self.k):
                # Get all points assigned to this cluster
                points_in_cluster = data[labels == i]
                
                if len(points_in_cluster) > 0:
                    # Formula: Sum of points / Count of points
                    sum_of_points = np.sum(points_in_cluster, axis=0)
                    new_centroids[i] = sum_of_points / len(points_in_cluster)
                else:
                    # If cluster is empty, keep the old centroid
                    new_centroids[i] = self.centroids[i]

            # Step 4: Convergence Check (Manual comparison)
            diff = np.sum((self.centroids - new_centroids)**2)
            if diff < 1e-6: # Practically zero change
                break
                
            self.centroids = new_centroids
            
        return self.centroids

# --- MAIN SYSTEM INITIALIZATION ---
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    model = tf.keras.models.load_model(model_path, compile=False)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    feature_vectors = [] 
    captured_frame = None 
    system_active = True  
    final_mood = "Unknown"

    print(f"--- SentiSymphonics: Analyzing for {TRACKING_DURATION}s ---")

    while system_active:
        ret, frame = cap.read()
        if not ret: break

        elapsed_time = time.time() - start_time
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi = cv2.resize(gray_frame[y:y+h, x:x+w], (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            # Store the raw probabilities for K-Means Analysis
            prediction = model.predict(roi, verbose=0)[0]
            feature_vectors.append(prediction)
            captured_frame = frame.copy()

            # On-screen feedback
            remaining = max(0, int(TRACKING_DURATION - elapsed_time))
            cv2.putText(frame, f"Capturing Mood: {remaining}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if elapsed_time >= TRACKING_DURATION:
            if len(feature_vectors) > 5:
                # 1. K-Means Denoising
                data = np.array(feature_vectors)
                km = KMeansScratch(k=3)
                centroids = km.fit(data)

                # 2. Extract Winner
                best_centroid = centroids[np.argmax(np.max(centroids, axis=1))]
                winning_idx = np.argmax(best_centroid)
                
                # Strip spaces and normalize text to match Dictionary Keys
                final_mood = str(EMOTION_LABELS[winning_idx]).strip()

                print(f"\n--- ANALYSIS COMPLETE ---")
                print(f"Detected Mood: '{final_mood}'")

                # 3. Secure URL Retrieval
                music_url = MOOD_PLAYLISTS.get(final_mood)

                if music_url:
                    print(f"SUCCESS: Opening playlist for {final_mood}...")
                    webbrowser.open(music_url)
                else:
                    print(f"CRITICAL ERROR: Mood '{final_mood}' not found in MOOD_PLAYLISTS dictionary.")
                    print(f"Check your dictionary keys spelling!")

                system_active = False 
            else:
                print("⚠️ No faces detected. Restarting timer...")
                start_time = time.time()

        cv2.imshow('SentiSymphonics - Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Final Result Overlay Window
    if not system_active:
        result_window = np.zeros((300, 600, 3), dtype="uint8")
        cv2.putText(result_window, f"RESULT: {final_mood}", (50, 130), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
        cv2.putText(result_window, "Opening Spotify...", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow('Final Recommendation', result_window)
        print("Done. Press any key on the Result window to exit.")
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()