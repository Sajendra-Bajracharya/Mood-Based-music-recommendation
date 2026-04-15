import cv2
import numpy as np
import tensorflow as tf
import os
import time
import webbrowser

model_path = 'emotion_model_best.keras'
TRACKING_DURATION = 10
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

MOOD_PLAYLISTS = {
    'Happy':   'https://open.spotify.com/playlist/03PKLp1XC1GdqhzIMWgsBa',
    'Sad':     'https://open.spotify.com/search/sad%20lofi',
    'Angry':   'https://open.spotify.com/search/heavy%20metal',
    'Fear':    'https://open.spotify.com/search/calm%20piano',
    'Neutral': 'https://open.spotify.com/search/chill%20mix'
}


class KMeansScratch:
    def __init__(self, k=3, max_iters=10):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, data):
        n_samples, n_features = data.shape
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = data[random_indices].copy()

        for _ in range(self.max_iters):
            labels = []
            for x in data:
                distances = [float(np.sum((x - c) ** 2) ** 0.5) for c in self.centroids]
                labels.append(np.argmin(distances))

            labels = np.array(labels)
            new_centroids = np.zeros((self.k, n_features))

            for i in range(self.k):
                points = data[labels == i]
                new_centroids[i] = np.mean(points, axis=0) if len(points) > 0 else self.centroids[i]

            if np.sum((self.centroids - new_centroids) ** 2) < 1e-6:
                break

            self.centroids = new_centroids

        return self.centroids


def run_emotion_analysis():
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return {"status": "error", "message": f"{model_path} not found", "final_mood": "Unknown"}

    model = tf.keras.models.load_model(model_path, compile=False)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    feature_vectors = []
    system_active = True
    final_mood = "Unknown"

    print(f"--- Analyzing for {TRACKING_DURATION}s ---")

    try:
        while system_active:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # FIX: stay grayscale — no RGB conversion, no resnet preprocessing
                roi = cv2.resize(gray_frame[y:y + h, x:x + w], (48, 48))
                roi = roi.astype("float32") / 255.0
                roi = np.reshape(roi, (1, 48, 48, 1))   # (1, 48, 48, 1) ✅

                prediction = model.predict(roi, verbose=0)[0]
                feature_vectors.append(prediction)

                remaining = max(0, int(TRACKING_DURATION - elapsed_time))
                cv2.putText(frame, f"Capturing Mood: {remaining}s",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if elapsed_time >= TRACKING_DURATION:
                if len(feature_vectors) > 5:
                    data = np.array(feature_vectors)
                    km = KMeansScratch(k=3)
                    centroids = km.fit(data)

                    best_centroid = centroids[np.argmax(np.max(centroids, axis=1))]
                    winning_idx = np.argmax(best_centroid)
                    final_mood = str(EMOTION_LABELS[winning_idx]).strip()

                    print(f"\n--- ANALYSIS COMPLETE ---")
                    print(f"Detected Mood: '{final_mood}'")

                    music_url = MOOD_PLAYLISTS.get(final_mood)
                    if music_url:
                        print(f"Opening playlist for {final_mood}...")
                        webbrowser.open(music_url)
                    else:
                        print(f"Mood '{final_mood}' not found in MOOD_PLAYLISTS.")

                    system_active = False
                else:
                    print("No faces detected. Restarting timer...")
                    start_time = time.time()

            cv2.imshow('Melo - Monitoring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not system_active:
            result_window = np.zeros((300, 600, 3), dtype="uint8")
            cv2.putText(result_window, f"RESULT: {final_mood}",
                        (50, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
            cv2.putText(result_window, "Opening Spotify...",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.imshow('Final Recommendation', result_window)
            cv2.waitKey(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {"status": "success", "final_mood": final_mood}


def main():
    run_emotion_analysis()


if __name__ == "__main__":
    main()