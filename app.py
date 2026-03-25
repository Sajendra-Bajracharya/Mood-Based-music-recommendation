from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from Emotion import KMeansScratch, EMOTION_LABELS, model_path
from emotional_paths import get_paths_for_emotion, get_playlist_uri, DEFAULT_PLAYLIST


app = Flask(__name__)
CORS(app)

class NoFaceDetectedError(Exception):
    """Raised when no face is detected in the provided image."""


# Load the trained emotion model once at startup
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as exc:
    # Fail fast with a clear message if the model cannot be loaded
    raise RuntimeError(f"Failed to load emotion model from '{model_path}': {exc}") from exc

# Keep a small history of feature vectors so that KMeansScratch
# can perform a meaningful clustering across recent requests.
FEATURE_HISTORY: list[np.ndarray] = []
FEATURE_HISTORY_MAX_LEN = 100

# Face detector (same Haar cascade used in the original Emotion.py script)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_base64_image(image_b64: str) -> np.ndarray:
    """
    Accepts a base64-encoded image string (optionally a data URL),
    converts it to a 48x48 grayscale tensor of shape (1, 48, 48, 1)
    normalized to [0, 1].
    """
    # Handle data URLs of the form "data:image/jpeg;base64,AAAA..."
    if "," in image_b64:
        _, image_b64 = image_b64.split(",", 1)

    image_bytes = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.asarray(pil_img)

    # Convert to grayscale for Haar cascade face detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise NoFaceDetectedError("No face detected.")

    # Use the first detected face
    x, y, w, h = faces[0]
    roi = gray[y : y + h, x : x + w]
    roi = cv2.resize(roi, (48, 48))

    arr = roi.astype("float32") / 255.0
    arr = np.reshape(arr, (1, 48, 48, 1))
    return arr


def update_feature_history(vector: np.ndarray) -> None:
    """Append a new feature vector and keep the history bounded."""
    FEATURE_HISTORY.append(vector)
    if len(FEATURE_HISTORY) > FEATURE_HISTORY_MAX_LEN:
        FEATURE_HISTORY.pop(0)


def compute_cluster(prediction: np.ndarray) -> int:
    """
    Use the manual KMeansScratch implementation to compute a cluster index
    based on recent prediction vectors.
    - If there is enough history, run KMeansScratch(k=3) and assign the
      current prediction to the nearest centroid.
    - If history is too short, fall back to a single-cluster assignment (0),
      still invoking KMeansScratch to satisfy the custom constraint.
    """
    if len(FEATURE_HISTORY) >= 3:
        data = np.vstack(FEATURE_HISTORY)
        k = min(3, data.shape[0])
        kmeans = KMeansScratch(k=k)
        centroids = kmeans.fit(data)

        # Assign current prediction to nearest centroid (manual distance)
        distances = []
        for c in centroids:
            diff = prediction - c
            sq = diff ** 2
            dist = float(np.sum(sq) ** 0.5)
            distances.append(dist)
        return int(np.argmin(distances))
    else:
        # Still invoke KMeansScratch on the minimal dataset so that
        # the custom K-Means constraint is respected.
        data = np.expand_dims(prediction, axis=0)
        kmeans = KMeansScratch(k=1)
        _ = kmeans.fit(data)
        return 0


# ---------------------------------------------------------------------------
# Goal-based emotional regulation: analyze (emotion + paths) and recommend (playlist)
# ---------------------------------------------------------------------------


def _analyze_image(payload: dict) -> tuple[str, int, list, str]:
    """
    Run emotion detection on image and return (emotion, cluster, paths, prompt).
    paths and prompt come from emotional_paths; if emotion is unknown, returns empty paths.
    """
    image_tensor = preprocess_base64_image(payload["image"])
    prediction = model.predict(image_tensor, verbose=0)[0]
    update_feature_history(prediction)

    winning_idx = int(np.argmax(prediction))
    try:
        emotion = str(EMOTION_LABELS[winning_idx]).strip()
    except (IndexError, TypeError):
        emotion = "Unknown"

    cluster_index = compute_cluster(prediction)
    path_config = get_paths_for_emotion(emotion)

    if path_config:
        paths = path_config["paths"]
        prompt = path_config["prompt"]
    else:
        paths = []
        prompt = "Choose how you'd like to feel."

    return emotion, int(cluster_index), paths, prompt


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Goal-based step 1: detect emotion and return path options.

    Expected JSON: { "image": "<base64 or data URL>" }
    Response: { "emotion", "cluster", "paths": [{ "id", "label", "description", "music_profile" }], "prompt" }
    """
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Missing 'image' field in JSON body."}), 400

    try:
        emotion, cluster_index, paths, prompt = _analyze_image(payload)
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to decode image: {exc}"}), 400

    return jsonify({
        "emotion": emotion,
        "cluster": cluster_index,
        "paths": paths,
        "prompt": prompt,
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Goal-based step 2: return Spotify playlist for the chosen emotion + path.

    Expected JSON: { "emotion": "Angry", "path_id": "cool_down" }
    Response: { "spotify_uri": "https://open.spotify.com/embed/playlist/..." }
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required."}), 400

    emotion = payload.get("emotion") or ""
    path_id = payload.get("path_id") or ""

    if not emotion or not path_id:
        return jsonify({
            "error": "Both 'emotion' and 'path_id' are required.",
        }), 400

    spotify_uri = get_playlist_uri(emotion, path_id)

    return jsonify({
        "spotify_uri": spotify_uri,
    })


if __name__ == "__main__":
    # Default to port 5000 so that the React dev server can proxy to it easily.
    app.run(host="0.0.0.0", port=5000, debug=True)

