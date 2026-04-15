from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import io
import threading

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from Emotion import KMeansScratch, EMOTION_LABELS, model_path, run_emotion_analysis
from emotional_paths import get_paths_for_emotion, get_playlist_uri, DEFAULT_PLAYLIST


app = Flask(__name__)
CORS(app)

class NoFaceDetectedError(Exception):
    """Raised when no face is detected in the provided image."""


try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as exc:
    raise RuntimeError(f"Failed to load emotion model from '{model_path}': {exc}") from exc

FEATURE_HISTORY: list[np.ndarray] = []
FEATURE_HISTORY_MAX_LEN = 100
ANALYSIS_THREAD: threading.Thread | None = None
ANALYSIS_LOCK = threading.Lock()

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_base64_image(image_b64: str) -> np.ndarray:
    """
    Accepts a base64-encoded image string (optionally a data URL),
    detects a face region, converts it to a 48x48 grayscale tensor
    of shape (1, 48, 48, 1) normalized to [0, 1].
    """
    if "," in image_b64:
        _, image_b64 = image_b64.split(",", 1)

    image_bytes = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.asarray(pil_img)

    # Grayscale for face detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise NoFaceDetectedError("No face detected.")

    x, y, w, h = faces[0]
    roi = gray[y : y + h, x : x + w]
    roi = cv2.resize(roi, (48, 48))

    # FIX: normalize to [0,1], keep grayscale — shape (1, 48, 48, 1)
    roi = roi.astype("float32") / 255.0
    return np.reshape(roi, (1, 48, 48, 1))


def update_feature_history(vector: np.ndarray) -> None:
    FEATURE_HISTORY.append(vector)
    if len(FEATURE_HISTORY) > FEATURE_HISTORY_MAX_LEN:
        FEATURE_HISTORY.pop(0)


def compute_cluster(prediction: np.ndarray) -> int:
    if len(FEATURE_HISTORY) >= 3:
        data = np.vstack(FEATURE_HISTORY)
        k = min(3, data.shape[0])
        kmeans = KMeansScratch(k=k)
        centroids = kmeans.fit(data)

        distances = []
        for c in centroids:
            diff = prediction - c
            dist = float(np.sum(diff ** 2) ** 0.5)
            distances.append(dist)
        return int(np.argmin(distances))
    else:
        data = np.expand_dims(prediction, axis=0)
        kmeans = KMeansScratch(k=1)
        _ = kmeans.fit(data)
        return 0


def _analysis_runner() -> None:
    try:
        run_emotion_analysis()
    finally:
        with ANALYSIS_LOCK:
            global ANALYSIS_THREAD
            ANALYSIS_THREAD = None


def _analyze_image(payload: dict) -> tuple[str, int, list, str, str | None]:
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

    if emotion == "Fear":
        auto_spotify_uri = get_playlist_uri(emotion, "ground")
        return emotion, int(cluster_index), [], "Take it easy.", auto_spotify_uri

    if emotion == "Happy":
        auto_spotify_uri = get_playlist_uri(emotion, "celebrate")
        return emotion, int(cluster_index), [], "Keep the good vibes.", auto_spotify_uri

    if path_config:
        paths = path_config["paths"]
        prompt = path_config["prompt"]
    else:
        paths = []
        prompt = "Choose how you'd like to feel."

    return emotion, int(cluster_index), paths, prompt, None


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Missing 'image' field in JSON body."}), 400

    try:
        emotion, cluster_index, paths, prompt, spotify_uri = _analyze_image(payload)
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to decode image: {exc}"}), 400

    return jsonify({
        "emotion": emotion,
        "cluster": cluster_index,
        "paths": paths,
        "prompt": prompt,
        "spotify_uri": spotify_uri,
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required."}), 400

    emotion = payload.get("emotion") or ""
    path_id = payload.get("path_id") or ""

    if not emotion or not path_id:
        return jsonify({"error": "Both 'emotion' and 'path_id' are required."}), 400

    spotify_uri = get_playlist_uri(emotion, path_id)
    return jsonify({"spotify_uri": spotify_uri})


@app.route("/", methods=["GET"])
def landing_page():
    return render_template("index.html")


@app.route("/run-analysis", methods=["POST"])
def run_analysis():
    global ANALYSIS_THREAD
    with ANALYSIS_LOCK:
        if ANALYSIS_THREAD is not None and ANALYSIS_THREAD.is_alive():
            return jsonify({
                "status": "already_running",
                "message": "Analysis is already running.",
            }), 202

        ANALYSIS_THREAD = threading.Thread(target=_analysis_runner, daemon=True)
        ANALYSIS_THREAD.start()

    return jsonify({
        "status": "started",
        "message": "Emotion analysis started. Check the webcam window.",
    }), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)