"""
Flask Routes — YOLOv8 Explorer Web Dashboard (v2)
===================================================
All HTTP endpoints for the upgraded dashboard.

Endpoints:
  GET  /                      → Dashboard HTML
  POST /detect/image          → Upload image → annotate → return + save to history
  GET  /webcam/stream         → MJPEG live webcam stream
  POST /webcam/stop           → Stop stream
  POST /webcam/snapshot       → Capture annotated frame from live stream
  GET  /api/classes           → All 80 COCO class names
  GET  /api/model-info        → Current model metadata
  POST /api/set-model         → Hot-swap the YOLOv8 model (nano/small/medium...)
  GET  /api/history           → Detection history (last 10)
  DELETE /api/history         → Clear history

Engineering note — Why in-memory history (no database)?
  For a learning project, a simple Python list is perfect:
    - Zero setup (no SQLite, no ORM)
    - Automatically clears on server restart (expected behaviour)
    - Fast — O(1) insert, O(n) retrieval
  In production you'd use SQLite (sqlite3 module) or PostgreSQL.
"""

import os
import time
import uuid
import base64
import threading

import cv2
import numpy as np
from flask import (
    Blueprint, render_template, request, jsonify,
    Response, current_app,
)
from werkzeug.utils import secure_filename

from app.detector import get_detector, set_detector_model, COCO_CLASSES
from app.utils import frame_to_jpeg_bytes, resize_for_display

main_bp = Blueprint("main", __name__)

# ─── Constants ────────────────────────────────────────────────────────────────
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
HISTORY_MAX = 10

# ─── Shared State ─────────────────────────────────────────────────────────────
# Detection history: list of dicts, newest first
_detection_history: list = []

# Webcam streaming state
_webcam_active: bool = False
_webcam_cap: cv2.VideoCapture | None = None
_last_webcam_frame: np.ndarray | None = None   # Updated every frame for snapshots
_webcam_lock = threading.Lock()

# Current active model name (tracks what the user switched to)
_current_model_name: str = "yolov8n.pt"


def _allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _make_thumbnail(annotated_bgr: np.ndarray, size: int = 120) -> str:
    """
    Resize an annotated frame to a tiny square thumbnail and return as base64.
    Used for the history panel — we don't want to store full-res images in memory.

    Engineering insight: Always store thumbnails instead of full images for history/logs.
    A 640×640 JPEG is ~80KB; a 120×120 thumbnail is ~4KB — 20× smaller.
    """
    h, w = annotated_bgr.shape[:2]
    scale = size / max(h, w)
    thumb = cv2.resize(annotated_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


# ─── Main Dashboard ───────────────────────────────────────────────────────────

@main_bp.route("/")
def index():
    return render_template("index.html")


# ─── Image Detection ─────────────────────────────────────────────────────────

@main_bp.route("/detect/image", methods=["POST"])
def detect_image():
    """
    POST multipart/form-data: field 'image' (file) + 'confidence' (float)

    Flow:
      1. Validate & save uploaded file
      2. Run YOLOv8 detection
      3. Annotate image with bounding boxes
      4. Save to history (thumbnail only)
      5. Return annotated image (base64) + detection JSON

    Why base64 in JSON instead of a separate image URL?
      Simpler for the client — one fetch, one response.
      No need to manage temporary file URLs or cleanup.
      Downside: ~33% larger payload. Fine for ≤32MB images.
    """
    global _detection_history, _current_model_name

    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    conf = float(request.form.get("confidence", 0.35))

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed_file(file.filename, ALLOWED_IMAGE):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_IMAGE)}"}), 400

    # Save upload temporarily
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    try:
        detector = get_detector(confidence=conf)

        # Run detection
        results = detector.detect_image(upload_path)

        # Get annotated frame via YOLOv8's built-in plotter
        raw = detector.model(upload_path, conf=conf, verbose=False)
        annotated = raw[0].plot()                         # BGR numpy array
        annotated = resize_for_display(annotated, 1024)   # Cap at 1024px wide

        # Encode full-res result for display
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64 = base64.b64encode(buf).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_b64}"

        # Build history entry (thumbnail only to save memory)
        history_entry = {
            "id": uuid.uuid4().hex[:8],
            "timestamp": time.strftime("%H:%M:%S"),
            "date": time.strftime("%b %d"),
            "thumbnail": _make_thumbnail(annotated),
            "full_image": img_data_url,        # Full image for re-viewing
            "count": results["count"],
            "top_class": (
                next(iter(results["class_summary"]), "—")
                if results["class_summary"] else "—"
            ),
            "class_summary": results["class_summary"],
            "inference_ms": results["inference_ms"],
            "model": _current_model_name,
            "detections": results["detections"],
        }

        _detection_history.insert(0, history_entry)
        _detection_history = _detection_history[:HISTORY_MAX]

        return jsonify({
            "success": True,
            "annotated_image": img_data_url,
            "detections": results["detections"],
            "count": results["count"],
            "inference_ms": results["inference_ms"],
            "class_summary": results["class_summary"],
            "model": _current_model_name,
            "history_count": len(_detection_history),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


# ─── Webcam Stream ────────────────────────────────────────────────────────────

def _generate_webcam_frames(confidence: float = 0.35):
    """
    MJPEG frame generator — yields indefinitely while _webcam_active is True.

    How MJPEG streaming works (engineering concept):
      HTTP/1.1 supports persistent connections. Instead of closing after one
      response, the server keeps the connection open and sends a sequence of
      JPEG images separated by MIME boundaries. The browser's <img> tag
      renders each frame as it arrives — effectively a video stream over HTTP,
      no WebSockets or WebRTC needed.

      Boundary format:
        --frame\r\n
        Content-Type: image/jpeg\r\n\r\n
        <jpeg bytes>
        \r\n
    """
    global _webcam_active, _webcam_cap, _last_webcam_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _webcam_cap = cap
    _webcam_active = True
    detector = get_detector(confidence=confidence)

    try:
        while _webcam_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Save raw frame for snapshot endpoint
            with _webcam_lock:
                _last_webcam_frame = frame.copy()

            # Run detection + annotate
            detector.confidence = confidence
            annotated, _ = detector.detect_and_annotate(frame)

            # Encode to JPEG
            jpeg = frame_to_jpeg_bytes(annotated, quality=75)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg + b"\r\n"
            )
    finally:
        cap.release()
        _webcam_active = False
        _webcam_cap = None


@main_bp.route("/webcam/stream")
def webcam_stream():
    conf = float(request.args.get("confidence", 0.35))
    return Response(
        _generate_webcam_frames(confidence=conf),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@main_bp.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    global _webcam_active
    _webcam_active = False
    return jsonify({"success": True})


@main_bp.route("/webcam/snapshot", methods=["POST"])
def webcam_snapshot():
    """
    Capture an annotated snapshot from the current webcam stream.

    The /webcam/stream generator saves each raw frame to _last_webcam_frame.
    This endpoint reads that frame, runs fresh detection, and returns the result.

    Why separate from the stream?
      The stream generator runs in its own thread. We can't interrupt it.
      Instead we share state via a thread-safe variable + lock.
    """
    global _last_webcam_frame

    with _webcam_lock:
        frame = _last_webcam_frame.copy() if _last_webcam_frame is not None else None

    if frame is None:
        return jsonify({"error": "No webcam frame available. Start the webcam first."}), 400

    data = request.get_json() or {}
    conf = float(data.get("confidence", 0.35))

    detector = get_detector(confidence=conf)
    annotated, results = detector.detect_and_annotate(frame)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "success": True,
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "thumbnail": _make_thumbnail(annotated),
        "count": results["count"],
        "class_summary": results["class_summary"],
        "timestamp": time.strftime("%H:%M:%S"),
        "inference_ms": results["inference_ms"],
    })


# ─── Model Switcher ───────────────────────────────────────────────────────────

@main_bp.route("/api/set-model", methods=["POST"])
def api_set_model():
    """
    Hot-swap the YOLOv8 model while the server is running.

    This is a great engineering demonstration:
      - The model is loaded once and cached as a singleton
      - Switching replaces the singleton with a new instance
      - All subsequent requests automatically use the new model
      - No server restart needed

    The client shows a loading spinner during the 1–5 second load time.
    """
    global _current_model_name

    data = request.get_json() or {}
    model_name = data.get("model", "yolov8n.pt")

    if model_name not in ALLOWED_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {list(ALLOWED_MODELS)}"}), 400

    try:
        t_start = time.time()
        set_detector_model(model_name)
        _current_model_name = model_name
        elapsed = round((time.time() - t_start) * 1000)

        return jsonify({
            "success": True,
            "model": model_name,
            "load_ms": elapsed,
            "message": f"Switched to {model_name} in {elapsed}ms",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── History ──────────────────────────────────────────────────────────────────

@main_bp.route("/api/history", methods=["GET"])
def api_history():
    """Return the last N detection history entries."""
    return jsonify({
        "history": _detection_history,
        "count": len(_detection_history),
    })


@main_bp.route("/api/history", methods=["DELETE"])
def api_history_clear():
    """Clear the detection history."""
    global _detection_history
    _detection_history = []
    return jsonify({"success": True, "message": "History cleared"})


# ─── Info APIs ────────────────────────────────────────────────────────────────

@main_bp.route("/api/classes")
def api_classes():
    return jsonify({
        "classes": COCO_CLASSES,
        "count": len(COCO_CLASSES),
        "dataset": "COCO (Common Objects in Context)",
    })


@main_bp.route("/api/model-info")
def api_model_info():
    specs = {
        "yolov8n.pt": {"variant": "nano",       "params": "3.2M",  "size_mb": 6,   "map": 37.3, "speed": 1.4},
        "yolov8s.pt": {"variant": "small",      "params": "11.2M", "size_mb": 22,  "map": 44.9, "speed": 2.4},
        "yolov8m.pt": {"variant": "medium",     "params": "25.9M", "size_mb": 52,  "map": 50.2, "speed": 5.2},
        "yolov8l.pt": {"variant": "large",      "params": "43.7M", "size_mb": 87,  "map": 52.9, "speed": 8.1},
        "yolov8x.pt": {"variant": "extra-large","params": "68.2M", "size_mb": 136, "map": 53.9, "speed": 13.4},
    }
    info = specs.get(_current_model_name, specs["yolov8n.pt"])
    return jsonify({"model": _current_model_name, "num_classes": 80, **info})
