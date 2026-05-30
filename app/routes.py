"""
Flask Routes — Web Dashboard
==============================
All HTTP endpoints for the YOLOv8 Explorer web interface.

Endpoints:
  GET  /                   → Dashboard HTML
  POST /detect/image       → Upload image, run detection, return annotated image
  GET  /webcam/stream      → MJPEG live webcam stream with detections
  GET  /webcam/stop        → Stop the webcam stream
  GET  /api/classes        → List all 80 COCO class names
  GET  /api/model-info     → Return model metadata
"""

import os
import time
import uuid
import base64
import cv2
import numpy as np
from flask import (
    Blueprint, render_template, request, jsonify,
    Response, current_app, send_from_directory,
)
from werkzeug.utils import secure_filename

from app.detector import get_detector, COCO_CLASSES
from app.utils import frame_to_jpeg_bytes, resize_for_display

main_bp = Blueprint("main", __name__)

# Allowed upload extensions
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv"}

# Global webcam state
_webcam_active = False
_webcam_cap = None


def _allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# ─── Main Dashboard ───────────────────────────────────────────────────────────

@main_bp.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


# ─── Image Detection ─────────────────────────────────────────────────────────

@main_bp.route("/detect/image", methods=["POST"])
def detect_image():
    """
    Upload an image → run YOLOv8 → return annotated image + detection data.

    Expects: multipart/form-data with field 'image'
    Returns: JSON with base64-encoded annotated image and detection list
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    conf = float(request.form.get("confidence", 0.35))

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename, ALLOWED_IMAGE):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_IMAGE)}"}), 400

    # Save the uploaded file
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    try:
        detector = get_detector(confidence=conf)
        results = detector.detect_image(upload_path)

        # Load the image and draw detections
        frame = cv2.imread(upload_path)
        if frame is None:
            return jsonify({"error": "Failed to read uploaded image"}), 500

        # Use YOLOv8's built-in annotator for cleaner visuals
        from ultralytics import YOLO
        yolo = detector.model
        raw = yolo(upload_path, conf=conf, verbose=False)
        annotated = raw[0].plot()  # Returns BGR numpy array

        # Resize for web display
        annotated = resize_for_display(annotated, max_width=1024)

        # Encode to base64 for JSON transport
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
            "detections": results["detections"],
            "count": results["count"],
            "inference_ms": results["inference_ms"],
            "class_summary": results["class_summary"],
            "model": results["model"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)


# ─── Webcam Stream ────────────────────────────────────────────────────────────

def _generate_webcam_frames(confidence: float = 0.35):
    """
    Generator that yields MJPEG frames from webcam with YOLOv8 detections.

    This is how MJPEG streaming works:
    - Browser opens a persistent HTTP connection
    - Server keeps sending JPEG frames, separated by MIME boundaries
    - Browser renders each frame as it arrives → live video!
    """
    global _webcam_active, _webcam_cap

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        return

    _webcam_cap = cap
    _webcam_active = True
    detector = get_detector(confidence=confidence)

    try:
        while _webcam_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection on the frame
            annotated, _ = detector.detect_and_annotate(frame)

            # Encode frame as JPEG
            jpeg_bytes = frame_to_jpeg_bytes(annotated, quality=80)

            # MJPEG format: each frame is wrapped in multipart headers
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes +
                b"\r\n"
            )
    finally:
        cap.release()
        _webcam_active = False
        _webcam_cap = None


@main_bp.route("/webcam/stream")
def webcam_stream():
    """Stream live webcam video with YOLOv8 detections (MJPEG)."""
    conf = float(request.args.get("confidence", 0.35))
    return Response(
        _generate_webcam_frames(confidence=conf),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@main_bp.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    """Stop the webcam stream."""
    global _webcam_active
    _webcam_active = False
    return jsonify({"success": True, "message": "Webcam stream stopped"})


# ─── API Endpoints ────────────────────────────────────────────────────────────

@main_bp.route("/api/classes")
def api_classes():
    """Return all 80 COCO class names."""
    return jsonify({
        "classes": COCO_CLASSES,
        "count": len(COCO_CLASSES),
        "dataset": "COCO (Common Objects in Context)",
    })


@main_bp.route("/api/model-info")
def api_model_info():
    """Return information about the loaded model."""
    model_sizes = {
        "yolov8n.pt": {"params": "3.2M",  "size_mb": 6,   "map50_95": 37.3, "speed_ms": "~1.4"},
        "yolov8s.pt": {"params": "11.2M", "size_mb": 22,  "map50_95": 44.9, "speed_ms": "~2.4"},
        "yolov8m.pt": {"params": "25.9M", "size_mb": 52,  "map50_95": 50.2, "speed_ms": "~5.2"},
        "yolov8l.pt": {"params": "43.7M", "size_mb": 87,  "map50_95": 52.9, "speed_ms": "~8.1"},
        "yolov8x.pt": {"params": "68.2M", "size_mb": 136, "map50_95": 53.9, "speed_ms": "~13.4"},
    }

    current_model = "yolov8n.pt"
    info = model_sizes.get(current_model, {})

    return jsonify({
        "model": current_model,
        "variant": "nano",
        "task": "object detection",
        "dataset": "COCO",
        "num_classes": 80,
        **info,
        "framework": "Ultralytics YOLOv8",
        "description": (
            "YOLOv8n is the smallest and fastest YOLOv8 variant. "
            "It uses a C2f backbone and an anchor-free detection head."
        ),
    })
