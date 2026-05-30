"""
YOLOv8 Detection Engine
========================
Core wrapper around Ultralytics YOLOv8 for all detection modes.

What this does:
  - Loads a YOLOv8 model (auto-downloads weights if not found)
  - Exposes clean methods for image, video frame, and batch detection
  - Returns structured Python dicts, not raw YOLO result objects

YOLOv8 Model Sizes (nano → xlarge):
  yolov8n  →  ~6 MB,   fastest,  ~37.3 mAP
  yolov8s  →  ~22 MB,  fast,     ~44.9 mAP
  yolov8m  →  ~52 MB,  balanced, ~50.2 mAP
  yolov8l  →  ~87 MB,  accurate, ~52.9 mAP
  yolov8x  →  ~136 MB, best,     ~53.9 mAP

We default to 'yolov8n' — perfect for learning, runs on CPU.
"""

import os
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─── COCO class names (80 object categories) ──────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class YOLODetector:
    """
    A clean, reusable wrapper around YOLOv8.

    How YOLOv8 works (simplified):
    ┌────────────────────────────────────────────────────────┐
    │  Image → Backbone (feature extraction)                 │
    │       → Neck (multi-scale feature fusion, FPN+PAN)     │
    │       → Head (predict bounding boxes + class scores)   │
    │       → NMS (remove duplicate boxes)                   │
    │       → Final detections ✅                            │
    └────────────────────────────────────────────────────────┘

    Unlike two-stage detectors (RCNN family) that first propose
    regions then classify them, YOLO does it in a single forward
    pass — that's what "You Only Look Once" means!
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        """
        Args:
            model_name: Which model to use. Options:
                        yolov8n.pt (nano)   ← default
                        yolov8s.pt (small)
                        yolov8m.pt (medium)
                        yolov8l.pt (large)
                        yolov8x.pt (extra-large)
            confidence: Minimum confidence score to keep a detection (0–1).
                        Lower = more detections but more false positives.
                        Higher = fewer but more accurate detections.
        """
        self.model_name = model_name
        self.confidence = confidence
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model. Weights auto-download on first run."""
        print(f"[YOLODetector] Loading model: {self.model_name}")
        start = time.time()
        self.model = YOLO(self.model_name)
        elapsed = time.time() - start
        print(f"[YOLODetector] Model loaded in {elapsed:.2f}s ✅")

    def detect_image(self, image_path: str) -> dict:
        """
        Run YOLOv8 detection on a single image file.

        Args:
            image_path: Path to the input image (jpg, png, bmp, etc.)

        Returns:
            dict with keys:
              - detections: list of detection dicts
              - count: total number of objects found
              - inference_ms: how long the model took (milliseconds)
              - class_summary: {class_name: count} breakdown
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        start = time.time()
        results = self.model(image_path, conf=self.confidence, verbose=False)
        inference_ms = (time.time() - start) * 1000

        return self._parse_results(results[0], inference_ms)

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Run YOLOv8 detection on a raw NumPy image frame (from OpenCV).

        Args:
            frame: BGR image array from cv2.VideoCapture

        Returns:
            Same dict structure as detect_image()
        """
        start = time.time()
        results = self.model(frame, conf=self.confidence, verbose=False)
        inference_ms = (time.time() - start) * 1000

        return self._parse_results(results[0], inference_ms)

    def detect_and_annotate(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Detect objects AND return the annotated frame in one call.
        Used by the webcam stream.

        Returns:
            (annotated_frame, detection_dict)
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        annotated = results[0].plot()  # YOLOv8's built-in visualizer
        detection_data = self._parse_results(results[0], 0)
        return annotated, detection_data

    def _parse_results(self, result, inference_ms: float) -> dict:
        """
        Convert raw YOLOv8 result object into a clean Python dict.

        The result object contains:
          - result.boxes.xyxy   → bounding box corners [x1, y1, x2, y2]
          - result.boxes.conf   → confidence scores
          - result.boxes.cls    → class indices (integers)
          - result.names        → {index: class_name} mapping
        """
        detections = []
        class_summary = {}

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()       # [N, 4]
            confs = result.boxes.conf.cpu().numpy()       # [N]
            classes = result.boxes.cls.cpu().numpy().astype(int)  # [N]
            names = result.names  # {0: 'person', 1: 'bicycle', ...}

            for box, conf, cls_id in zip(boxes, confs, classes):
                class_name = names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.tolist()

                detections.append({
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": round(float(conf), 4),
                    "confidence_pct": f"{conf * 100:.1f}%",
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                    },
                })

                # Update class summary
                class_summary[class_name] = class_summary.get(class_name, 0) + 1

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return {
            "detections": detections,
            "count": len(detections),
            "inference_ms": round(inference_ms, 1),
            "class_summary": class_summary,
            "model": self.model_name,
            "confidence_threshold": self.confidence,
        }


# ─── Module-level singleton (lazy loaded) ─────────────────────────────────────
_detector_instance: YOLODetector | None = None


def get_detector(model_name: str = "yolov8n.pt", confidence: float = 0.35) -> YOLODetector:
    """
    Returns a shared YOLODetector instance (creates on first call).
    This avoids reloading the model on every request.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector(model_name=model_name, confidence=confidence)
    return _detector_instance
