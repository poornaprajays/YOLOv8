"""
Detection Utilities
====================
Helper functions for drawing bounding boxes and formatting detection output.
"""

import cv2
import numpy as np
from typing import List


# ─── Color palette for 80 COCO classes ────────────────────────────────────────
# Each class gets a consistent, visually distinct color
_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
]


def get_class_color(class_id: int) -> tuple:
    """Returns a consistent BGR color for a given class ID."""
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(frame: np.ndarray, detections: list, alpha: float = 0.85) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores on an image frame.

    Args:
        frame:      BGR image array (from OpenCV)
        detections: List of detection dicts from YOLODetector
        alpha:      Opacity for the label background (0–1)

    Returns:
        Annotated BGR image array
    """
    overlay = frame.copy()

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        color = get_class_color(det["class_id"])
        conf = det["confidence"]
        label = f"{det['class_name']} {conf:.2f}"

        # Draw filled rectangle (bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2)

        # Label background
        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(overlay, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)

        # Label text
        cv2.putText(
            overlay, label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness,
            lineType=cv2.LINE_AA,
        )

    # Blend overlay with original for semi-transparent boxes
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """
    Convert a NumPy BGR frame to JPEG bytes for streaming.

    Args:
        frame:   BGR image array
        quality: JPEG quality (1–100). 85 is a good balance.

    Returns:
        Raw JPEG bytes
    """
    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def resize_for_display(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """
    Resize a frame to fit within max_width while preserving aspect ratio.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def format_results_for_cli(results: dict) -> str:
    """
    Format detection results as a pretty CLI table string.

    Example output:
    ┌─────────────────┬────────────┬──────────────────────────┐
    │ Class           │ Confidence │ BBox (x1,y1,w,h)         │
    ├─────────────────┼────────────┼──────────────────────────┤
    │ person          │   94.3%    │ (120, 45, 180, 320)      │
    └─────────────────┴────────────┴──────────────────────────┘
    """
    if not results["detections"]:
        return "  No objects detected."

    lines = []
    lines.append(f"\n  Model      : {results['model']}")
    lines.append(f"  Inference  : {results['inference_ms']} ms")
    lines.append(f"  Objects    : {results['count']} detected")
    lines.append(f"  Threshold  : {results['confidence_threshold']}")
    lines.append("")
    lines.append(f"  {'#':<4} {'Class':<20} {'Confidence':>12}  BBox (x1,y1 → x2,y2)")
    lines.append("  " + "─" * 70)

    for i, det in enumerate(results["detections"], 1):
        b = det["bbox"]
        bbox_str = f"({b['x1']},{b['y1']}) → ({b['x2']},{b['y2']})"
        lines.append(
            f"  {i:<4} {det['class_name']:<20} {det['confidence_pct']:>12}  {bbox_str}"
        )

    lines.append("")
    lines.append("  Class Summary:")
    for cls, cnt in sorted(results["class_summary"].items(), key=lambda x: -x[1]):
        lines.append(f"    • {cls}: {cnt}")

    return "\n".join(lines)
