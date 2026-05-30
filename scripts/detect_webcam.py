#!/usr/bin/env python3
"""
Real-Time Webcam Detection — Standalone Script
================================================
Usage:
    python scripts/detect_webcam.py
    python scripts/detect_webcam.py --conf 0.4 --model yolov8s.pt --camera 1

What this script does:
  1. Opens your webcam (camera index 0 by default)
  2. Runs YOLOv8 on EVERY FRAME in real time
  3. Draws bounding boxes and labels live
  4. Shows FPS counter in the corner

Controls (while the window is open):
  Q or ESC  →  Quit
  S         →  Save a screenshot
  +/-       →  Increase/decrease confidence threshold
  SPACE     →  Pause/resume

Note:
  Real-time performance depends on your hardware.
  On a modern CPU: ~15–30 FPS with yolov8n
  On a GPU:        ~60+ FPS with yolov8n
"""

import argparse
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import YOLODetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Real-Time Webcam Detection — YOLOv8 Explorer",
    )
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                        help="YOLOv8 model (default: yolov8n)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    return parser.parse_args()


def draw_hud(frame, fps, count, conf, paused):
    """Draw the heads-up display (FPS, object count, shortcuts)."""
    h, w = frame.shape[:2]

    # Top-left info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    status = "⏸ PAUSED" if paused else "● LIVE"
    lines = [
        f"{status}",
        f"FPS: {fps:.1f}",
        f"Objects: {count}",
        f"Conf: {conf:.2f}",
    ]
    colors = [(0, 255, 150), (255, 255, 255), (255, 200, 0), (200, 200, 200)]
    for i, (txt, col) in enumerate(zip(lines, colors)):
        cv2.putText(frame, txt, (10, 25 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1, cv2.LINE_AA)

    # Bottom shortcuts bar
    shortcuts = "Q/ESC: quit  |  S: screenshot  |  SPACE: pause  |  +/-: confidence"
    cv2.putText(frame, shortcuts, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()
    conf = args.conf

    print("\n" + "━" * 60)
    print("  📷  YOLOv8 Real-Time Webcam Detection")
    print("━" * 60)
    print(f"  Model    : {args.model}")
    print(f"  Camera   : index {args.camera}")
    print(f"  Conf     : {conf}")
    print("━" * 60)
    print("\n  Controls:")
    print("    Q / ESC  → Quit")
    print("    S        → Save screenshot")
    print("    SPACE    → Pause / Resume")
    print("    + / -    → Adjust confidence threshold")
    print("\n  Starting webcam... (may take a moment)\n")

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Could not open camera index {args.camera}")
        print("   Try: --camera 1  or  --camera 2")
        sys.exit(1)

    # Optimize webcam buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Load detector
    detector = YOLODetector(model_name=args.model, confidence=conf)

    # FPS tracking
    fps = 0.0
    frame_times = []
    paused = False
    screenshot_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "static", "uploads"
    )
    os.makedirs(screenshot_dir, exist_ok=True)
    last_annotated = None
    last_count = 0

    print("  ✅ Webcam ready! Window opening...")

    while True:
        t_start = time.time()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n❌ Failed to read from webcam")
                break

            # Update detector confidence if it changed
            detector.confidence = conf

            # Run detection
            annotated, results = detector.detect_and_annotate(frame)
            last_annotated = annotated
            last_count = results["count"]
        else:
            annotated = last_annotated if last_annotated is not None else frame

        # Calculate FPS (rolling average over last 30 frames)
        frame_times.append(time.time() - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        # Draw HUD
        annotated = draw_hud(annotated, fps, last_count, conf, paused)

        # Show frame
        cv2.imshow("YOLOv8 Webcam — press Q to quit", annotated)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            print("\n  Quitting...")
            break

        elif key == ord("s"):  # Screenshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"webcam_{ts}.jpg")
            cv2.imwrite(path, annotated)
            print(f"\n  📸 Screenshot saved: {path}")

        elif key == ord(" "):  # Pause
            paused = not paused
            print(f"\n  {'⏸ Paused' if paused else '▶ Resumed'}")

        elif key == ord("+") or key == ord("="):
            conf = min(0.95, conf + 0.05)
            print(f"\n  Confidence → {conf:.2f}")

        elif key == ord("-"):
            conf = max(0.05, conf - 0.05)
            print(f"\n  Confidence → {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Webcam released. Bye! 👋\n")


if __name__ == "__main__":
    main()
