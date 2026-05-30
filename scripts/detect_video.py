#!/usr/bin/env python3
"""
Detect Objects in a Video File — Standalone Script
====================================================
Usage:
    python scripts/detect_video.py --video path/to/video.mp4
    python scripts/detect_video.py --video traffic.mp4 --conf 0.4 --model yolov8s.pt

What this script does:
  1. Opens a video file frame by frame
  2. Runs YOLOv8 detection on each frame
  3. Draws bounding boxes on every frame
  4. Saves the annotated video as a new file
  5. Prints a progress bar and summary stats

Tips:
  - Large videos take longer — try on a short clip (5–30 seconds) first
  - Use yolov8n for speed or yolov8m for better accuracy
  - The output file will be saved next to the input with '_detected' suffix
"""

import argparse
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import YOLODetector
from app.utils import get_class_color


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Video Detection — YOLOv8 Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/detect_video.py --video traffic.mp4
  python scripts/detect_video.py --video clip.mp4 --conf 0.4 --skip 2
        """,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                        help="YOLOv8 model variant (default: yolov8n)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (default: 1 = every frame). Use 2 for 2× speed.")
    return parser.parse_args()


def draw_frame_info(frame, frame_num, total_frames, fps, count):
    """Overlay frame info text on the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    texts = [
        f"Frame: {frame_num}/{total_frames}",
        f"FPS: {fps:.1f}",
        f"Objects: {count}",
    ]
    for i, txt in enumerate(texts):
        cv2.putText(frame, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    print("\n" + "━" * 60)
    print("  🎬  YOLOv8 Video Detection")
    print("━" * 60)
    print(f"  Video : {args.video}")
    print(f"  Model : {args.model}")
    print(f"  Conf  : {args.conf}")
    print(f"  Skip  : every {args.skip} frame(s)")
    print("━" * 60)

    if not os.path.exists(args.video):
        print(f"\n❌ Error: Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"\n❌ Error: Could not open video: {args.video}")
        sys.exit(1)

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Resolution : {width}×{height}")
    print(f"  FPS        : {orig_fps:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {total_frames/orig_fps:.1f}s\n")

    # Output video writer
    base, ext = os.path.splitext(args.video)
    output_path = f"{base}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

    # Load model
    detector = YOLODetector(model_name=args.model, confidence=args.conf)

    # Stats tracking
    frame_num = 0
    processed = 0
    total_detections = 0
    class_counts = {}
    start_time = time.time()
    last_annotated = None  # Cache last annotated frame for skipped frames

    print("  Processing frames...")
    print("  " + "─" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % args.skip == 0:
            # Run detection
            annotated, results = detector.detect_and_annotate(frame)
            last_annotated = annotated
            processed += 1
            total_detections += results["count"]

            # Accumulate class stats
            for cls, cnt in results["class_summary"].items():
                class_counts[cls] = class_counts.get(cls, 0) + cnt

            # Overlay frame info
            elapsed = time.time() - start_time
            fps = processed / elapsed if elapsed > 0 else 0
            annotated = draw_frame_info(annotated, frame_num, total_frames, fps, results["count"])
        else:
            annotated = last_annotated if last_annotated is not None else frame

        writer.write(annotated)

        # Progress bar
        progress = frame_num / total_frames
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        elapsed = time.time() - start_time
        eta = (elapsed / frame_num) * (total_frames - frame_num) if frame_num > 0 else 0
        print(f"\r  [{bar}] {progress*100:.1f}%  ETA: {eta:.0f}s", end="", flush=True)

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    print(f"\n\n  ✅ Done! Processed {frame_num} frames in {total_time:.1f}s")
    print(f"  Output saved: {output_path}")
    print(f"\n  Detection Summary:")
    print(f"    Total detections : {total_detections}")
    print(f"    Avg per frame    : {total_detections/max(processed, 1):.1f}")
    if class_counts:
        print(f"    Top classes:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"      • {cls}: {cnt}")
    print()


if __name__ == "__main__":
    main()
