#!/usr/bin/env python3
"""
Detect Objects in an Image — Standalone Script
================================================
Usage:
    python scripts/detect_image.py --image path/to/image.jpg
    python scripts/detect_image.py --image photo.jpg --conf 0.5 --model yolov8s.pt

What this script does:
  1. Loads a YOLOv8 model
  2. Runs object detection on your image
  3. Saves an annotated copy with bounding boxes drawn
  4. Prints a clean detection table to your terminal

This is the simplest way to see YOLOv8 in action!
"""

import argparse
import os
import sys
import cv2

# Add parent directory to path so we can import from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import YOLODetector
from app.utils import format_results_for_cli


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Image Detection — YOLOv8 Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/detect_image.py --image dog.jpg
  python scripts/detect_image.py --image traffic.png --conf 0.25
  python scripts/detect_image.py --image photo.jpg --model yolov8s.pt --save
        """,
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                        help="YOLOv8 model variant (default: yolov8n — fastest)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold 0–1 (default: 0.35)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated image (default: True)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show annotated image in a window")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "━" * 60)
    print("  🎯  YOLOv8 Image Detection")
    print("━" * 60)
    print(f"  Image : {args.image}")
    print(f"  Model : {args.model}")
    print(f"  Conf  : {args.conf}")
    print("━" * 60)

    # Validate image path
    if not os.path.exists(args.image):
        print(f"\n❌ Error: Image not found: {args.image}")
        sys.exit(1)

    # Load model and detect
    detector = YOLODetector(model_name=args.model, confidence=args.conf)
    results = detector.detect_image(args.image)

    # Print detection table
    print(format_results_for_cli(results))

    # Save annotated image
    if args.save or args.show:
        # Re-run with built-in annotator for best visuals
        from ultralytics import YOLO
        model = YOLO(args.model)
        raw = model(args.image, conf=args.conf, verbose=False)
        annotated = raw[0].plot()

        if args.save:
            base, ext = os.path.splitext(args.image)
            output_path = f"{base}_detected{ext}"
            cv2.imwrite(output_path, annotated)
            print(f"\n  ✅ Annotated image saved: {output_path}")

        if args.show:
            cv2.imshow("YOLOv8 Detection — press Q to close", annotated)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            cv2.destroyAllWindows()

    print("\n  Done! 🎉\n")


if __name__ == "__main__":
    main()
