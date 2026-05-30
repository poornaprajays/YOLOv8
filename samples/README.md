# 📸 Sample Images / Videos

Place your test images and videos here to try with the detection scripts.

## Quick Test Images (Free Sources)

You can download sample images from:
- **Unsplash** (free): https://unsplash.com
  - Search for: "street", "traffic", "dog", "people", "kitchen"
- **Pexels** (free): https://pexels.com
- **COCO val2017 images**: http://images.cocodataset.org/val2017/

## Quick Commands

Once you have an image in this folder:

```bash
# Detect objects in an image
python scripts/detect_image.py --image samples/your_image.jpg

# Detect on a video
python scripts/detect_video.py --video samples/your_video.mp4

# Real-time webcam
python scripts/detect_webcam.py
```

## Supported Formats

| Type  | Formats                        |
|-------|-------------------------------|
| Image | .jpg .jpeg .png .bmp .webp    |
| Video | .mp4 .avi .mov .mkv           |
