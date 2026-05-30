# 🎯 YOLOv8 Object Detection Explorer

> A complete, beginner-friendly project to **understand, run, and visualize** YOLOv8 object detection — with standalone scripts, a live webcam detector, and a beautiful web dashboard.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 What is YOLOv8?

### The Big Idea: You Only Look Once

Traditional object detectors worked in **two stages**:
1. First, propose regions of interest (where objects might be)
2. Then, classify each proposed region

This was slow because the network runs twice. YOLO changed everything in 2016 by asking:

> **"What if we did detection in a single forward pass?"**

That's the core idea behind **YOLO (You Only Look Once)** — the network sees the full image once, and simultaneously predicts:
- **Where** objects are (bounding boxes)
- **What** they are (class labels)
- **How confident** it is (confidence scores)

### YOLOv8 — The State of the Art (2023)

YOLOv8, released by [Ultralytics](https://ultralytics.com) in January 2023, is the latest and most powerful generation:

| Feature | What it means |
|---|---|
| **Anchor-Free Detection** | No longer needs pre-defined anchor boxes — faster training, better accuracy |
| **C2f Backbone** | Cross-Stage Partial network with 2 feature pathways — richer feature extraction |
| **FPN + PAN Neck** | Feature Pyramid Network + Path Aggregation Network — detects objects at all scales |
| **Decoupled Head** | Separate heads for classification and box regression — higher accuracy |
| **Ultralytics Library** | Clean Python API, auto-downloads weights, works out of the box |

### How Detection Works (Step by Step)

```
Input Image (any size, resized to 640x640 internally)
      |
      v
Backbone — C2f modules
  * Extracts hierarchical visual features
  * Early layers: edges, textures
  * Deep layers: semantic concepts (faces, wheels, etc.)
      |
      v
Neck — FPN + PAN
  * Fuses features from multiple scales
  * Ensures both small and large objects are detected well
      |
      v
Detection Head (Anchor-Free)
  * Predicts bounding box (x, y, w, h) per grid cell
  * Predicts class probabilities for all 80 classes
  * No anchor priors needed — truly end-to-end
      |
      v
NMS — Non-Maximum Suppression
  * Removes overlapping duplicate detections
  * Keeps the box with highest confidence per object
      |
      v
Final Detections: [(class, confidence, bbox), ...]
```

### Key Concepts Explained

**Bounding Box**: A rectangle defined by (x1, y1, x2, y2) corners that surrounds a detected object.

**Confidence Score**: How sure the model is that something is an object AND that it belongs to the predicted class. Range: 0 to 1.

**IoU (Intersection over Union)**: Measures how much two boxes overlap. Used to evaluate detection quality.
```
IoU = Area of Overlap / Area of Union
```

**mAP (mean Average Precision)**: The standard metric for object detection. Higher = better.
- mAP50 = measured at IoU threshold 0.50
- mAP50-95 = averaged over thresholds 0.50 to 0.95

**NMS (Non-Max Suppression)**: Algorithm that removes duplicate detections of the same object by keeping the highest-confidence box and removing all others that overlap with it above an IoU threshold.

---

## 🗂️ Project Structure

```
YOLOv8/
├── app/
│   ├── __init__.py        # Flask application factory
│   ├── detector.py        # ⭐ Core YOLOv8 engine (read this first!)
│   ├── routes.py          # Flask HTTP endpoints
│   └── utils.py           # Drawing, formatting helpers
│
├── scripts/
│   ├── detect_image.py    # Detect on a single image
│   ├── detect_video.py    # Detect on a video file
│   └── detect_webcam.py   # Real-time webcam detection
│
├── static/
│   ├── css/style.css      # Dashboard styles
│   ├── js/app.js          # Dashboard frontend logic
│   └── uploads/           # Temp files (auto-created)
│
├── templates/
│   └── index.html         # Web dashboard HTML
│
├── samples/               # Put your test images/videos here
│   └── README.md
│
├── requirements.txt       # Python dependencies
├── run.py                 # Flask server entry point
└── README.md              # This file!
```

### Start Here: `app/detector.py`

The `YOLODetector` class in `app/detector.py` is the heart of the project. It wraps Ultralytics YOLOv8 and exposes three clean methods:

```python
detector = YOLODetector(model_name="yolov8n.pt", confidence=0.35)

# 1. Detect on an image file
results = detector.detect_image("dog.jpg")

# 2. Detect on a NumPy frame (from OpenCV)
results = detector.detect_frame(frame)

# 3. Detect + get annotated frame in one call
annotated_frame, results = detector.detect_and_annotate(frame)
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip
- A webcam (optional, for live detection)
- GPU (optional, but makes it much faster)

### Step 1 — Open the project

```bash
cd "c:\Users\poorn\OneDrive\Desktop\YOLOv8"
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note**: When you first run any detection, Ultralytics will automatically download the `yolov8n.pt` weights (~6MB) from the internet. This only happens once.

---

## 🎮 Usage

### Option A: Web Dashboard (Recommended to start!)

```bash
python run.py
```

Then open: **http://localhost:5000**

The dashboard has 4 tabs:

| Tab | What you can do |
|---|---|
| Image Detection | Upload any image, get bounding boxes + confidence scores |
| Live Webcam | Real-time detection from your webcam (MJPEG stream) |
| COCO Classes | Browse all 80 detectable object categories |
| Model Sizes | Compare yolov8n vs yolov8s vs yolov8m vs yolov8l vs yolov8x |

---

### Option B: Detect on an Image

```bash
python scripts/detect_image.py --image samples/your_image.jpg
```

**All options:**
```
--image   PATH    Image file to detect on (required)
--model   NAME    Model: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
--conf    FLOAT   Confidence threshold 0-1 (default: 0.35)
--show            Show result in a window (press Q to close)
```

**Example output:**
```
Model      : yolov8n.pt
Inference  : 8.3 ms
Objects    : 3 detected

#    Class                Confidence  BBox (x1,y1 to x2,y2)
----------------------------------------------------------------------
1    dog                       94.1%  (120,45) to (380,480)
2    person                    87.3%  (400,10) to (640,500)
3    frisbee                   61.2%  (250,200) to (340,280)
```

The annotated image is saved as `your_image_detected.jpg` in the same folder.

---

### Option C: Detect on a Video

```bash
python scripts/detect_video.py --video samples/your_video.mp4
```

**All options:**
```
--video   PATH    Video file to detect on (required)
--model   NAME    YOLOv8 model variant
--conf    FLOAT   Confidence threshold (default: 0.35)
--skip    INT     Process every Nth frame (default: 1). Use 2-4 for faster processing.
```

Output is saved as `your_video_detected.mp4`.

---

### Option D: Real-Time Webcam

```bash
python scripts/detect_webcam.py
```

**Controls while running:**

| Key | Action |
|---|---|
| Q or ESC | Quit |
| S | Save screenshot |
| SPACE | Pause / Resume |
| + | Increase confidence threshold |
| - | Decrease confidence threshold |

**All options:**
```
--model    NAME   YOLOv8 model (default: yolov8n.pt)
--conf     FLOAT  Confidence threshold (default: 0.35)
--camera   INT    Camera index (default: 0). Try 1 or 2 for external webcams.
```

---

## 📊 Model Size Comparison

| Model | File Size | Parameters | mAP50-95 | CPU Speed | Best For |
|---|---|---|---|---|---|
| **yolov8n** (this project) | 6 MB | 3.2M | 37.3 | ~1.4ms | Learning, real-time CPU |
| yolov8s | 22 MB | 11.2M | 44.9 | ~2.4ms | Edge devices |
| yolov8m | 52 MB | 25.9M | 50.2 | ~5.2ms | Balanced |
| yolov8l | 87 MB | 43.7M | 52.9 | ~8.1ms | High accuracy |
| yolov8x | 136 MB | 68.2M | 53.9 | ~13.4ms | Maximum accuracy |

> This project uses **yolov8n** (nano) by default. It's the fastest, runs on CPU, and is perfect for learning. The weights auto-download on first use.

To switch models, just change the `--model` argument:
```bash
python scripts/detect_image.py --image dog.jpg --model yolov8s.pt
```

---

## 🏷️ COCO Dataset — 80 Object Classes

YOLOv8 is pre-trained on the [COCO dataset](https://cocodataset.org) (Common Objects in Context). Here are all 80 categories it can detect:

**People & Vehicles**: `person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`

**Traffic & Outdoor**: `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`

**Animals**: `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`

**Accessories**: `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`

**Sports**: `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`

**Kitchen & Food**: `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`

**Furniture & Indoors**: `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`

**Electronics**: `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`

**Miscellaneous**: `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

---

## 🔧 How the Code is Organized

### `app/detector.py` — The Engine

```python
class YOLODetector:
    def __init__(self, model_name, confidence):
        self.model = YOLO(model_name)  # Auto-downloads if needed

    def detect_image(self, image_path) -> dict:
        results = self.model(image_path, conf=self.confidence)
        return self._parse_results(results[0])

    def _parse_results(self, result) -> dict:
        # Converts raw YOLO tensors into clean Python dicts
        # boxes.xyxy  -> [x1, y1, x2, y2] for each detection
        # boxes.conf  -> confidence score per detection
        # boxes.cls   -> class index per detection
        ...
```

### `app/routes.py` — The Web API

```python
@main_bp.route("/detect/image", methods=["POST"])
def detect_image():
    # 1. Receive uploaded image
    # 2. Run YOLODetector
    # 3. Return annotated image as base64 + detection JSON

@main_bp.route("/webcam/stream")
def webcam_stream():
    # Generator function — yields MJPEG frames indefinitely
    # Browser renders each JPEG as it arrives = live video!
```

### `static/js/app.js` — The Frontend

```javascript
async function runDetection() {
    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("confidence", conf);

    const res = await fetch("/detect/image", { method: "POST", body: formData });
    const data = await res.json();
    renderResults(data);  // Updates the DOM with annotated image + boxes
}
```

---

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### `cv2.error: can't open camera`
- Make sure your webcam is connected
- Try a different camera index: `--camera 1` or `--camera 2`
- Close other apps that might be using the webcam

### `RuntimeError: CUDA out of memory`
- You don't need a GPU! The model runs fine on CPU.
- Alternatively, use a smaller model: `--model yolov8n.pt`

### Model not downloading
- Check your internet connection
- Ultralytics downloads weights from GitHub releases
- Manually download from: https://github.com/ultralytics/assets/releases

### Flask server not starting
```bash
# Make sure you're in the project root directory
cd "c:\Users\poorn\OneDrive\Desktop\YOLOv8"
python run.py
```

### Port 5000 already in use
```bash
# Kill the process using port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

---

## 📚 Learn More

| Resource | Link |
|---|---|
| Ultralytics YOLOv8 Docs | https://docs.ultralytics.com |
| Original YOLO Paper (2016) | https://arxiv.org/abs/1506.02640 |
| COCO Dataset | https://cocodataset.org |
| YOLOv8 GitHub | https://github.com/ultralytics/ultralytics |

---

---

*Built with love using [Ultralytics YOLOv8](https://ultralytics.com) · [Flask](https://flask.palletsprojects.com) · [OpenCV](https://opencv.org)*
