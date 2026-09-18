# AI-Based Intelligent Video Analytics Platform
### Border Surveillance — Detection Module

---

## Table of Contents

1. [What this module does](#1-what-this-module-does)
2. [Project structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Model — YOLO11n](#4-model--yolo11n)
5. [Quick start](#5-quick-start)
6. [Input format](#6-input-format)
7. [Output format](#7-output-format)
8. [Supported classes](#8-supported-classes)
9. [Configuration](#9-configuration)
10. [CPU / GPU usage](#10-cpu--gpu-usage)
11. [Running the demo script](#11-running-the-demo-script)
12. [Running tests](#12-running-tests)
13. [Architecture and future pipeline](#13-architecture-and-future-pipeline)
14. [Limitations and assumptions](#14-limitations-and-assumptions)

---

## 1. What this module does

`src/detection` is a **self-contained, model-agnostic detection module** that accepts a single BGR image frame and returns a list of standardised `Detection` objects.

It answers exactly one question:

> *"What supported objects are present in this frame, where are they, and how confident is the model?"*

It does **not** perform: tracking, ANPR, face recognition, behaviour analysis, intrusion detection, alert generation, or any database / network I/O.

---

## 2. Project structure

```
src/
├── config.py                        # Global project configuration
├── __init__.py
│
├── detection/                       # ✅ IMPLEMENTED
│   ├── __init__.py                  # Public API: Detector, Detection, DetectorConfig
│   ├── detector.py                  # Detector class — loads model, runs inference
│   ├── models.py                    # Detection dataclass (frozen, immutable)
│   ├── config.py                    # DetectorConfig frozen dataclass
│   ├── postprocess.py               # PostProcessor — filtering, clamping, conversion
│   └── exceptions.py                # DetectionError, ModelLoadError, InvalidFrameError, InferenceError
│
├── tracking/                        # 🔲 Stub — BoT-SORT / ByteTrack
├── anpr/                            # 🔲 Stub — Licence plate detection + OCR
├── face/                            # 🔲 Stub — Face detection / recognition
├── analytics/                       # 🔲 Stub — Behaviour analysis
├── pipeline/                        # 🔲 Stub — Frame scheduling & orchestration
├── api_client/                      # 🔲 Stub — REST / WebSocket client
├── utils/                           # 🔲 Stub — Shared utilities
│
└── night_detection/                 # 🔲 Stub — Thermal / IR pipeline
    ├── __init__.py
    ├── thermal_preprocessing.py     # Normalise, CLAHE, denoise
    ├── hotspot_segmentation.py      # Contour / blob detection
    ├── thermal_detector.py          # ThermalDetector + ThermalDetection
    ├── tracker.py                   # ThermalTracker + TrackedThermalDetection
    ├── movement_analysis.py         # MovementAnalyser + MovementEvent
    └── alert_engine.py              # AlertEngine + NightAlert

tests/
└── detection/
    ├── test_detector.py             # 14 detector test classes (mock-based)
    ├── test_models.py               # Detection dataclass tests
    └── test_postprocess.py          # PostProcessor tests (no model required)

scripts/
└── test_detection.py                # Visual demo — image / video / webcam

models/                              # Place downloaded .pt files here
requirements.txt
pytest.ini
```

---

## 3. Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

For GPU inference (CUDA 12.1):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Model — YOLO11n

| Property | Value |
|---|---|
| **Model** | YOLO11n (Ultralytics, 2024) |
| **Weights** | COCO pretrained |
| **Parameters** | ~2.6 M |
| **Input size** | 640 × 640 (default) |
| **Why chosen** | Fastest YOLO11 variant; real-time on CPU; all 5 required classes in COCO |

Weights are **downloaded automatically** on first use when `model_path="yolo11n.pt"` is passed.  
To bundle weights: download `yolo11n.pt` to the `models/` directory and pass the full path.

```python
detector = Detector(model_path="models/yolo11n.pt")
```

> **Note:** YOLO26 does not exist. YOLO11n is the latest Ultralytics release.  
> When a future version is released, swap `model_path` in `DetectorConfig` — no other code changes needed.

---

## 5. Quick start

```python
import cv2
from src.detection import Detector, Detection

# Initialise once — model is loaded here, NOT inside detect()
detector = Detector(
    model_path="yolo11n.pt",       # auto-downloads on first use
    confidence_threshold=0.5,
    iou_threshold=0.45,
    device="auto",                  # CUDA if available, else CPU
)

# Run on any OpenCV frame
frame = cv2.imread("test.jpg")
detections: list[Detection] = detector.detect(frame)

for d in detections:
    print(f"{d.class_name:12s}  conf={d.confidence:.2f}  bbox={d.bbox}")
```

**Example output:**

```
person        conf=0.94  bbox=(120.0, 80.0, 300.0, 500.0)
car           conf=0.91  bbox=(500.0, 250.0, 800.0, 450.0)
truck         conf=0.78  bbox=(820.0, 200.0, 1100.0, 480.0)
```

---

## 6. Input format

| Property | Requirement |
|---|---|
| Type | `numpy.ndarray` |
| Shape | `(H, W, 3)` |
| dtype | `uint8` |
| Channel order | BGR (standard OpenCV) |
| Min size | At least 1 pixel in each dimension |

**Validation errors raised:**

```python
detector.detect(None)                              # → InvalidFrameError: Frame is None
detector.detect(np.zeros((0, 640, 3), np.uint8))  # → InvalidFrameError: empty
detector.detect(np.zeros((480, 640, 1), np.uint8))# → InvalidFrameError: 3 channels
detector.detect(np.zeros((480, 640, 3), np.float32)) # → InvalidFrameError: uint8
```

---

## 7. Output format

Returns `list[Detection]`. Never returns `None`. Returns `[]` when nothing is detected.

```python
@dataclass(frozen=True, slots=True)
class Detection:
    class_id:   int                              # COCO class index
    class_name: str                              # e.g. "person", "car"
    confidence: float                            # [0.0, 1.0]
    bbox:       tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
```

**Bounding box convention:**

```
(x1, y1) ─────────────┐
    │                  │
    │                  │
    └──────────── (x2, y2)

Origin: top-left of frame  |  Coordinates: absolute pixels  |  Guaranteed: x1 < x2, y1 < y2
```

**Convenience properties:**

```python
d.width   # x2 - x1
d.height  # y2 - y1
d.area    # width × height
d.center  # ((x1+x2)/2, (y1+y2)/2)
```

---

## 8. Supported classes

| class_name | COCO class_id |
|---|---|
| `person` | 0 |
| `car` | 2 |
| `motorcycle` | 3 |
| `bus` | 5 |
| `truck` | 7 |

All other COCO classes (bicycle, airplane, boat, etc.) are **silently filtered out**.

To add more classes, pass `supported_classes` to `Detector`:

```python
from src.detection import Detector
from src.detection.config import DEFAULT_SUPPORTED_CLASSES

detector = Detector(
    model_path="yolo11n.pt",
    supported_classes=DEFAULT_SUPPORTED_CLASSES | {"bicycle"},
)
```

---

## 9. Configuration

```python
from src.detection.config import DetectorConfig

config = DetectorConfig(
    model_path="yolo11n.pt",
    confidence_threshold=0.5,    # float, (0, 1]
    iou_threshold=0.45,          # float, (0, 1]
    device="auto",               # "auto" | "cpu" | "cuda" | "cuda:0" | "mps"
    supported_classes=frozenset({"person", "car", "motorcycle", "bus", "truck"}),
    imgsz=640,                   # int, multiple of 32
)
```

Pass `config` directly to `Detector`:

```python
detector = Detector(
    model_path=config.model_path,
    confidence_threshold=config.confidence_threshold,
    iou_threshold=config.iou_threshold,
    device=config.device,
)
```

---

## 10. CPU / GPU usage

| Device string | Behaviour |
|---|---|
| `"auto"` (default) | Uses CUDA if available, else MPS (Apple), else CPU |
| `"cpu"` | Forces CPU inference |
| `"cuda"` / `"cuda:0"` | Forces CUDA — raises at runtime if unavailable |
| `"mps"` | Apple Silicon GPU (macOS only) |

```python
# Explicit CPU (useful on machines without GPU)
detector = Detector(model_path="yolo11n.pt", device="cpu")

# Multi-GPU: use second GPU
detector = Detector(model_path="yolo11n.pt", device="cuda:1")
```

---

## 11. Running the demo script

```bash
# Single image
python scripts/test_detection.py --source path/to/image.jpg

# Webcam (device 0)
python scripts/test_detection.py --source 0

# Video file
python scripts/test_detection.py --source path/to/video.mp4

# Save annotated output
python scripts/test_detection.py --source image.jpg --output result.jpg

# Custom threshold + headless (no GUI window)
python scripts/test_detection.py --source image.jpg --conf 0.4 --no-display

# Custom model path
python scripts/test_detection.py --source image.jpg --model models/yolo11n.pt
```

---

## 12. Running tests

```bash
# Run all unit tests (no model download required)
pytest

# With coverage report
pytest --cov=src/detection --cov-report=term-missing

# Run integration tests (downloads yolo11n.pt ~6 MB on first run)
pytest --run-integration
```

**Test suite covers:**

| # | Test | Model required? |
|---|---|---|
| 1 | Valid frame → `list[Detection]` | Mock |
| 2 | `None` frame → `InvalidFrameError` | Mock |
| 3 | Empty frame → `InvalidFrameError` | Mock |
| 4 | Wrong dtype → `InvalidFrameError` | Mock |
| 5 | Wrong channels → `InvalidFrameError` | Mock |
| 6 | Wrong ndim → `InvalidFrameError` | Mock |
| 7 | No objects → `[]` | Mock |
| 8 | Unsupported class filtered | Mock |
| 9 | Confidence threshold filtering | Mock |
| 10 | `x1 < x2`, `y1 < y2` guaranteed | Mock |
| 11 | Model loaded only once | Mock |
| 12 | CPU execution | Mock |
| 13 | `ModelLoadError` on bad path | Mock |
| 14 | Output always `list`, never `None` | Mock |
| Integration | Real YOLO11n on blank frame | **Real model** |

---

## 13. Architecture and future pipeline

```
CCTV Frame (np.ndarray)
        │
        ▼
  src.detection.Detector          ← IMPLEMENTED
        │
        ▼
  list[Detection]
        │
        ▼
  src.tracking.Tracker            ← BoT-SORT / ByteTrack  [stub]
        │
        ▼
  list[TrackedDetection]
        │
   ┌────┼──────────┬─────────────┐
   ▼    ▼          ▼             ▼
 ANPR  Face   Analytics     Intrusion
[stub] [stub]  [stub]         [stub]
        │
        ▼
  src.api_client                  [stub]
        │
        ▼
  Backend / Frontend
```

The `Detection` dataclass is the **contract** between detection and all downstream modules.  
Downstream modules consume `list[Detection]` — they must not call `Detector.detect()` themselves.

---

## 14. Limitations and assumptions

- **YOLO11n is not YOLO26.** YOLO26 does not exist as of September 2026. When it is released, replace `model_path` — no other code changes needed.
- The detector is **synchronous**. Frame scheduling and concurrency are the pipeline's responsibility (`src.pipeline`).
- Coordinates are in **pixel space** relative to the input frame, not normalised [0, 1].
- `confidence_threshold` is applied **twice** — once inside YOLO's predict (for speed) and once in `PostProcessor` (for correctness). Both use the same configured value.
- The demo script requires a display for `--source 0` (webcam). Use `--no-display --output out.mp4` on headless servers.
