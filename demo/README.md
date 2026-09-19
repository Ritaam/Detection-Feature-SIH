# Thermal Detection Engine — Live Video Demo

This directory contains the standalone demonstration application for the IBVAP Thermal Detection Engine located in [`src/thermal_detection/`](../src/thermal_detection/).

---

## 1. Purpose

The demo proves that the thermal detection module operates end-to-end on thermal/infrared video streams without requiring any external tracking or production pipeline dependencies:

```text
Thermal / IR Video Frame (uint8 / uint16 / float32)
        ↓
ThermalPreprocessor (Normalization + CLAHE + Denoising + 3-channel format)
        ↓
YOLO Inference (Loaded once during initialization)
        ↓
ThermalPostProcessor (Confidence filtering + Class filtering + BBox clamping)
        ↓
Standardized ThermalDetection[] Value Objects
        ↓
Real-Time HUD Annotated Video & Live Visualization
```

It is intended for:
* **Development Verification**: Proving that thermal frames are read, preprocessed, and detected properly.
* **Team Demonstrations & SIH Evaluation**: Explaining the thermal detection architecture clearly to judges and technical evaluators.

---

## 2. Quick Start

### Standard Run with GUI Window
```powershell
python demo/thermal_video_demo.py --video data/sample_thermal.mp4
```

### Headless Mode (Automated / CI / Servers)
```powershell
python demo/thermal_video_demo.py --video data/sample_thermal.mp4 --no-display
```

### Recording the Annotated Output Video
```powershell
python demo/thermal_video_demo.py --video data/sample_thermal.mp4 --save output/thermal_demo.mp4
```

---

## 3. Command-Line Options

| Argument | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | `-v` | `data/sample_thermal.mp4` | Path to the thermal/IR video source (`.mp4`, `.avi`, `.mov`, `.mkv`). |
| `--model` | `-m` | `models/yolo11n.pt` | Path to YOLO model weights. |
| `--confidence` | `-c`, `--conf` | `0.25` | Confidence threshold in `(0.0, 1.0]`. |
| `--device` | `-d` | `auto` | Target device: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`. |
| `--output` / `--save` | `-o` | `None` | Save annotated video to specified destination file. |
| `--no-display` | | `False` | Run in headless mode without opening an OpenCV GUI window. |
| `--debug` | | `False` | Enable debug pipeline inspection in `ThermalDetectorConfig`. |
| `--print-interval`| | `15` | Terminal logging frequency (every N frames). |
| `--max-frames` | | `None` | Max frames to process before cleanly exiting. |

---

## 4. Interactive Keyboard Controls

When running in GUI mode (default), the following controls are available:

* `SPACE` : **Pause / Resume** playback. While paused, frame inspection is frozen.
* `S` : **Save Snapshot**. Exports the current annotated frame to `output/snapshot_frame_<ID>.png`.
* `Q` or `ESC` : **Quit** demo cleanly and print summary statistics.

---

## 5. Output Explanation

### Visual HUD & Bounding Boxes
* **Green Bounding Boxes**: Human detections (`person`).
* **Orange/Cyan/Magenta Bounding Boxes**: Vehicles (`car`, `bus`, `truck`, `motorcycle`).
* **Top HUD Banner**: Displays system identity (`IBVAP - THERMAL DETECTION ENGINE`) and status.
* **Metrics Card**: Real-time object count, inference latency (ms), and frame throughput (FPS).

### Standardized `ThermalDetection` Objects
The demo strictly outputs and verifies the project's canonical `ThermalDetection` dataclass instances:

```text
ThermalDetection(
    bbox=(59.9, 222.9, 212.8, 479.4),
    confidence=0.92,
    class_id=0,
    class_name="person",
    mean_intensity=None,
    temperature_range=None
)
```

> **Thermal Data Integrity Rule**: Pixel intensity is not temperature. Calibrated temperature values are never fabricated from grayscale intensity and are only displayed if radiometric calibration data is present.

---

## 6. Model Limitations & Baseline Disclosure

> **IMPORTANT**: The current demo validates the thermal detection pipeline and model integration. The default YOLO11n COCO weights are used as a baseline because dedicated thermal-trained weights are not included in the repository. Detection accuracy on uncalibrated thermal imagery has not been established by this demo.

When fine-tuned thermal weights (e.g. trained on FLIR or KAIST thermal benchmarks) are available, they can be plugged in directly with zero code modifications:

```powershell
python demo/thermal_video_demo.py --video path/to/thermal_video.mp4 --model models/thermal_fine_tuned.pt
```

---

## 7. Generating Sample Video

If you do not have an external thermal video handy, a synthetic infrared test video can be generated with:

```powershell
python demo/generate_sample_thermal.py --output data/sample_thermal.mp4 --frames 150
```
