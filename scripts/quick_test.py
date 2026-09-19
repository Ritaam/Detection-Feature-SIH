"""
quick_test.py — Minimal smoke test for the detection module.

Runs directly without pytest or a real image file.
Creates a synthetic test frame and verifies the full pipeline:

    yolo11n.pt  →  Detector  →  PostProcessor  →  list[Detection]

Usage:
    python scripts/quick_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from src.detection import Detector, Detection
from src.detection.exceptions import DetectionError

# ── ANSI colours for terminal output ────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}"); sys.exit(1)
def info(msg): print(f"  {CYAN}→{RESET} {msg}")


def make_test_frame(h=640, w=640) -> np.ndarray:
    """Return a blank BGR uint8 frame (valid input for Detector)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def print_detections(detections: list[Detection]) -> None:
    if not detections:
        info("No detections in this frame (expected for blank frame).")
        return
    print(f"\n  {'CLASS':<14} {'CONF':>6}  {'BBOX (x1,y1,x2,y2)'}")
    print(f"  {'-'*14} {'-'*6}  {'-'*30}")
    for d in detections:
        print(f"  {d.class_name:<14} {d.confidence:>6.3f}  {d.bbox}")


def main() -> None:
    print(f"\n{BOLD}{'='*55}")
    print("  Detection Module — Smoke Test")
    print(f"{'='*55}{RESET}\n")

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"{BOLD}[1/5] Loading YOLO11n pretrained model ...{RESET}")
    t0 = time.perf_counter()
    try:
        detector = Detector(
            model_path="models/yolo11n.pt",   # pretrained COCO weights
            confidence_threshold=0.25, # lower threshold for test image
            iou_threshold=0.45,
            device="auto",
        )
    except DetectionError as e:
        fail(f"Failed to load model: {e}")
        return
    load_ms = (time.perf_counter() - t0) * 1000
    ok(f"Model loaded in {load_ms:.0f} ms  |  device: {detector.device}")
    ok(f"Full class list ({len(detector.class_names)} COCO classes available)")
    ok(f"Supported classes: {sorted(detector.config.supported_classes)}")

    # ── 2. Valid blank frame ──────────────────────────────────────────────────
    print(f"\n{BOLD}[2/5] Running detect() on blank 640×640 frame ...{RESET}")
    frame = make_test_frame()
    t0 = time.perf_counter()
    detections = detector.detect(frame)
    infer_ms = (time.perf_counter() - t0) * 1000
    ok(f"detect() returned in {infer_ms:.1f} ms")
    ok(f"Output type: {type(detections).__name__}  (never None)")
    ok(f"Detections: {len(detections)}")
    print_detections(detections)

    # ── 3. Input validation ───────────────────────────────────────────────────
    print(f"\n{BOLD}[3/5] Verifying input validation ...{RESET}")
    from src.detection.exceptions import InvalidFrameError

    bad_inputs = [
        (None,                                        "None frame"),
        (np.array([]),                                "Empty array"),
        (np.zeros((480, 640, 3), dtype=np.float32),  "Wrong dtype (float32)"),
        (np.zeros((480, 640, 1), dtype=np.uint8),    "Wrong channels (1-ch)"),
        (np.zeros((480, 640),    dtype=np.uint8),    "Wrong ndim (2D)"),
    ]
    for bad_input, label in bad_inputs:
        try:
            detector.detect(bad_input)  # type: ignore
            fail(f"Should have raised for: {label}")
        except InvalidFrameError:
            ok(f"InvalidFrameError raised correctly for: {label}")
        except Exception as e:
            fail(f"Wrong exception type for '{label}': {type(e).__name__}: {e}")

    # ── 4. Model loaded once (object identity) ────────────────────────────────
    print(f"\n{BOLD}[4/5] Verifying model is loaded only once ...{RESET}")
    id_before = id(detector._model)
    for _ in range(5):
        detector.detect(frame)
    id_after = id(detector._model)
    if id_before == id_after:
        ok("Model object identity unchanged across 5 detect() calls ✓")
    else:
        fail("Model was re-loaded between detect() calls!")

    # ── 5. Run on a real image if provided ────────────────────────────────────
    print(f"\n{BOLD}[5/5] Real image test ...{RESET}")
    import sys
    real_image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if real_image_path:
        img = cv2.imread(real_image_path)
        if img is None:
            fail(f"Could not read image: {real_image_path}")
        else:
            t0 = time.perf_counter()
            real_dets = detector.detect(img)
            infer_ms = (time.perf_counter() - t0) * 1000
            ok(f"detect() on '{Path(real_image_path).name}': "
               f"{len(real_dets)} detection(s) in {infer_ms:.1f} ms")
            print_detections(real_dets)

            # Draw and save annotated image
            for d in real_dets:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{d.class_name} {d.confidence:.2f}",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1, cv2.LINE_AA)
            out_path = "quick_test_output.jpg"
            cv2.imwrite(out_path, img)
            ok(f"Annotated image saved → {out_path}")
    else:
        info("No image path provided. Pass one as argument to test on a real image:")
        info("    python scripts/quick_test.py path/to/image.jpg")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}{'='*55}")
    print("  All checks passed ✓  Detection module is working.")
    print(f"{'='*55}{RESET}\n")


if __name__ == "__main__":
    main()
