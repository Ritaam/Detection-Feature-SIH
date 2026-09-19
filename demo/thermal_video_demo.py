#!/usr/bin/env python3
"""
thermal_video_demo.py — Live Thermal Detection Engine Demonstration.

Demonstrates the existing `src/thermal_detection/` engine on thermal/IR video.
Provides live visualization, HUD performance metrics overlay, standardized
ThermalDetection object inspection, video recording, and interactive controls.

Usage:
    python demo/thermal_video_demo.py --video path/to/thermal_video.mp4
    python demo/thermal_video_demo.py --video data/sample_thermal.mp4 --save
    python demo/thermal_video_demo.py --video data/sample_thermal.mp4 --no-display
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermal_detection import (
    ThermalDetection,
    ThermalDetector,
    ThermalDetectorConfig,
)
from src.thermal_detection.exceptions import (
    ThermalInferenceError,
    ThermalInvalidFrameError,
    ThermalModelLoadError,
)
from src.thermal_detection.visualization import draw_thermal_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("thermal_demo")


# ---------------------------------------------------------------------------
# HUD & Visual Overlay Utilities
# ---------------------------------------------------------------------------

def draw_hud_overlay(
    canvas: np.ndarray,
    frame_idx: int,
    total_frames: int,
    num_detections: int,
    infer_ms: float,
    instant_fps: float,
    model_name: str,
    device: str,
    is_paused: bool = False,
) -> np.ndarray:
    """Render a clean information HUD on top of the annotated frame.

    Displays:
    - Header: IBVAP - THERMAL DETECTION ENGINE
    - Objects detected count
    - Inference time (ms)
    - Instantaneous / Running FPS
    - Frame counter and progress
    - Active model and hardware device
    - Interactive keyboard controls hint

    Parameters
    ----------
    canvas:
        Annotated BGR frame (H×W×3, uint8).
    frame_idx:
        Current frame index (1-based).
    total_frames:
        Total frames in video source.
    num_detections:
        Number of objects detected in the current frame.
    infer_ms:
        Inference execution time in milliseconds.
    instant_fps:
        Current processing FPS.
    model_name:
        Loaded model filename.
    device:
        Resolved compute device (e.g. CPU, CUDA:0).
    is_paused:
        Whether playback is currently paused.

    Returns
    -------
    np.ndarray
        Frame with HUD overlay applied.
    """
    h, w = canvas.shape[:2]
    out = canvas.copy()

    # --- Top Header Bar (Semi-transparent dark slate) ---
    header_h = 42
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 24), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    # Title text
    title = "IBVAP - THERMAL DETECTION ENGINE"
    cv2.putText(
        out,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (0, 220, 255),  # Amber gold
        1,
        cv2.LINE_AA,
    )

    # Playback status badge
    status_text = "PAUSED" if is_paused else "LIVE"
    status_bg_color = (0, 0, 180) if is_paused else (0, 160, 40)
    badge_w = 70
    badge_x1 = w - badge_w - 14
    cv2.rectangle(out, (badge_x1, 8), (w - 14, 34), status_bg_color, -1)
    cv2.putText(
        out,
        status_text,
        (badge_x1 + (12 if is_paused else 18), 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # --- Left HUD Info Panel ---
    panel_w = 260
    panel_h = 135
    panel_x1, panel_y1 = 14, header_h + 10
    panel_x2, panel_y2 = panel_x1 + panel_w, panel_y1 + panel_h

    # Semi-transparent dark background for readability
    sub_overlay = out.copy()
    cv2.rectangle(sub_overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 15, 18), -1)
    cv2.addWeighted(sub_overlay, 0.75, out, 0.25, 0, out)
    cv2.rectangle(out, (panel_x1, panel_y1), (panel_x2, panel_y2), (60, 60, 70), 1)

    # Metrics lines
    progress_str = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
    pct_str = f"({(frame_idx / total_frames * 100):.0f}%)" if total_frames > 0 else ""

    lines = [
        (f"Frame       : {progress_str} {pct_str}", (220, 220, 220)),
        (f"Objects     : {num_detections}", (0, 255, 120) if num_detections > 0 else (180, 180, 180)),
        (f"Inference   : {infer_ms:.1f} ms", (255, 215, 0)),
        (f"FPS         : {instant_fps:.1f}", (255, 255, 255)),
        (f"Model       : {model_name}", (200, 200, 200)),
        (f"Device      : {device.upper()}", (140, 220, 255)),
    ]

    line_y = panel_y1 + 22
    for text, color in lines:
        cv2.putText(
            out,
            text,
            (panel_x1 + 10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )
        line_y += 19

    # --- Bottom Control Help Bar ---
    footer_h = 24
    footer_y = h - footer_h
    foot_overlay = out.copy()
    cv2.rectangle(foot_overlay, (0, footer_y), (w, h), (15, 15, 18), -1)
    cv2.addWeighted(foot_overlay, 0.80, out, 0.20, 0, out)

    help_msg = "[SPACE] Pause/Resume    [S] Snapshot    [Q / ESC] Quit"
    cv2.putText(
        out,
        help_msg,
        (14, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return out


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live demonstration of the existing Thermal Detection Engine on thermal/IR video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to thermal / IR video file (e.g. data/sample_thermal.mp4).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt). Defaults to models/yolo11n.pt or yolo11n.pt.",
    )
    parser.add_argument(
        "--confidence", "-c", "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0.0 to 1.0).",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Inference device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save annotated output video (e.g. output/thermal_demo.mp4).",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="output/thermal_demo.mp4",
        default=None,
        help="Save annotated video. Can optionally specify output file path.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying a GUI window (useful for headless / automated runs).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in ThermalDetectorConfig.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=15,
        help="Print detection details to terminal every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (optional).",
    )

    return parser.parse_args()


def resolve_video_path(video_arg: str | None) -> Path:
    """Locate the video file, with fallback checks for repository samples."""
    if video_arg:
        vp = Path(video_arg)
        if vp.is_file():
            return vp.resolve()
        raise FileNotFoundError(f"Specified video file not found: {video_arg}")

    # Check known sample locations
    default_candidates = [
        PROJECT_ROOT / "data" / "sample_thermal.mp4",
        PROJECT_ROOT / "data" / "thermal_video.mp4",
        PROJECT_ROOT / "sample_thermal.mp4",
        PROJECT_ROOT / "thermal_video.mp4",
    ]
    for cand in default_candidates:
        if cand.is_file():
            logger.info("No --video argument provided. Using default sample: %s", cand)
            return cand.resolve()

    # Search project directory for any video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    found_videos = [
        p for p in PROJECT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    if found_videos:
        logger.info("Using discovered video file: %s", found_videos[0])
        return found_videos[0].resolve()

    raise FileNotFoundError(
        "No video specified and no video found in repository.\n"
        "Please specify a thermal video using:\n"
        "    python demo/thermal_video_demo.py --video <path_to_video.mp4>\n"
        "Or generate a synthetic sample video using:\n"
        "    python demo/generate_sample_thermal.py"
    )


def resolve_model_path(model_arg: str | None) -> str:
    """Locate the model weights file."""
    if model_arg:
        mp = Path(model_arg)
        if mp.is_file():
            return str(mp.resolve())
        raise FileNotFoundError(f"Specified model weights file not found: {model_arg}")

    candidates = [
        PROJECT_ROOT / "models" / "yolo11n.pt",
        PROJECT_ROOT / "yolo11n.pt",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())

    # Fall back to "yolo11n.pt" which Ultralytics can locate or download
    return "models/yolo11n.pt"


# ---------------------------------------------------------------------------
# Main Demo Execution Pipeline
# ---------------------------------------------------------------------------

def run_demo() -> int:
    """Run the thermal video detection demo."""
    args = parse_arguments()

    # 1. Resolve paths
    try:
        video_path = resolve_video_path(args.video)
        model_path = resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 1

    # Resolve output video path if --save or --output was given
    save_path: Path | None = None
    if args.output:
        save_path = Path(args.output).resolve()
    elif args.save:
        save_path = Path(args.save).resolve()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Open Video Source
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open video source: {video_path}\n", file=sys.stderr)
        return 1

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if src_fps <= 0.0 or np.isnan(src_fps):
        src_fps = 25.0  # Safe fallback FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Initialize Thermal Detector (Loaded ONCE)
    print("\n" + "=" * 60)
    print("  IBVAP — THERMAL DETECTION ENGINE DEMO")
    print("=" * 60)
    print(f"Video Source    : {video_path}")
    print(f"Resolution      : {src_w} × {src_h}")
    print(f"Video FPS       : {src_fps:.2f}")
    print(f"Total Frames    : {total_frames if total_frames > 0 else 'Unknown'}")
    print(f"Model Weights   : {model_path}")
    print(f"Requested Dev   : {args.device}")
    print(f"Confidence Thresh: {args.confidence}")
    print(f"GUI Display     : {'Disabled (--no-display)' if args.no_display else 'Enabled'}")
    print(f"Save Recording  : {save_path if save_path else 'Disabled'}")
    print("-" * 60)
    print("Loading ThermalDetector into memory ...")

    t_load_start = time.perf_counter()
    try:
        config = ThermalDetectorConfig(
            model_path=model_path,
            confidence_threshold=args.confidence,
            device=args.device,
            debug=args.debug,
        )
        detector = ThermalDetector(config)
    except (ThermalModelLoadError, Exception) as exc:
        print(f"\n[ERROR] Failed to initialize ThermalDetector: {exc}\n", file=sys.stderr)
        cap.release()
        return 1

    load_time_sec = time.perf_counter() - t_load_start
    print(f"ThermalDetector ready in {load_time_sec:.2f}s  |  Device: {detector.device.upper()}")
    print(f"Supported classes: {sorted(detector.config.allowed_classes) if detector.config.allowed_classes else 'ALL'}")
    print("-" * 60)
    print("Controls: [SPACE] Pause/Resume  |  [S] Save Frame  |  [Q/ESC] Quit")
    print("=" * 60 + "\n")

    # 4. Optional Video Writer Setup
    video_writer: cv2.VideoWriter | None = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(save_path),
            fourcc,
            src_fps,
            (src_w, src_h),
            isColor=True,
        )
        if not video_writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(
                str(save_path),
                fourcc,
                src_fps,
                (src_w, src_h),
                isColor=True,
            )

    # 5. Processing Loop
    window_name = "IBVAP — Thermal Detection Engine"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, src_w), min(720, src_h))

    frame_idx = 0
    total_detections_count = 0
    class_counter: dict[str, int] = defaultdict(int)
    total_inference_sec = 0.0

    overall_start_time = time.perf_counter()
    paused = False

    try:
        while True:
            # Handle Pause State
            if paused and not args.no_display:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("Execution stopped by user.")
                    break
                elif key == ord(" "):
                    paused = False
                    print(f"Resumed at frame {frame_idx}")
                elif key in (ord("s"), ord("S")):
                    snap_path = PROJECT_ROOT / "output" / f"snapshot_frame_{frame_idx:06d}.png"
                    snap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap_path), annotated_frame)
                    print(f"Snapshot saved → {snap_path}")
                continue

            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # ---------------------------------------------------------------
            # 6. Thermal Engine Detection (Calling existing detector.detect())
            # ---------------------------------------------------------------
            t_infer_0 = time.perf_counter()
            try:
                detections: list[ThermalDetection] = detector.detect(frame)
            except (ThermalInvalidFrameError, ThermalInferenceError) as err:
                logger.warning("Frame %d detection error: %s", frame_idx, err)
                detections = []

            infer_duration = time.perf_counter() - t_infer_0
            total_inference_sec += infer_duration
            infer_ms = infer_duration * 1000.0

            # Update stats
            num_dets = len(detections)
            total_detections_count += num_dets
            for d in detections:
                class_counter[d.class_name] += 1

            # Calculate instantaneous end-to-end FPS
            elapsed_so_far = time.perf_counter() - overall_start_time
            running_fps = frame_idx / max(1e-5, elapsed_so_far)

            # ---------------------------------------------------------------
            # 7. Console Logging (Periodic and on detection)
            # ---------------------------------------------------------------
            if (frame_idx % args.print_interval == 0) or (frame_idx == 1):
                print(
                    f"Frame {frame_idx:<5} | Detections: {num_dets:<2} | "
                    f"Inference: {infer_ms:>5.1f} ms | E2E FPS: {running_fps:>5.1f}"
                )
                for i, d in enumerate(detections, 1):
                    x1, y1, x2, y2 = d.bbox
                    temp_info = f" temp={d.temperature_range}" if d.temperature_range is not None else ""
                    print(
                        f"  [{i}] {d.class_name:<11} conf={d.confidence:.2f} "
                        f"bbox=({x1:5.1f}, {y1:5.1f}, {x2:5.1f}, {y2:5.1f}){temp_info}"
                    )

            # ---------------------------------------------------------------
            # 8. Visualization (Reusing existing draw_thermal_detections)
            # ---------------------------------------------------------------
            annotated_frame = draw_thermal_detections(frame, detections)

            # Add HUD Overlay
            annotated_frame = draw_hud_overlay(
                canvas=annotated_frame,
                frame_idx=frame_idx,
                total_frames=total_frames,
                num_detections=num_dets,
                infer_ms=infer_ms,
                instant_fps=running_fps,
                model_name=Path(model_path).name,
                device=detector.device,
                is_paused=paused,
            )

            # 9. Record Output Frame if enabled
            if video_writer:
                video_writer.write(annotated_frame)

            # 10. Display Window & Keyboard Handling
            if not args.no_display:
                cv2.imshow(window_name, annotated_frame)
                # Normal playback delay (matching video fps rate)
                wait_time = max(1, int(1000.0 / src_fps) - int(infer_ms))
                key = cv2.waitKey(max(1, wait_time)) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print(f"\nPlayback interrupted by user at frame {frame_idx}.")
                    break
                elif key == ord(" "):  # SPACE
                    paused = True
                    print(f"\n[PAUSED] Frame {frame_idx} (Press SPACE to resume)")
                elif key in (ord("s"), ord("S")):  # S
                    snap_path = PROJECT_ROOT / "output" / f"snapshot_frame_{frame_idx:06d}.png"
                    snap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap_path), annotated_frame)
                    print(f"Snapshot saved → {snap_path}")

    finally:
        # Cleanup video resources
        cap.release()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # 11. Final Summary Report
    # -----------------------------------------------------------------------
    total_time_sec = time.perf_counter() - overall_start_time
    avg_inference_ms = (total_inference_sec / frame_idx * 1000.0) if frame_idx > 0 else 0.0
    inference_fps = (frame_idx / total_inference_sec) if total_inference_sec > 0 else 0.0
    end_to_end_fps = (frame_idx / total_time_sec) if total_time_sec > 0 else 0.0

    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Frames processed : {frame_idx}")
    print(f"Total time       : {total_time_sec:.2f} sec")
    print(f"Average FPS      : {end_to_end_fps:.2f} (End-to-end)")
    print(f"Inference FPS    : {inference_fps:.2f} (Pure Model Inference)")
    print(f"Average inference: {avg_inference_ms:.2f} ms")
    print(f"Total detections : {total_detections_count}")

    if class_counter:
        print("Detections by class:")
        for cls_name, count in sorted(class_counter.items()):
            print(f"  - {cls_name:<14}: {count}")

    if save_path and save_path.is_file():
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"Annotated video  : {save_path} ({file_size_mb:.2f} MB)")

    print("=" * 60)
    print(
        "\nNote: The current demo validates the thermal detection pipeline and model\n"
        "integration. The default YOLO11n COCO weights are used as a baseline because\n"
        "dedicated thermal-trained weights are not included. Detection accuracy on thermal\n"
        "imagery has not been established by this demo.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
