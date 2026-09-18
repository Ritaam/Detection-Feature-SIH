"""
pipeline — Frame scheduling and module orchestration.

Planned responsibilities
------------------------
* Read frames from camera ingestion layer (RTSP / file / webcam).
* Schedule frames through the detection → tracking → analytics pipeline.
* Handle concurrency, frame dropping, and back-pressure.
* Route ``list[TrackedDetection]`` to downstream modules (ANPR, face, analytics).
* Collect events and forward to the alert / API layer.

Status: NOT YET IMPLEMENTED.

Sub-modules (planned)
---------------------
ingestion.py        : Frame source abstraction (RTSP, file, webcam)
orchestrator.py     : Main pipeline loop
frame_queue.py      : Thread-safe / async frame queue
router.py           : Route detections to appropriate sub-modules

Dependencies (planned)
----------------------
* src.detection   → Detector
* src.tracking    → Tracker
* src.anpr        → ANPRProcessor
* src.face        → FaceProcessor
* src.analytics   → BehaviourAnalyser
* src.api_client  → EventClient
"""

# NOT YET IMPLEMENTED
