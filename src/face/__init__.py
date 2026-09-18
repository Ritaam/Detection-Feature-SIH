"""
face — Face detection and recognition module.

Planned responsibilities
------------------------
* Detect faces within person bounding boxes returned by the main detector.
* Extract face embeddings using a lightweight model (e.g. FaceNet, ArcFace).
* Match embeddings against a watchlist database.
* Return structured ``FaceMatch`` results.

Pipeline position
-----------------
::

    Detector
        ↓
    list[Detection]  (person class only)
        ↓
    Face module
        ↓
    list[FaceMatch]

Status: NOT YET IMPLEMENTED.

Sub-modules (planned)
---------------------
face_detector.py    : Locate faces within person crops
embedder.py         : Extract 128/512-d face embeddings
matcher.py          : Match embeddings against watchlist
models.py           : FaceMatch dataclass
config.py           : FaceConfig dataclass

Dependencies (planned)
----------------------
* src.detection.models.Detection  (data model only)
* deepface / insightface / facenet-pytorch  (TBD)
"""

# NOT YET IMPLEMENTED
