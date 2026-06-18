"""Model-asset registry + lazy downloader. [BUILD]

Frozen perception models (pose/face landmarkers, face detector) are downloaded on first
use and cached under models/. Nothing here is trained; these are pretrained weights.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"
_ASSETS: dict[str, tuple[str, str]] = {
    "pose_landmarker": (
        f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "models/pose_landmarker_lite.task",
    ),
    "face_landmarker": (
        f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "models/face_landmarker.task",
    ),
    "face_detector": (
        f"{_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "models/blaze_face_short_range.tflite",
    ),
}


def ensure(name: str) -> str:
    """Return the local path to a model asset, downloading it on first use."""
    url, dst = _ASSETS[name]
    path = Path(dst)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


def is_available(name: str) -> bool:
    return Path(_ASSETS[name][1]).exists()
